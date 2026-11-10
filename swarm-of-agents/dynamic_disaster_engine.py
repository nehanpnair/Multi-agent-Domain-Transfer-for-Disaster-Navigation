"""
dynamic_disaster_engine.py
--------------------------
Temporal disaster engine for the Tokyo multi-agent SNN simulation.

Disaster types and their progression models
-------------------------------------------

EARTHQUAKE
  t=0       : mainshock — apply_disaster() at full severity, radius r
  t=1..5    : aftershock cascade — 2-4 secondary epicentres, severity 0.3-0.6×main,
              radius 0.4-0.7×main, damaging previously undamaged edges
  t=6..end  : rubble spread — nodes at damage>0.5 propagate +0.1 damage to
              neighbours each step, capped at 1.0

FLOOD
  t=0       : initial inundation zone, severity 0.4, radius r
  each step : radius expands by FLOOD_SPREAD_RATE (default 150m/step),
              severity increases by 0.03/step up to 0.9
  rivers    : flood preferentially spreads to lower-elevation neighbours
              (approximated by proximity to low-lying roads)

FIRE
  t=0       : ignition points (1-3 nodes), severity 0.2
  each step : fire spreads along connected road edges to k nearest unburned
              neighbours — spread probability = f(wind_direction, severity)
  extinguish: fire_station within 800m reduces spread probability by 0.5

COMPOUND
  t=0       : earthquake mainshock
  t=15      : flood triggered (damaged drainage → inundation)
  t=30      : fire ignition from ruptured gas lines in high-damage zones

Each engine step returns a DisasterEvent list so SwarmSimulation can:
  1. Update graph._node_damage and _disabled_edges via graph.apply_damage_patch()
  2. Push newly blocked nodes into shared_memory["blocked_nodes"]
  3. Log events for the evaluator

Usage
-----
    engine = DynamicDisasterEngine(graph, disaster_type="earthquake",
                                   centre=(35.68, 139.76),
                                   radius_m=2000, severity=0.7,
                                   seed=42)
    engine.initialise()             # t=0 mainshock

    for step in range(200):
        events = engine.tick(step)  # returns list[DisasterEvent]
        for ev in events:
            print(ev)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Haversine helper (local, no import from snn_core to keep engine standalone)
# ---------------------------------------------------------------------------

def _hav(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Event record — one per change applied to the graph
# ---------------------------------------------------------------------------

@dataclass
class DisasterEvent:
    step:         int
    event_type:   str          # "aftershock" | "flood_expand" | "fire_spread"
                               # | "rubble_spread" | "compound_trigger"
    centre:       tuple        # (lat, lon) of this sub-event
    radius_m:     float
    severity:     float
    nodes_newly_damaged: list  # node IDs whose damage increased this step
    edges_newly_disabled: int  # count of edges disabled this step
    description:  str = ""

    def __str__(self):
        return (f"[step {self.step:>3}] {self.event_type:<18} "
                f"centre=({self.centre[0]:.4f},{self.centre[1]:.4f}) "
                f"r={self.radius_m:.0f}m sev={self.severity:.2f} "
                f"→ {len(self.nodes_newly_damaged)} nodes, "
                f"{self.edges_newly_disabled} edges   {self.description}")


# ---------------------------------------------------------------------------
# DynamicDisasterEngine
# ---------------------------------------------------------------------------

class DynamicDisasterEngine:
    """
    Manages temporal disaster progression and mutates the graph each step.

    Parameters
    ----------
    graph           HybridTokyoGraph instance (mutated in place)
    disaster_type   "earthquake" | "flood" | "fire" | "compound"
    centre          (lat, lon) of primary epicentre
    radius_m        initial disaster radius in metres
    severity        initial severity [0.0, 1.0]
    seed            RNG seed for reproducibility
    wind_bearing_deg  wind direction for fire spread (degrees, 0=N, 90=E)
    """

    FLOOD_SPREAD_RATE   = 150.0   # metres per step
    FLOOD_SEVERITY_RATE = 0.03    # severity increase per step
    FLOOD_SEVERITY_MAX  = 0.90
    FLOOD_MAX_RADIUS_FACTOR = 2.0 # flood never grows beyond this multiple of radius_m

    FIRE_SPREAD_K       = 3       # max neighbours fire jumps to per step
    FIRE_SPREAD_BASE    = 0.45    # base probability of spreading to one neighbour
    FIRE_STATION_RADIUS = 800.0   # m within which a fire station suppresses spread
    FIRE_MAX_RADIUS_FACTOR = 1.5  # fire cannot spread beyond this multiple of radius_m from centre

    RUBBLE_SPREAD_RATE  = 0.10    # damage increment per step for rubble propagation
    RUBBLE_START_STEP   = 6       # rubble propagation begins this many steps after mainshock
    RUBBLE_WINDOW       = 30      # steps over which rubble keeps spreading (then it settles)
    RUBBLE_DECAY        = 0.95    # a neighbour can reach at most source_damage * decay, so
                                  # spread dies out after a few hops instead of flooding the map

    AFTERSHOCK_WINDOW   = 5       # steps during which aftershocks can occur
    AFTERSHOCK_PROB     = 0.55    # probability of an aftershock each step in window
    AFTERSHOCK_SEV_MIN  = 0.30    # severity = uniform(min, max) × main_severity
    AFTERSHOCK_SEV_MAX  = 0.65
    AFTERSHOCK_R_MIN    = 0.40    # radius = uniform(min, max) × main_radius
    AFTERSHOCK_R_MAX    = 0.75
    AFTERSHOCK_OFFSET   = 0.025   # max lat/lon offset of aftershock epicentre (°)

    COMPOUND_FLOOD_STEP = 15      # step at which flood is triggered (compound mode)
    COMPOUND_FIRE_STEP  = 30      # step at which fire is triggered (compound mode)

    def __init__(
        self,
        graph,
        disaster_type:   str   = "earthquake",
        centre:          tuple = (35.68, 139.76),
        radius_m:        float = 2000.0,
        severity:        float = 0.7,
        seed:            Optional[int] = None,
        wind_bearing_deg: float = 45.0,
    ):
        self.graph          = graph
        self.disaster_type  = disaster_type
        self.centre         = centre
        self.radius_m       = radius_m
        self.severity       = severity
        self.wind_bearing_deg = wind_bearing_deg

        self._rng = random.Random(seed)

        # Mutable state
        self._flood_radius   = radius_m
        self._flood_severity = min(severity * 0.6, 0.4)   # floods start weaker

        self._fire_nodes: set = set()      # currently burning node IDs
        self._burned_nodes: set = set()    # nodes that have been on fire

        self._aftershocks_fired = 0
        self._mainshock_summary: dict = {}

        # Compound state
        self._compound_flood_active = False
        self._compound_fire_active  = False

        self.events: list[DisasterEvent] = []   # full log, all steps

    # -----------------------------------------------------------------------
    # Initialise — apply t=0 mainshock
    # -----------------------------------------------------------------------

    def initialise(self) -> DisasterEvent:
        """
        Apply the t=0 event and return its DisasterEvent.
        Must be called before the first tick().
        """
        self.graph.reset_disaster()

        if self.disaster_type in ("earthquake", "compound"):
            summary = self.graph.apply_disaster(
                "earthquake", self.centre, self.radius_m, self.severity
            )
            self._mainshock_summary = summary
            ev = DisasterEvent(
                step=0,
                event_type="mainshock",
                centre=self.centre,
                radius_m=self.radius_m,
                severity=self.severity,
                nodes_newly_damaged=list(self.graph._node_damage.keys()),
                edges_newly_disabled=summary.get("edges_disabled", 0),
                description=f"M-shock: {summary['nodes_affected']} nodes, "
                            f"pop={summary['population_at_risk']:,}",
            )

        elif self.disaster_type == "flood":
            self._flood_radius   = self.radius_m
            self._flood_severity = min(self.severity * 0.5, 0.4)
            summary = self.graph.apply_disaster(
                "flood", self.centre, self._flood_radius, self._flood_severity
            )
            ev = DisasterEvent(
                step=0,
                event_type="flood_init",
                centre=self.centre,
                radius_m=self._flood_radius,
                severity=self._flood_severity,
                nodes_newly_damaged=list(self.graph._node_damage.keys()),
                edges_newly_disabled=summary.get("edges_disabled", 0),
                description="Initial inundation zone",
            )

        elif self.disaster_type == "fire":
            ev = self._ignite_fire(step=0)

        else:
            raise ValueError(f"Unknown disaster_type: {self.disaster_type!r}. "
                             "Choose: earthquake | flood | fire | compound")

        self.events.append(ev)
        return ev

    # -----------------------------------------------------------------------
    # Tick — advance one simulation step
    # -----------------------------------------------------------------------

    def tick(self, step: int) -> list[DisasterEvent]:
        """
        Advance disaster state by one step.
        Returns list of DisasterEvent objects applied this step (may be empty).
        Call once per simulation step BEFORE agents move.
        """
        new_events: list[DisasterEvent] = []

        if self.disaster_type == "earthquake":
            new_events.extend(self._tick_earthquake(step))

        elif self.disaster_type == "flood":
            new_events.extend(self._tick_flood(step))

        elif self.disaster_type == "fire":
            new_events.extend(self._tick_fire(step))

        elif self.disaster_type == "compound":
            new_events.extend(self._tick_earthquake(step))
            # Trigger flood at step 15
            if step == self.COMPOUND_FLOOD_STEP:
                self._compound_flood_active = True
                ev = self._trigger_compound_flood(step)
                if ev:
                    new_events.append(ev)
            # Trigger fire at step 30
            if step == self.COMPOUND_FIRE_STEP:
                self._compound_fire_active = True
                ev = self._ignite_fire(step)
                if ev:
                    new_events.append(ev)
            # Advance compound sub-disasters
            if self._compound_flood_active and step > self.COMPOUND_FLOOD_STEP:
                new_events.extend(self._tick_flood(step))
            if self._compound_fire_active and step > self.COMPOUND_FIRE_STEP:
                new_events.extend(self._tick_fire(step))

        self.events.extend(new_events)
        return new_events

    # -----------------------------------------------------------------------
    # Earthquake: aftershocks + rubble spread
    # -----------------------------------------------------------------------

    def _tick_earthquake(self, step: int) -> list[DisasterEvent]:
        evs = []

        # Aftershock phase (steps 1 to AFTERSHOCK_WINDOW)
        if 1 <= step <= self.AFTERSHOCK_WINDOW:
            if self._rng.random() < self.AFTERSHOCK_PROB:
                ev = self._apply_aftershock(step)
                if ev:
                    evs.append(ev)

        # Rubble propagation phase (steps RUBBLE_START_STEP onward)
        if self.RUBBLE_START_STEP <= step < self.RUBBLE_START_STEP + self.RUBBLE_WINDOW:
            ev = self._propagate_rubble(step)
            if ev:
                evs.append(ev)

        return evs

    def _apply_aftershock(self, step: int) -> Optional[DisasterEvent]:
        # Pick a random offset from the mainshock centre
        dlat = self._rng.uniform(-self.AFTERSHOCK_OFFSET, self.AFTERSHOCK_OFFSET)
        dlon = self._rng.uniform(-self.AFTERSHOCK_OFFSET, self.AFTERSHOCK_OFFSET)
        centre = (self.centre[0] + dlat, self.centre[1] + dlon)

        sev = self.severity * self._rng.uniform(self.AFTERSHOCK_SEV_MIN, self.AFTERSHOCK_SEV_MAX)
        rad = self.radius_m * self._rng.uniform(self.AFTERSHOCK_R_MIN,   self.AFTERSHOCK_R_MAX)

        before_damage   = set(self.graph._node_damage.keys())
        before_disabled = len(self.graph._disabled_edges)

        self.graph.apply_disaster("earthquake", centre, rad, sev)

        after_damage   = set(self.graph._node_damage.keys())
        newly_damaged  = list(after_damage - before_damage)
        new_edges_disabled = len(self.graph._disabled_edges) - before_disabled

        self._aftershocks_fired += 1
        return DisasterEvent(
            step=step,
            event_type="aftershock",
            centre=centre,
            radius_m=rad,
            severity=sev,
            nodes_newly_damaged=newly_damaged,
            edges_newly_disabled=max(0, new_edges_disabled),
            description=f"AS#{self._aftershocks_fired}",
        )

    def _propagate_rubble(self, step: int) -> Optional[DisasterEvent]:
        """
        Nodes with damage > 0.5 spread +RUBBLE_SPREAD_RATE to their neighbours.
        Models secondary collapse / debris blocking adjacent roads.
        """
        high_damage = [
            nid for nid, d in self.graph._node_damage.items() if d > 0.5
        ]
        if not high_damage:
            return None

        newly_damaged = []
        for nid in high_damage:
            # Damage passed on is capped below the source's own damage, so each
            # hop is weaker and propagation terminates (source <= 0.5 stops).
            cap = min(self.graph._node_damage[nid] * self.RUBBLE_DECAY, 1.0)
            for nb_id, edge in self.graph._adj.get(nid, []):
                cur = self.graph._node_damage.get(nb_id, 0.0)
                new_dmg = min(cur + self.RUBBLE_SPREAD_RATE, cap)
                if new_dmg > cur:
                    self.graph._node_damage[nb_id] = new_dmg
                    # report first damage and crossings of the 0.6 "blocked" line
                    if cur == 0.0 or (cur <= 0.6 < new_dmg):
                        newly_damaged.append(nb_id)
                    # Disable edge if neighbour now critically damaged
                    if new_dmg > 0.6:          # same line agents treat as blocked
                        key = frozenset({nid, nb_id})
                        self.graph._disabled_edges.add(key)

        if not newly_damaged:
            return None

        return DisasterEvent(
            step=step,
            event_type="rubble_spread",
            centre=self.centre,
            radius_m=self.radius_m,
            severity=self.RUBBLE_SPREAD_RATE,
            nodes_newly_damaged=newly_damaged,
            edges_newly_disabled=0,
            description=f"Rubble from {len(high_damage)} high-damage nodes",
        )

    # -----------------------------------------------------------------------
    # Flood: expanding radius + rising severity
    # -----------------------------------------------------------------------

    def _tick_flood(self, step: int) -> list[DisasterEvent]:
        # Expand radius and severity
        max_radius = self.FLOOD_MAX_RADIUS_FACTOR * self.radius_m
        new_radius = min(self._flood_radius + self.FLOOD_SPREAD_RATE, max_radius)
        new_severity = min(
            self._flood_severity + self.FLOOD_SEVERITY_RATE,
            self.FLOOD_SEVERITY_MAX,
        )
        if (new_radius == self._flood_radius
                and new_severity == self._flood_severity):
            return []          # flood has reached its maximum extent: nothing changes

        # Only apply the annular ring (radius delta) to avoid re-damaging inner nodes
        # We use apply_disaster on the full new circle; the method clamps damage
        # values with min(current, new), so inner nodes only increase if severity rose.
        before_damage   = set(self.graph._node_damage.keys())
        before_disabled = len(self.graph._disabled_edges)

        self.graph.apply_disaster(
            "flood",
            self.centre if not self._compound_flood_active else self._compound_flood_centre,
            new_radius,
            new_severity,
        )

        self._flood_radius   = new_radius
        self._flood_severity = new_severity

        after_damage   = set(self.graph._node_damage.keys())
        newly_damaged  = list(after_damage - before_damage)
        new_edges_disabled = len(self.graph._disabled_edges) - before_disabled

        return [DisasterEvent(
            step=step,
            event_type="flood_expand",
            centre=self.centre,
            radius_m=new_radius,
            severity=new_severity,
            nodes_newly_damaged=newly_damaged,
            edges_newly_disabled=max(0, new_edges_disabled),
            description=f"Flood r={new_radius:.0f}m sev={new_severity:.2f}",
        )]

    def _trigger_compound_flood(self, step: int) -> Optional[DisasterEvent]:
        """
        In compound mode the flood originates from high-damage zones,
        not the mainshock centre — models ruptured water mains.
        Pick the densest cluster of high-damage nodes as the flood origin.
        """
        high_damage = [
            nid for nid, d in self.graph._node_damage.items() if d > 0.6
        ]
        if not high_damage:
            self._compound_flood_centre = self.centre
        else:
            # Centroid of high-damage cluster
            lats = [self.graph._nodes[n]["lat"] for n in high_damage if n in self.graph._nodes]
            lons = [self.graph._nodes[n]["lon"] for n in high_damage if n in self.graph._nodes]
            if lats:
                self._compound_flood_centre = (
                    sum(lats) / len(lats),
                    sum(lons) / len(lons),
                )
            else:
                self._compound_flood_centre = self.centre

        self._flood_radius   = self.radius_m * 0.4
        self._flood_severity = 0.3

        summary = self.graph.apply_disaster(
            "flood", self._compound_flood_centre,
            self._flood_radius, self._flood_severity
        )
        return DisasterEvent(
            step=step,
            event_type="compound_trigger",
            centre=self._compound_flood_centre,
            radius_m=self._flood_radius,
            severity=self._flood_severity,
            nodes_newly_damaged=list(self.graph._node_damage.keys()),
            edges_newly_disabled=summary.get("edges_disabled", 0),
            description="Compound flood triggered by earthquake damage",
        )

    # -----------------------------------------------------------------------
    # Fire: ignition + wind-driven spread
    # -----------------------------------------------------------------------

    def _ignite_fire(self, step: int) -> DisasterEvent:
        """
        Ignite 1-3 nodes in the high-damage zone (or random road nodes if no
        damage exists yet, e.g. standalone fire scenario).
        """
        high_damage = [
            nid for nid, d in self.graph._node_damage.items()
            if d > 0.4 and nid in self.graph._adj
        ]

        if high_damage:
            k = min(self._rng.randint(1, 3), len(high_damage))
            ignition_nodes = self._rng.sample(high_damage, k)
        else:
            # No prior damage — pick random road nodes near centre
            candidates = self.graph.get_nodes_in_radius(
                self.centre[0], self.centre[1], self.radius_m, "road_node"
            )
            candidates = [n for n in candidates if n["id"] in self.graph._adj]
            k = min(self._rng.randint(1, 3), len(candidates))
            ignition_nodes = [n["id"] for n in self._rng.sample(candidates, k)] if candidates else []

        for nid in ignition_nodes:
            self.graph._node_damage[nid] = max(
                self.graph._node_damage.get(nid, 0.0), 0.2
            )
            self._fire_nodes.add(nid)
            self._burned_nodes.add(nid)

        return DisasterEvent(
            step=step,
            event_type="fire_ignition",
            centre=self.centre,
            radius_m=self.radius_m,
            severity=0.2,
            nodes_newly_damaged=ignition_nodes,
            edges_newly_disabled=0,
            description=f"Fire ignition at {len(ignition_nodes)} node(s)",
        )

    def _tick_fire(self, step: int) -> list[DisasterEvent]:
        if not self._fire_nodes:
            return []

        # Wind vector: unit vector in wind_bearing direction
        bear = math.radians(self.wind_bearing_deg)
        wind_lat = math.cos(bear)
        wind_lon = math.sin(bear)

        # Check for nearby fire stations (suppress spread)
        fire_stations = self.graph.get_fire_stations()

        newly_damaged = []
        new_fire_nodes: set = set()
        edges_disabled = 0

        for burning_nid in list(self._fire_nodes):
            bn = self.graph._nodes.get(burning_nid)
            if bn is None:
                continue

            # Intensify burning node
            cur_dmg = self.graph._node_damage.get(burning_nid, 0.2)
            self.graph._node_damage[burning_nid] = min(cur_dmg + 0.15, 1.0)

            # Check fire station suppression
            suppressed = any(
                _hav(bn["lat"], bn["lon"], fs["lat"], fs["lon"])
                < self.FIRE_STATION_RADIUS
                for fs in fire_stations
            )
            spread_prob = self.FIRE_SPREAD_BASE * (0.5 if suppressed else 1.0)

            # Spread to neighbours — bias toward wind direction
            neighbours = self.graph._adj.get(burning_nid, [])
            if not neighbours:
                continue

            # Score neighbours by wind alignment
            scored = []
            for nb_id, edge in neighbours:
                if nb_id in self._burned_nodes:
                    continue
                nb = self.graph._nodes.get(nb_id)
                if nb is None:
                    continue
                if _hav(self.centre[0], self.centre[1], nb["lat"], nb["lon"]) \
                        > self.FIRE_MAX_RADIUS_FACTOR * self.radius_m:
                    continue
                dlat = nb["lat"] - bn["lat"]
                dlon = nb["lon"] - bn["lon"]
                mag  = math.sqrt(dlat**2 + dlon**2)
                if mag > 0:
                    alignment = (dlat/mag)*wind_lat + (dlon/mag)*wind_lon
                else:
                    alignment = 0.0
                # Wind alignment in [-1,1] → spread multiplier [0.5, 1.5]
                p = spread_prob * (1.0 + 0.5 * alignment)
                scored.append((nb_id, min(p, 0.95)))

            scored.sort(key=lambda x: -x[1])
            candidates = scored[:self.FIRE_SPREAD_K]

            for nb_id, p in candidates:
                if self._rng.random() < p:
                    self.graph._node_damage[nb_id] = max(
                        self.graph._node_damage.get(nb_id, 0.0), 0.2
                    )
                    new_fire_nodes.add(nb_id)
                    self._burned_nodes.add(nb_id)
                    newly_damaged.append(nb_id)

                    # Disable edges to/from burned node
                    key = frozenset({burning_nid, nb_id})
                    if key not in self.graph._disabled_edges:
                        self.graph._disabled_edges.add(key)
                        edges_disabled += 1

        # Nodes that burned fully are no longer "active fire" but remain damaged
        self._fire_nodes = {
            nid for nid in (self._fire_nodes | new_fire_nodes)
            if self.graph._node_damage.get(nid, 0.0) < 0.95
        }

        if not newly_damaged and not new_fire_nodes:
            return []

        return [DisasterEvent(
            step=step,
            event_type="fire_spread",
            centre=self.centre,
            radius_m=self.radius_m,
            severity=0.5,
            nodes_newly_damaged=newly_damaged,
            edges_newly_disabled=edges_disabled,
            description=f"Fire front: {len(self._fire_nodes)} active nodes",
        )]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def newly_blocked_nodes(self, step: int) -> set:
        """
        Return the set of node IDs whose damage crossed 0.6 this step.
        Used by SwarmSimulation to push into shared_memory["blocked_nodes"].
        """
        return {
            nid for nid, d in self.graph._node_damage.items()
            if d > 0.6
        }

    def summary(self) -> dict:
        """Aggregate statistics across all disaster events."""
        all_newly_damaged = set()
        total_edge_changes = 0
        event_counts: dict[str, int] = {}

        for ev in self.events:
            all_newly_damaged.update(ev.nodes_newly_damaged)
            total_edge_changes += ev.edges_newly_disabled
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1

        return {
            "disaster_type":       self.disaster_type,
            "total_events":        len(self.events),
            "event_counts":        event_counts,
            "total_nodes_damaged": len(all_newly_damaged),
            "total_edges_disabled":total_edge_changes,
            "final_node_damage":   len(self.graph._node_damage),
            "final_edges_disabled":len(self.graph._disabled_edges),
        }
