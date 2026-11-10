"""
roles.py
--------
Specialised roles and coordination for the swarm.

Before this module every agent was an identical SNNAgent; the Mapping / Risk /
Resource / Routing agents existed only as Gemini prompts in run.py. They are now
real swarm roles, each defined by what it senses, how it moves, what it sends
and what it does with what it receives:

  run.py agent   Swarm role   Behaviour
  -------------  -----------  ---------------------------------------------------
  Routing        NAVIGATOR    Has a goal. Plans a route on KNOWN hazards (A*),
                              replans only when a hazard on the route becomes
                              known, and is the only role counted in success.
  Mapping        SCOUT        No fixed goal: patrols waypoints, sees two hops
                              out, favours unexplored ground. Its job is to
                              discover hazards and report them.
  Risk           ASSESSOR     Stationary hub with a large antenna. Collects every
                              hazard report it hears and re-broadcasts the
                              consolidated picture as "advisory" messages.
  Resource       ALLOCATOR    Stationary hub that owns the shelter pool: it
                              chooses initial goals under capacity limits and,
                              in adaptive mode, reassigns navigators whose goal
                              it learns is blocked. Inactive without shelters.

"Adaptive coordination" (defined)
---------------------------------
Static   roles and goal assignments are fixed at t = 0; scouts patrol by
         coverage only; the assessor advises on a fixed schedule.
Adaptive the coordinator re-evaluates every `period` steps using only what the
         swarm has REPORTED (never the ground-truth damage map):
           1. scouts steer toward the known hazard front instead of patrolling
              uniformly;
           2. the allocator reassigns navigators whose goal has become known-
              blocked to the nearest shelter with spare capacity;
           3. the assessor advises as soon as it learns anything new rather
              than on a timer.
         Every decision is appended to the simulation's coordination_log.

Role mix is assigned by index with navigators first, so changing the number of
scouts or hubs never changes navigator i's task or random stream.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from feasibility import _stream
from snn_core_v2 import haversine_distance


class Role(str, Enum):
    NAVIGATOR = "navigator"
    SCOUT     = "scout"
    ASSESSOR  = "assessor"
    ALLOCATOR = "allocator"


@dataclass(frozen=True)
class RoleSpec:
    role:          Role
    radius_mult:   float = 1.0      # radio range multiplier
    sense_hops:    int   = 1
    stationary:    bool  = False
    use_planner:   bool  = False
    use_lookahead: bool  = True
    weights:       dict  = field(default_factory=dict)   # SNNDecision overrides


# Used when roles=None: the homogeneous baseline (the pre-role behaviour).
HOMOGENEOUS_SPEC = RoleSpec(Role.NAVIGATOR)

ROLE_SPECS = {
    # w_plan=100 so the planned route outweighs the taboo/blacklist/stall logic that
    # otherwise traps agents against walls. Chosen on the synthetic wall scenario
    # (16 held-out seeds: 0.46 -> 0.58-0.61 success vs w_plan=12); re-tune on Tokyo.
    Role.NAVIGATOR: RoleSpec(Role.NAVIGATOR, use_planner=True, weights={"w_plan": 100.0}),
    Role.SCOUT:     RoleSpec(Role.SCOUT, sense_hops=2,
                             weights={"w_goal": 4.0, "w_progress": 4.0,
                                      "w_explore": 4.0, "w_visited": -3.0}),
    Role.ASSESSOR:  RoleSpec(Role.ASSESSOR, radius_mult=3.0, stationary=True),
    Role.ALLOCATOR: RoleSpec(Role.ALLOCATOR, radius_mult=3.0, stationary=True),
}


@dataclass(frozen=True)
class CoordinationConfig:
    mode:            str   = "static"     # "static" | "adaptive"
    period:          int   = 10           # adaptive re-evaluation interval (steps)
    advisory_period: int   = 5            # static advisory schedule (steps)
    scout_range_m:   tuple = (600.0, 1800.0)
    scout_candidates: int  = 24

    def __post_init__(self):
        if self.mode not in ("static", "adaptive"):
            raise ValueError(f"mode must be 'static' or 'adaptive', got {self.mode!r}")


# ---------------------------------------------------------------------------
# Role mix
# ---------------------------------------------------------------------------

def default_role_counts(n: int, with_allocator: bool = False) -> dict:
    scouts    = max(1, round(0.2 * n)) if n >= 5 else 0
    assessor  = 1 if n >= 6 else 0
    allocator = 1 if (with_allocator and n >= 8) else 0
    return {Role.NAVIGATOR: n - scouts - assessor - allocator, Role.SCOUT: scouts,
            Role.ASSESSOR: assessor, Role.ALLOCATOR: allocator}


def resolve_roles(roles, n: int, with_allocator: bool = False) -> list:
    """roles: None (homogeneous) | "default" | {role: count} | [role, ...]."""
    if roles is None:
        return [Role.NAVIGATOR] * n
    if roles == "default":
        counts = default_role_counts(n, with_allocator)
    elif isinstance(roles, dict):
        counts = {Role(k): v for k, v in roles.items()}
    else:
        out = [Role(r) for r in roles]
        if len(out) != n:
            raise ValueError(f"roles has {len(out)} entries, need {n}")
        return out
    out = []
    for r in (Role.NAVIGATOR, Role.SCOUT, Role.ASSESSOR, Role.ALLOCATOR):
        out += [r] * counts.get(r, 0)
    if len(out) != n:
        raise ValueError(f"role counts sum to {len(out)}, need {n}")
    return out


# ---------------------------------------------------------------------------
# Resource allocation
# ---------------------------------------------------------------------------

class Allocator:
    """Shelter pool with capacity. Pure bookkeeping; the ALLOCATOR agent decides
    when to use it and what it is allowed to know."""

    def __init__(self, shelters: list, capacity: int):
        self.shelters = list(shelters)
        self.capacity = capacity
        self.load = Counter()

    def pick(self, graph, node, is_blocked=lambda s: False, comp=None,
             exclude=(), max_dist=None) -> Optional[int]:
        n = graph._nodes[node]
        ranked = []
        for s in self.shelters:
            if s in exclude or self.load[s] >= self.capacity or is_blocked(s):
                continue
            if comp is not None and (s not in comp or node not in comp
                                     or comp[s] != comp[node]):
                continue
            sn = graph._nodes[s]
            d = haversine_distance(n["lat"], n["lon"], sn["lat"], sn["lon"])
            if max_dist is not None and d > max_dist:
                continue
            ranked.append((d, s))
        return min(ranked)[1] if ranked else None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """Runs role behaviours for one DynamicSwarmSimulation. Reads only agent
    knowledge and bus traffic, never the engine or the damage map."""

    def __init__(self, sim, cfg: CoordinationConfig):
        self.sim = sim
        self.cfg = cfg
        self.log: list = []
        self._scout_rng: dict = {}
        self._pending_assign: dict = {}      # navigator idx -> goal in flight

    @property
    def adaptive(self):
        return self.cfg.mode == "adaptive"

    def _event(self, kind, **detail):
        self.log.append({"step": self.sim.current_step, "type": kind, **detail})

    # ----------------------------- scouts ----------------------------------

    def scout_waypoint(self, idx):
        sim, a = self.sim, self.sim.agents[idx]
        rng = self._scout_rng.setdefault(idx, _stream(sim.seed, "scout", idx))
        cur = sim.graph._nodes[a.current_node]
        lo, hi = self.cfg.scout_range_m
        centroid = self._hazard_centroid(a) if self.adaptive else None

        best, best_score = None, -math.inf
        for _ in range(self.cfg.scout_candidates):
            c = rng.choice(sim.cands)
            if c == a.current_node or a.knows_blocked(c):
                continue
            cn = sim.graph._nodes[c]
            d = haversine_distance(cur["lat"], cur["lon"], cn["lat"], cn["lon"])
            if not (lo <= d <= hi):
                continue
            cell = (int(cn["lat"] / a.CELL_DEG), int(cn["lon"] / a.CELL_DEG))
            score = -a.cell_counts.get(cell, 0) + rng.random() * 1e-3
            if centroid is not None:
                score -= haversine_distance(cn["lat"], cn["lon"], *centroid) / 500.0
            if score > best_score:
                best, best_score = c, score
        if best is None:
            best = rng.choice(sim.cands)
        a.set_goal(best)
        if self.adaptive and centroid is not None:
            self._event("scout_retarget", agent=idx, waypoint=best,
                        toward=[round(centroid[0], 5), round(centroid[1], 5)])

    def _hazard_centroid(self, agent):
        """Mean position of hazards the agent knows about (own + told + board),
        or the publicly announced alert centre before anything is known."""
        sim = self.sim
        known = set(agent.local_memory["blocked_nodes"])
        if sim.comm.shared_memory:
            known |= sim.shared_memory["blocked_nodes"]
        if known:
            pts = [sim.graph._nodes[n] for n in list(known)[:2000] if n in sim.graph._nodes]
            if pts:
                return (sum(p["lat"] for p in pts) / len(pts),
                        sum(p["lon"] for p in pts) / len(pts))
        return sim.alert_centre

    # ---------------------------- assessor ---------------------------------

    def assessor_step(self, t):
        sim = self.sim
        for idx, a in enumerate(sim.agents):
            if a.role != Role.ASSESSOR.value or not a.alive:
                continue
            if not self.adaptive and t % self.cfg.advisory_period != 0:
                continue
            new_b = a.local_memory["blocked_nodes"] - sim._radio_rep_blocked[idx]
            new_d = {n: d for n, d in a.local_memory["danger_nodes"].items()
                     if d > sim._radio_rep_danger[idx].get(n, 0.0) + 0.05}
            if not new_b and not new_d:
                continue
            msg = sim.bus.send(idx, "advisory",
                               {"blocked": tuple(new_b), "danger": new_d},
                               t, sim._in_range(idx))
            if msg is not None:
                sim._radio_rep_blocked[idx] |= new_b
                sim._radio_rep_danger[idx].update(new_d)

    # ---------------------------- allocator --------------------------------

    def allocator_step(self, t):
        """Adaptive only: reassign navigators whose goal the allocator knows is
        blocked. Needs a shelter pool and the navigator in radio range."""
        sim = self.sim
        if not self.adaptive or sim.allocator is None or not sim.comm.local_messaging:
            return
        if t % self.cfg.period != 0:
            return
        for hub_idx, hub in enumerate(sim.agents):
            if hub.role != Role.ALLOCATOR.value or not hub.alive:
                continue
            for j in sim._in_range(hub_idx):
                nav = sim.agents[j]
                if nav.role != Role.NAVIGATOR.value or nav.reached_goal:
                    continue
                if not hub.knows_blocked(nav.goal_node):
                    continue
                if self._pending_assign.get(j) is not None:
                    continue
                new = sim.allocator.pick(
                    sim.graph, nav.current_node, is_blocked=hub.knows_blocked,
                    exclude={nav.goal_node})
                if new is None:
                    self._event("reassign_failed", agent=j, goal=nav.goal_node)
                    continue
                sim.allocator.load[nav.goal_node] -= 1
                sim.allocator.load[new] += 1
                self._pending_assign[j] = new
                sim.bus.send(hub_idx, "assign", {"goal": new, "target": j}, t, [j])
                self._event("reassign", agent=j, old_goal=nav.goal_node, new_goal=new,
                            by=hub_idx)

    def on_assign(self, recipient, msg):
        a = self.sim.agents[recipient]
        goal = msg.payload["goal"]
        self._pending_assign.pop(recipient, None)
        if a.alive and not a.reached_goal and a.role == Role.NAVIGATOR.value:
            a.set_goal(goal)
            a.task_info = {**getattr(a, "task_info", {}), "goal": goal, "reassigned": True}
