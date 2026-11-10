import heapq
import math
import random
from collections import deque

from snn_core_v2 import SNNDecision, haversine_distance


class SNNAgent:
    """
    Three-tier memory navigation agent — v3 fixes:

    Cycle window   CYCLE_DETECT_LEN raised 6→10. The Agent-1 failure in run 2
                   was a 4-node cycle that repeated but the rolling window (maxlen=20)
                   was being appended BEFORE detection, so the second repeat started
                   at position -8 not -4, failing the exact equality check.
                   Larger window catches more cycle shapes.

    Near-goal taboo  _near_goal() threshold reduced 200→100m. At 200m the agent
                   was shrinking its taboo window too early, causing oscillation
                   in the 100–200m band where the goal is not yet directly reachable.

    Position history  Append moved to AFTER cycle check so the current node is
                   not already in the window when we test — fixes off-by-one in
                   cycle equality check.

    Stall threshold  MAX_STEPS_WITHOUT_PROGRESS reduced 15→10. Agents in run 3
                   spent 20+ steps clearly oscillating before the stall handler
                   fired.
    """

    # Node damage above this is treated as impassable (matches the 0.6
    # "blocked" threshold used by shared memory and the disaster engine).
    BLOCK_DAMAGE_THRESHOLD = 0.6

    PLAN_MAX_EXPANSIONS = 40000   # A* budget per (re)plan
    CELL_DEG = 0.002              # ~200 m coverage grid

    def __init__(self, graph, start_node, goal_node, enforce_passability=True, rng=None,
                 role="navigator"):
        self.graph        = graph
        self.role         = role
        self.clock        = 0            # current sim step, set by the simulation
        self.stationary   = False        # hubs sense and communicate but do not move
        self.sense_hops   = 1            # 2 = scouts see two hops out
        self.use_lookahead = True        # score candidates by what is known beyond them
        self.use_planner  = False        # follow a route planned on KNOWN hazards
        self.plan         = None
        self.plan_events  = []           # [{step, reason, ok}] for replans
        self.stats = {"msgs_received": 0, "blocked_learned_self": 0,
                      "blocked_learned_msg": 0, "replans": 0,
                      "plan_invalidations": 0, "goal_changes": 0}
        self.learned_at   = {}           # blocked node -> (step, "self" | "msg" | "board")
        self.cell_counts  = {}           # coverage grid cell -> visits

        # Per-agent random stream: an agent's stochastic choices no longer depend
        # on how many other agents exist or in what order they act.
        self.rng          = rng if rng is not None else random
        self.enforce_passability = enforce_passability
        self.death_reason = None          # "hazard" | "trapped" | None
        # What this agent physically sensed (1-hop) on its latest step.
        self.last_observation = {"blocked_nodes": set(), "danger": {}}
        self.current_node = start_node
        self.goal_node    = goal_node

        self.snn = SNNDecision(rng=self.rng)

        self.path         = [start_node]
        self.alive        = True
        self.reached_goal = False
        self.prev_node    = None
        self.visited      = set([start_node])
        self.visit_counts = {start_node: 1}
        self.shared_memory = None
        self.local_memory  = {"danger_nodes": {}, "blocked_nodes": set()}

        # Tier 1: short-term taboo
        self.taboo_set   = set()
        self.taboo_queue = deque()
        self.TABOO_SIZE      = 6
        self.TABOO_SIZE_NEAR = 4

        # Tier 2: permanent blacklist
        self.blacklist = set()

        # Tier 3: cycle detection — FIX: raised window length
        self.position_history = deque(maxlen=30)   # was 20
        self.CYCLE_DETECT_LEN = 10                 # was 6

        # Oscillation counter
        self.oscillation_counter = 0
        self.OSCILLATION_LIMIT   = 6   # was 8 — fire sooner

        # Progress tracking — FIX: tighter stall threshold
        self._last_dist_to_goal       = None
        self._moving_away_counter     = 0
        self._steps_without_progress  = 0
        self.MAX_STEPS_WITHOUT_PROGRESS = 10   # was 15

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #

    def _taboo_add(self, node):
        near = self._near_goal()
        size = self.TABOO_SIZE_NEAR if near else self.TABOO_SIZE
        self.taboo_queue.append(node)
        self.taboo_set.add(node)
        while len(self.taboo_queue) > size:
            evicted = self.taboo_queue.popleft()
            if evicted not in self.taboo_queue:
                self.taboo_set.discard(evicted)

    def _taboo_reset(self):
        self.taboo_set.clear()
        self.taboo_queue.clear()

    def _blacklist_cycle_nodes(self, cycle_len=2):
        """Permanently ban ALL nodes in the detected cycle window."""
        hist = list(self.position_history)
        cycle_nodes = set(hist[-cycle_len:]) if cycle_len <= len(hist) else set(hist)
        cycle_nodes.add(self.current_node)
        if self.prev_node:
            cycle_nodes.add(self.prev_node)
        self.blacklist.update(cycle_nodes)
        self._taboo_reset()

    def _detect_cycle(self):
        """Return (True, cycle_len) if recent history contains a repeated cycle."""
        hist = list(self.position_history)
        n    = len(hist)
        for cycle_len in range(2, self.CYCLE_DETECT_LEN + 1):
            if n < cycle_len * 2:
                continue
            if hist[-cycle_len:] == hist[-(cycle_len * 2):-cycle_len]:
                return True, cycle_len
        return False, 0

    def _near_goal(self):
        # FIX: reduced threshold 200→100m so taboo shrinkage doesn't trigger
        # too early and cause oscillation in the 100-200m approach band
        try:
            d = haversine_distance(
                self.graph._nodes[self.current_node]['lat'],
                self.graph._nodes[self.current_node]['lon'],
                self.graph._nodes[self.goal_node]['lat'],
                self.graph._nodes[self.goal_node]['lon'],
            )
            return d < 100   # was 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def get_all_neighbors(self):
        """Every adjacent node, ignoring closures (used for sensing/metrics)."""
        return [nid for nid, _ in self.graph._adj.get(self.current_node, [])]

    def is_passable(self, nid):
        """A move is legal only if the edge is open and the target is not
        destroyed. Both are locally observable from the current node."""
        if not self.graph.is_edge_passable(self.current_node, nid):
            return False
        return self.graph.get_node_damage(nid) <= self.BLOCK_DAMAGE_THRESHOLD

    def get_neighbors(self):
        """Neighbours the agent can actually move to."""
        nbrs = self.get_all_neighbors()
        if not self.enforce_passability:
            return nbrs
        return [n for n in nbrs if self.is_passable(n)]

    def sense(self):
        """1-hop local sensing. Records closed neighbours and damaged
        neighbours; the simulation decides what gets published to the swarm."""
        blocked, danger = set(), {}
        for nid in self.get_all_neighbors():
            dmg = self.graph.get_node_damage(nid)
            if dmg > 0.3:
                danger[nid] = dmg
            if (not self.graph.is_edge_passable(self.current_node, nid)
                    or dmg > self.BLOCK_DAMAGE_THRESHOLD):
                blocked.add(nid)
            if self.sense_hops >= 2:
                # Longer-range sensing (scouts): node damage two hops out.
                for m, _ in self.graph._adj.get(nid, ()):
                    d2 = self.graph.get_node_damage(m)
                    if d2 > 0.3:
                        danger[m] = max(d2, danger.get(m, 0.0))
                    if d2 > self.BLOCK_DAMAGE_THRESHOLD:
                        blocked.add(m)
        for nid, dmg in danger.items():
            self.local_memory["danger_nodes"][nid] = max(
                dmg, self.local_memory["danger_nodes"].get(nid, 0.0))
        for nid in blocked:
            self._learn_blocked(nid, "self")
        self.last_observation = {"blocked_nodes": blocked, "danger": danger}
        return self.last_observation

    # ------------------------------------------------------------------ #
    # Knowledge: what the agent has sensed itself, been told, or read from
    # the shared board. Never the ground-truth damage map.
    # ------------------------------------------------------------------ #

    def _learn_blocked(self, nid, source):
        lm = self.local_memory["blocked_nodes"]
        if nid in lm:
            return False
        lm.add(nid)
        self.learned_at[nid] = (self.clock, source)
        self.stats["blocked_learned_self" if source == "self" else "blocked_learned_msg"] += 1
        return True

    def receive(self, msg):
        """Merge a delivered Message into local knowledge. Returns #new blocked nodes."""
        self.stats["msgs_received"] += 1
        new = 0
        for nid in msg.payload.get("blocked", ()):
            new += self._learn_blocked(nid, "msg")
        for nid, dmg in msg.payload.get("danger", {}).items():
            cur = self.local_memory["danger_nodes"].get(nid, 0.0)
            if dmg > cur:
                self.local_memory["danger_nodes"][nid] = dmg
        return new

    def knows_blocked(self, nid):
        if nid in self.local_memory["blocked_nodes"]:
            return True
        return bool(self.shared_memory) and nid in self.shared_memory["blocked_nodes"]

    def known_danger(self, nid):
        d = self.local_memory["danger_nodes"].get(nid, 0.0)
        if self.shared_memory:
            d = max(d, self.shared_memory["danger_nodes"].get(nid, 0.0))
        return d

    def _exit_risk(self, nid):
        """(mean risk over onward exits, all-exits-blocked) for candidate `nid`,
        from KNOWN hazards only. Risk of an exit = 1 if known blocked, else its
        known danger level."""
        exits = [m for m, _ in self.graph._adj.get(nid, ()) if m != self.current_node]
        if not exits:
            return 0.0, False
        risks = [1.0 if self.knows_blocked(m) else self.known_danger(m) for m in exits]
        return sum(risks) / len(risks), all(self.knows_blocked(m) for m in exits)

    # ------------------------------------------------------------------ #
    # Planning on known hazards (event-based replanning)
    # ------------------------------------------------------------------ #

    def _astar_known(self):
        g, goal = self.graph, self.goal_node
        if self.knows_blocked(goal):
            return None
        gd = g._nodes[goal]
        def h(n):
            nd = g._nodes[n]
            return haversine_distance(nd['lat'], nd['lon'], gd['lat'], gd['lon'])
        start = self.current_node
        best = {start: 0.0}
        parent = {}
        heap = [(h(start), 0.0, start)]
        expanded = 0
        while heap:
            _, d, u = heapq.heappop(heap)
            if u == goal:
                path = [u]
                while u in parent:
                    u = parent[u]
                    path.append(u)
                return path[::-1]
            if d > best.get(u, float('inf')):
                continue
            expanded += 1
            if expanded > self.PLAN_MAX_EXPANSIONS:
                return None
            for v, edge in g._adj.get(u, ()):
                if v != goal and self.knows_blocked(v):
                    continue
                # known damage makes a node costlier but not forbidden
                nd_ = d + edge["length"] * (1.0 + 2.0 * self.known_danger(v))
                if nd_ < best.get(v, float('inf')):
                    best[v] = nd_
                    parent[v] = u
                    heapq.heappush(heap, (nd_ + h(v), nd_, v))
        return None

    def _ensure_plan(self):
        """Keep a route to the goal that avoids KNOWN blocked nodes. Replans only
        when there is no plan, the agent has left it, or a node ahead has become
        known-blocked (the event the old stall counter was trying to proxy)."""
        reason = None
        if self.plan is None:
            reason = "no_plan"
        else:
            try:
                pos = self.plan.index(self.current_node)
            except ValueError:
                pos = -1
            if pos < 0:
                reason = "off_plan"
            elif any(self.knows_blocked(n) for n in self.plan[pos + 1:]):
                reason = "invalidated"
                self.stats["plan_invalidations"] += 1
        if reason is None:
            return
        self.plan = self._astar_known()
        self.stats["replans"] += 1
        self.plan_events.append({"step": self.clock, "reason": reason,
                                 "ok": self.plan is not None, "node": self.current_node})

    def _plan_next(self):
        if not self.plan:
            return None
        try:
            i = self.plan.index(self.current_node)
        except ValueError:
            return None
        return self.plan[i + 1] if i + 1 < len(self.plan) else None

    def set_goal(self, node):
        """Give the agent a new goal (waypoint or reassignment) and clear every
        piece of goal-relative state so old progress/taboo logic cannot misfire."""
        self.goal_node = node
        self.reached_goal = False
        self._last_dist_to_goal = None
        self._steps_without_progress = 0
        self._moving_away_counter = 0
        self.oscillation_counter = 0
        self._taboo_reset()
        self.snn.reset_momentum()
        self.plan = None
        self.stats["goal_changes"] += 1

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #

    def build_features(self, neighbors):
        features     = []
        current_data = self.graph._nodes[self.current_node]
        goal_data    = self.graph._nodes[self.goal_node]
        current_dist = haversine_distance(
            current_data['lat'], current_data['lon'],
            goal_data['lat'],   goal_data['lon']
        )

        ph = {}
        if self.shared_memory:
            ph = self.shared_memory.get("pheromones", {})
        plan_next = None
        if self.use_planner:
            self._ensure_plan()
            plan_next = self._plan_next()

        for nid in neighbors:
            node_data = self.graph._nodes[nid]
            dist = haversine_distance(
                node_data['lat'], node_data['lon'],
                goal_data['lat'], goal_data['lon']
            )

            hazard = self.graph.get_node_damage(nid)
            if nid in self.local_memory["danger_nodes"]:
                hazard = min(1.0, hazard + 0.3 * self.local_memory["danger_nodes"][nid])
            if self.shared_memory and nid in self.shared_memory["blocked_nodes"]:
                hazard = max(hazard, 0.95)

            count           = self.visit_counts.get(nid, 0)
            visited_penalty = math.log1p(count)
            if nid in list(self.path)[-5:]:
                visited_penalty += 1.0

            is_prev_node  = 1.0 if nid == self.prev_node else 0.0
            is_new        = 1.0 if nid not in self.visited else 0.0
            progress      = max(0.0, current_dist - dist)
            taboo_penalty = 1.0 if nid in self.taboo_set else 0.0
            bl_penalty    = 1.0 if nid in self.blacklist else 0.0

            two_hop_goal = 0.0
            if nid != self.goal_node:
                if self.goal_node in [n for n, _ in self.graph._adj.get(nid, [])]:
                    two_hop_goal = 1.0

            pheromone = ph.get(nid, 0.0)

            lookahead, dead_end = 0.0, 0.0
            if self.use_lookahead and nid != self.goal_node:
                lookahead, all_blocked = self._exit_risk(nid)
                dead_end = 1.0 if all_blocked else 0.0

            features.append({
                "lookahead":    lookahead,
                "dead_end":     dead_end,
                "on_plan":      1.0 if (plan_next is not None and nid == plan_next) else 0.0,
                "node":         nid,
                "dist":         dist,
                "hazard":       hazard,
                "visited":      visited_penalty,
                "progress":     progress,
                "is_prev_node": is_prev_node,
                "is_new":       is_new,
                "taboo":        taboo_penalty,
                "blacklist":    bl_penalty,
                "two_hop_goal": two_hop_goal,
                "pheromone":    pheromone,
                "node_data":    node_data,
            })

        return features

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self):
        if not self.alive or self.reached_goal:
            return

        self.sense()
        if self.stationary:
            hz = self.graph.get_node_damage(self.current_node)
            if hz > 0.9:
                self.alive = False
                self.death_reason = "hazard"
            return
        neighbors = self.get_neighbors()
        if not neighbors:
            self.alive = False
            self.death_reason = "trapped"
            return

        current_pos  = self.graph._nodes[self.current_node]
        goal_pos     = self.graph._nodes[self.goal_node]

        # FIX: append BEFORE detection so the window reflects position
        # ENTERING this step, not the one we're about to decide on.
        # Old code appended current_node then immediately tested — the
        # second repeat of a 4-node cycle ended at index -4 in the window
        # but the first copy was already shifted by the append, so the
        # equality check hist[-4:] == hist[-8:-4] compared different slices.
        self.position_history.append(self.current_node)
        is_cycling, cycle_len = self._detect_cycle()

        if is_cycling:
            self._blacklist_cycle_nodes(cycle_len)
            escape = [n for n in neighbors if n not in self.blacklist]
            if not escape:
                self.blacklist.discard(self.current_node)
                escape = [n for n in neighbors if n not in self.blacklist]
            if not escape:
                escape = neighbors

            unvisited_escape = [n for n in escape if n not in self.visited]
            next_node = self.rng.choice(unvisited_escape if unvisited_escape else escape)
            self.oscillation_counter = 0
            self._execute_move(next_node, current_pos, goal_pos)
            return

        # --- Build features ---
        features = self.build_features(neighbors)

        non_bl = [f for f in features if f['blacklist'] < 0.5]
        if not non_bl:
            non_bl = features

        non_taboo = [f for f in non_bl if f['taboo'] < 0.5]
        if not non_taboo:
            best = min(non_bl, key=lambda f: (f['visited'], f['dist']))
            self._taboo_reset()
            self._execute_move(best['node'], current_pos, goal_pos)
            return

        # --- Normal scoring decision ---
        next_node = self.snn.decide(
            neighbor_features=non_bl,
            current_pos=current_pos,
            goal_pos=goal_pos,
            goal_node=self.goal_node
        )

        if next_node is None:
            self.alive = False
            return

        # --- Oscillation counter ---
        if next_node == self.prev_node:
            self.oscillation_counter += 1
            if self.oscillation_counter >= self.OSCILLATION_LIMIT:
                self._blacklist_cycle_nodes(cycle_len=2)
                escape = [n for n in neighbors
                          if n not in self.blacklist and n != self.prev_node]
                if not escape:
                    escape = [n for n in neighbors if n != self.prev_node] or neighbors
                next_node = self.rng.choice(escape)
                self.oscillation_counter = 0
        else:
            self.oscillation_counter = 0

        # --- Stall handler ---
        if self._last_dist_to_goal is not None:
            nxt_dist = haversine_distance(
                self.graph._nodes[next_node]['lat'],
                self.graph._nodes[next_node]['lon'],
                goal_pos['lat'], goal_pos['lon']
            )
            if nxt_dist >= self._last_dist_to_goal - 10:
                self._steps_without_progress += 1
            else:
                self._steps_without_progress = 0

            if self._steps_without_progress >= self.MAX_STEPS_WITHOUT_PROGRESS:
                self._taboo_reset()
                self.snn.reset_momentum()
                self._steps_without_progress = 0
                unvisited_far = [f for f in non_bl
                                 if f['node'] not in self.visited
                                 and f['node'] != self.prev_node]
                if unvisited_far:
                    next_node = max(unvisited_far, key=lambda f: f['dist'])['node']
                else:
                    pool = [f for f in non_bl if f['node'] != self.prev_node]
                    if pool:
                        next_node = self.rng.choice(pool)['node']

        self._execute_move(next_node, current_pos, goal_pos)

    # ------------------------------------------------------------------ #
    # Execute move
    # ------------------------------------------------------------------ #

    def _execute_move(self, next_node, current_pos, goal_pos):
        prev = self.current_node
        self.current_node = next_node
        self.path.append(next_node)

        self.snn.update_momentum(
            current_pos=self.graph._nodes[prev],
            next_pos=self.graph._nodes[next_node]
        )

        self.prev_node = prev
        self.visited.add(next_node)
        self.visit_counts[next_node] = self.visit_counts.get(next_node, 0) + 1
        nd = self.graph._nodes[next_node]
        cell = (int(nd['lat'] / self.CELL_DEG), int(nd['lon'] / self.CELL_DEG))
        self.cell_counts[cell] = self.cell_counts.get(cell, 0) + 1
        self._taboo_add(prev)

        nxt_dist = haversine_distance(
            self.graph._nodes[next_node]['lat'],
            self.graph._nodes[next_node]['lon'],
            goal_pos['lat'], goal_pos['lon']
        )
        if self._last_dist_to_goal is not None and nxt_dist > self._last_dist_to_goal:
            self._moving_away_counter += 1
            if self._moving_away_counter >= 5:
                self.snn.reset_momentum()
                self._moving_away_counter = 0
        else:
            self._moving_away_counter = 0
        self._last_dist_to_goal = nxt_dist

        hazard = self.graph.get_node_damage(next_node)
        if hazard > 0.3:
            self.local_memory["danger_nodes"][next_node] = hazard
        if hazard > 0.9:
            self.alive = False
            self.death_reason = "hazard"
            return
        if next_node == self.goal_node:
            self.reached_goal = True
            if self.role == "navigator":
                print(f"✓ Agent reached goal! Path length: {len(self.path)}")
