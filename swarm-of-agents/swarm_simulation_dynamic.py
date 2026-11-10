"""
swarm_simulation_dynamic.py
----------------------------
Dynamic-environment swarm simulation.

Extends SwarmSimulation v3 with per-step calls to DynamicDisasterEngine,
giving agents a genuinely changing environment to navigate.

Key differences from swarm_simulation_v3.py
--------------------------------------------
  Per-step disaster  DynamicDisasterEngine.tick() is called BEFORE agents
                     move each step. New blocked nodes are pushed into
                     shared_memory["blocked_nodes"] so agents that look ahead
                     can route around them.

  Path invalidation  After each engine tick, any agent whose NEXT intended
                     node has just been blocked has its taboo + blacklist reset
                     and is forced to re-plan (modelled by clearing momentum
                     and flagging _steps_without_progress=MAX so the stall
                     handler fires on the next step() call).

  Adaptation metrics For each agent we track:
                       - reroute_events  : how many times its path was
                         invalidated by a new disaster event
                       - adaptation_lag  : steps spent in the stall/cycle
                         handler after a path invalidation
                     These feed directly into the research paper's
                     "SNN adaptation rate" metric.

  Event log          self.disaster_log records every DisasterEvent with the
                     step it fired — useful for correlating with agent
                     behaviour in the evaluator.

Usage
-----
    from hybrid_spatial_index import HybridTokyoGraph
    from swarm_simulation_dynamic import DynamicSwarmSimulation

    graph = HybridTokyoGraph("dataset/tokyo_graph.json")
    graph.build_index()

    sim = DynamicSwarmSimulation(
        graph,
        num_agents=10,
        disaster_type="earthquake",       # or "flood" | "fire" | "compound"
        disaster_centre=(35.68, 139.76),
        disaster_radius_m=2000,
        disaster_severity=0.7,
        seed=42,
    )
    sim.run(steps=200)
    sim.summary()
"""

import random
import math
from agent_snn_v3 import SNNAgent, haversine_distance
from dynamic_disaster_engine import DynamicDisasterEngine
from feasibility import (agent_rng, spawn_rng, passable_components,
                         candidate_nodes, sample_feasible_task, validate_task,
                         SpawnError)
from comms import CommConfig, MessageBus, BOARD
from roles import (Role, RoleSpec, ROLE_SPECS, HOMOGENEOUS_SPEC, CoordinationConfig,
                   Coordinator, Allocator, resolve_roles)


class DynamicSwarmSimulation:
    """
    Multi-agent SNN swarm with a temporally evolving disaster environment.
    """

    def __init__(
        self,
        graph,
        num_agents:            int   = 10,
        min_start_goal_distance: float = 1000,
        max_start_goal_distance: float = 4000,
        # Disaster engine parameters
        disaster_type:         str   = "earthquake",
        disaster_centre:       tuple = (35.68, 139.76),
        disaster_radius_m:     float = 2000.0,
        disaster_severity:     float = 0.7,
        wind_bearing_deg:      float = 45.0,
        seed:                  int   = None,
        # Task control
        tasks:                 list  = None,   # [(start, goal) | {"start","goal"}, ...]
        require_feasible:      bool  = True,
        max_optimal_hops:      int   = None,
        spawn_attempts:        int   = 2000,
        # Communication and roles
        comm:                  CommConfig = None,
        roles                  = None,    # None | "default" | {role: n} | [role, ...]
        role_specs:            dict  = None,   # override ROLE_SPECS entries
        coordination:          CoordinationConfig = None,
        shelters:              list  = None,   # shared goal pool (ALLOCATOR role)
        shelter_capacity:      int   = None,
        alert_centre:          tuple = None,   # publicly announced epicentre
        engine                 = None,         # inject a custom engine (tests)
    ):
        self.graph      = graph
        self.num_agents = num_agents
        self.min_start_goal_distance = min_start_goal_distance
        self.max_start_goal_distance = max_start_goal_distance

        # No global seeding: every agent owns a private stream derived from
        # (seed, agent_index), and the engine owns its own, so results do not
        # depend on swarm size, agent order, or anything else using `random`.
        self.seed             = seed
        self.require_feasible = require_feasible
        self.max_optimal_hops = max_optimal_hops
        self.spawn_attempts   = spawn_attempts
        self._given_tasks     = tasks
        self.tasks: list      = []     # task info per agent (start, goal, optimal_m, ...)

        # ---- Disaster engine ----
        self.engine = engine or DynamicDisasterEngine(
            graph            = graph,
            disaster_type    = disaster_type,
            centre           = disaster_centre,
            radius_m         = disaster_radius_m,
            severity         = disaster_severity,
            seed             = seed,
            wind_bearing_deg = wind_bearing_deg,
        )

        # ---- Communication, roles, coordination ----
        self.comm         = comm or CommConfig()
        self.bus          = MessageBus(self.comm)
        self.coord_cfg    = coordination or CoordinationConfig()
        self.coordinator  = Coordinator(self, self.coord_cfg)
        self.alert_centre = alert_centre if alert_centre is not None else disaster_centre
        self.role_list    = resolve_roles(roles, num_agents, with_allocator=bool(shelters))
        self._specialised = roles is not None
        self._role_specs  = {**ROLE_SPECS, **{Role(k): v for k, v in (role_specs or {}).items()}}
        self.allocator    = None
        self._shelters    = list(shelters) if shelters else None
        self._shelter_capacity = shelter_capacity
        self.cands        = []
        self.board_visible: dict = {}     # blocked node -> step it became readable
        self._board_rep_blocked: dict = {}
        self._board_rep_danger:  dict = {}
        self._radio_rep_blocked: dict = {}
        self._radio_rep_danger:  dict = {}
        self._contacts: set = set()

        # ---- Shared memory ----
        self.shared_memory = {
            "blocked_nodes": set(),
            "danger_nodes":  {},
            "pheromones":    {},
        }

        self.PHEROMONE_DECAY   = 0.97
        self.PHEROMONE_DEPOSIT = 0.5

        # ---- Disaster and adaptation logs ----
        self.disaster_log: list = []          # [(step, DisasterEvent), ...]
        # node -> (step, agent_idx) of the FIRST time any agent sensed it blocked.
        # Shared memory only ever contains nodes that appear here.
        self.detection_log: dict = {}
        self.current_step: int = 0
        self._agent_reroute_events: dict = {} # agent_idx → count
        self._agent_adaptation_lag: dict = {} # agent_idx → total stall steps after reroute

        self.agents   = []
        self._goal_step: dict = {}
        self.coordination_log = self.coordinator.log

        # Apply t=0 mainshock, THEN spawn agents so spawn distances
        # reflect the post-disaster graph passability.
        init_ev = self.engine.initialise()
        self.disaster_log.append((0, init_ev))
        print(f"\n{init_ev}")

        self._spawn_agents()

    # -----------------------------------------------------------------------
    # Spawn (feasible tasks, per-agent seeded)
    # -----------------------------------------------------------------------

    def _spec(self, role):
        return self._role_specs[role] if self._specialised else HOMOGENEOUS_SPEC

    def _apply_spec(self, agent, spec):
        agent.role          = spec.role.value
        agent.sense_hops    = spec.sense_hops
        agent.stationary    = spec.stationary
        agent.use_planner   = spec.use_planner
        agent.use_lookahead = spec.use_lookahead
        agent.radius_mult   = spec.radius_mult
        agent.snn.apply_weights(spec.weights)

    def _make_view(self):
        """What one agent can read of the shared channels. Disabled channels are
        empty, so an ablation cannot leak through an agent reading the board."""
        sm = self.shared_memory
        c = self.comm
        return {
            "blocked_nodes": sm["blocked_nodes"] if c.shared_memory else frozenset(),
            "danger_nodes":  sm["danger_nodes"]  if c.shared_memory else {},
            "pheromones":    sm["pheromones"]    if c.pheromones    else {},
        }

    def _sample_shelter_task(self, i, comp, cands):
        rng = spawn_rng(self.seed, i)
        min_d = 0.25 * self.min_start_goal_distance
        for _ in range(self.spawn_attempts):
            s = rng.choice(cands)
            g = self.allocator.pick(self.graph, s, comp=comp,
                                    max_dist=self.max_start_goal_distance)
            if g is None or g == s or self._distance(s, g) < min_d:
                continue
            info = validate_task(self.graph, s, g, comp, self.max_optimal_hops)
            if info is not None:
                return info
        raise SpawnError(f"No feasible shelter task for agent {i}")

    def _spawn_agents(self):
        """
        Create agents with feasible tasks.

        Feasible = start and goal intact after the t=0 damage, connected by a
        passable route (same rules the agent obeys), and optionally within
        `max_optimal_hops`. Pass `tasks=` to reuse identical tasks across
        experiment arms; otherwise task i is drawn from spawn_rng(seed, i).
        With `shelters=`, navigators are assigned a shelter by the allocator
        (nearest with spare capacity); hubs are placed at the centroid of the
        navigators' starts; scouts get their first waypoint.
        """
        comp = passable_components(self.graph)
        cands = candidate_nodes(self.graph, comp)
        self.cands = cands
        given = self._given_tasks
        if given is not None and len(given) < self.num_agents:
            raise ValueError(f"tasks has {len(given)} entries, need {self.num_agents}")

        n_nav = sum(r == Role.NAVIGATOR for r in self.role_list)
        if self._shelters:
            cap = self._shelter_capacity or (-(-n_nav // len(self._shelters)) + 1)
            self.allocator = Allocator(self._shelters, cap)

        for i in range(self.num_agents):
            role = self.role_list[i]
            if given is not None:
                t = given[i]
                s_node, g_node = (t["start"], t["goal"]) if isinstance(t, dict) else t
                info = validate_task(self.graph, s_node, g_node, comp,
                                     self.max_optimal_hops)
                if info is None:
                    if self.require_feasible:
                        raise ValueError(
                            f"Task {i} ({s_node} -> {g_node}) is infeasible in this "
                            f"scenario at t=0. Pass require_feasible=False to allow it.")
                    info = {"start": s_node, "goal": g_node, "feasible": False}
            elif self.allocator is not None and role == Role.NAVIGATOR:
                info = self._sample_shelter_task(i, comp, cands)
            else:
                info = sample_feasible_task(
                    self.graph, spawn_rng(self.seed, i), comp, cands,
                    self.min_start_goal_distance, self.max_start_goal_distance,
                    self.max_optimal_hops, self.spawn_attempts)
            info.setdefault("feasible", True)

            agent = SNNAgent(self.graph, info["start"], info["goal"],
                             rng=agent_rng(self.seed, i), role=role.value)
            self._apply_spec(agent, self._spec(role))
            agent.shared_memory = self._make_view()
            agent.task_info = info
            self.agents.append(agent)
            self.tasks.append(info)

            if self.allocator is not None and role == Role.NAVIGATOR \
                    and info["goal"] in self.allocator.shelters:
                self.allocator.load[info["goal"]] += 1

            self._agent_reroute_events[i] = 0
            self._agent_adaptation_lag[i] = 0
            self._board_rep_blocked[i] = set()
            self._board_rep_danger[i]  = {}
            self._radio_rep_blocked[i] = set()
            self._radio_rep_danger[i]  = {}

            opt = info.get("optimal_m")
            opt_s = f"{opt:.0f}m route" if opt is not None else "route n/a"
            print(f"Agent {i:2d} [{role.value:<9}] | Start: {info['start']} → "
                  f"Goal: {info['goal']} | "
                  f"{info.get('straight_m', float('nan')):.0f}m straight, {opt_s}")

        self._place_hubs(cands)
        for i, a in enumerate(self.agents):
            if a.role == Role.SCOUT.value:
                self.coordinator.scout_waypoint(i)

    def _place_hubs(self, cands):
        hubs = [i for i, a in enumerate(self.agents) if a.stationary]
        if not hubs:
            return
        ref = [a for a in self.agents if a.role == Role.NAVIGATOR.value] or self.agents
        clat = sum(self.graph._nodes[a.current_node]["lat"] for a in ref) / len(ref)
        clon = sum(self.graph._nodes[a.current_node]["lon"] for a in ref) / len(ref)
        node = min(cands, key=lambda n: (self.graph._nodes[n]["lat"] - clat) ** 2
                                        + (self.graph._nodes[n]["lon"] - clon) ** 2)
        for i in hubs:
            a = self.agents[i]
            a.current_node = node
            a.path = [node]
            a.visited = {node}
            a.visit_counts = {node: 1}

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

    def run(self, steps: int = 200):
        nav = Role.NAVIGATOR.value
        for t in range(1, steps + 1):
            print(f"\n{'='*60}")
            print(f"STEP {t}")
            print(f"{'='*60}")
            self.current_step = t
            for a in self.agents:
                a.clock = t

            # 1. Advance disaster — BEFORE agents move
            new_events = self.engine.tick(t)
            for ev in new_events:
                self.disaster_log.append((t, ev))
                print(f"  ⚡ DISASTER: {ev}")

            # 2. Agents adjacent to newly blocked nodes are alerted (1-hop sensing).
            #    Shared memory is NOT updated from the engine.
            if new_events:
                self._invalidate_blocked_paths(t, new_events)

            # 3. Adaptive coordination (reads knowledge only; sends via the bus)
            self.coordinator.allocator_step(t)

            # 4. Move agents
            for i, agent in enumerate(self.agents):
                if not agent.alive:
                    continue
                if agent.reached_goal:
                    if agent.role == Role.SCOUT.value:
                        self.coordinator.scout_waypoint(i)     # next patrol leg
                    else:
                        continue

                is_nav = agent.role == nav
                if is_nav:
                    self._rescue_near_goal_stuck(agent)

                prev_node = agent.current_node
                prev_progress = agent._steps_without_progress

                agent.step()

                if (is_nav and self._agent_reroute_events[i] > 0
                        and agent._steps_without_progress > prev_progress):
                    self._agent_adaptation_lag[i] += 1

                if agent.current_node != prev_node:
                    cd  = self.graph._nodes[agent.current_node]
                    gd  = self.graph._nodes[agent.goal_node]
                    dtg = haversine_distance(cd['lat'], cd['lon'], gd['lat'], gd['lon'])
                    status = "ALIVE"
                    if not agent.alive:      status = "DEAD"
                    elif agent.reached_goal: status = "GOAL!"
                    print(f"  Agent {i:2d} ({agent.role[:3]}): {prev_node} → "
                          f"{agent.current_node} | DTG: {dtg:6.0f}m | {status}")

                self._publish_observation(i, agent)

                if is_nav and agent.reached_goal:
                    if self.comm.pheromones:
                        self._deposit_pheromones(agent)
                    self._goal_step[i] = t

            # 5. Communication phase: send, then deliver everything due
            self._comms_phase(t)
            if self.comm.pheromones:
                self._evaporate_pheromones()

            active_after = sum(1 for a in self.agents
                               if a.role == nav and a.alive and not a.reached_goal)
            print(f"\n  Active navigators: {active_after}")
            if active_after == 0 and any(a.role == nav for a in self.agents):
                print("\n  All navigators finished.")
                break

    # -----------------------------------------------------------------------
    # Dynamic disaster integration
    # -----------------------------------------------------------------------

    def _own_observation(self, agent):
        """(blocked set, danger dict) this agent physically sensed this step,
        including the node it stands on. Nothing it was merely told."""
        obs = agent.last_observation
        blocked = set(obs["blocked_nodes"])
        danger = dict(obs["danger"])
        node = agent.current_node
        hazard = agent.graph.get_node_damage(node)
        if hazard > agent.BLOCK_DAMAGE_THRESHOLD:
            blocked.add(node)
        if hazard > 0.0:
            danger[node] = max(hazard, danger.get(node, 0.0))
        return blocked, danger

    def _publish_observation(self, idx, agent):
        """
        Record first detections and post what `agent` sensed to the shared board
        (if enabled). Only the delta since its last post is sent, through the bus,
        so board latency and traffic are both modelled.
        """
        blocked, danger = self._own_observation(agent)
        for nid in blocked:
            self.detection_log.setdefault(nid, (self.current_step, idx))

        if not self.comm.shared_memory:
            return
        new_b = blocked - self._board_rep_blocked[idx]
        rep_d = self._board_rep_danger[idx]
        new_d = {n: d for n, d in danger.items() if d > rep_d.get(n, 0.0) + 0.05}
        if not new_b and not new_d:
            return
        self.bus.send(idx, "board", {"blocked": tuple(new_b), "danger": new_d},
                      self.current_step, [BOARD], delay=self.comm.board_delay_steps)
        self._board_rep_blocked[idx] |= new_b
        rep_d.update(new_d)

    # -----------------------------------------------------------------------
    # Communication phase
    # -----------------------------------------------------------------------

    def _range_m(self, idx):
        return self.comm.radius_m * getattr(self.agents[idx], "radius_mult", 1.0)

    def _in_range(self, idx):
        """Alive agents that can exchange messages with `idx` right now."""
        a = self.graph._nodes[self.agents[idx].current_node]
        ri = self._range_m(idx)
        out = []
        for j, b in enumerate(self.agents):
            if j == idx or not b.alive:
                continue
            bn = self.graph._nodes[b.current_node]
            d = haversine_distance(a['lat'], a['lon'], bn['lat'], bn['lon'])
            if d <= max(ri, self._range_m(j)):
                out.append(j)
        return out

    def _comms_phase(self, t):
        c = self.comm
        if c.local_messaging:
            self.coordinator.assessor_step(t)

            # hazard deltas: own-sensed information only (no re-broadcast of
            # other agents' reports; relaying is the assessor's job)
            for i, a in enumerate(self.agents):
                if not a.alive or a.stationary:
                    continue
                blocked, danger = self._own_observation(a)
                new_b = blocked - self._radio_rep_blocked[i]
                rep_d = self._radio_rep_danger[i]
                new_d = {n: d for n, d in danger.items() if d > rep_d.get(n, 0.0) + 0.05}
                if not new_b and not new_d:
                    continue
                msg = self.bus.send(i, "hazard",
                                    {"blocked": tuple(new_b), "danger": new_d},
                                    t, self._in_range(i))
                if msg is not None:       # nobody in range: keep it for later
                    self._radio_rep_blocked[i] |= new_b
                    rep_d.update(new_d)

            if c.sync_on_contact:
                self._sync_contacts(t)

        for recipient, msg in self.bus.deliver(t, is_alive=lambda r: self.agents[r].alive):
            self._apply_delivery(recipient, msg, t)

    def _sync_contacts(self, t):
        """When two agents come into range, each sends the other everything it
        knows. This is what lets knowledge travel with agents (store-carry-forward)."""
        pairs = set()
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            for j in self._in_range(i):
                if j > i:
                    pairs.add((i, j))
        for i, j in sorted(pairs - self._contacts):
            for src, dst in ((i, j), (j, i)):
                a = self.agents[src]
                if a.role == Role.ALLOCATOR.value:
                    continue
                blocked = a.local_memory["blocked_nodes"]
                danger = a.local_memory["danger_nodes"]
                if blocked or danger:
                    self.bus.send(src, "sync",
                                  {"blocked": tuple(blocked), "danger": dict(danger)},
                                  t, [dst])
        self._contacts = pairs

    def _apply_delivery(self, recipient, msg, t):
        if recipient == BOARD:
            sm = self.shared_memory
            for nid in msg.payload.get("blocked", ()):
                if nid not in sm["blocked_nodes"]:
                    sm["blocked_nodes"].add(nid)
                    self.board_visible[nid] = t
            for nid, dmg in msg.payload.get("danger", {}).items():
                if dmg > sm["danger_nodes"].get(nid, 0.0):
                    sm["danger_nodes"][nid] = dmg
        elif msg.kind == "assign":
            self.coordinator.on_assign(recipient, msg)
        else:
            self.agents[recipient].receive(msg)

    def _invalidate_blocked_paths(self, step: int, new_events: list):
        """
        For each agent, if the node it is about to step into has just been
        newly blocked (damage > 0.6), force a re-plan by:
          - Adding that node to the agent's blacklist
          - Resetting taboo (so it can explore freely)
          - Zeroing momentum (stale direction no longer valid)
          - Setting _steps_without_progress = MAX so the stall handler fires
            immediately on the next step() call
        This simulates agents receiving a "road closed" signal mid-route.
        """
        newly_blocked: set = set()
        for ev in new_events:
            for nid in ev.nodes_newly_damaged:
                if self.graph._node_damage.get(nid, 0.0) > 0.6:
                    newly_blocked.add(nid)

        for i, agent in enumerate(self.agents):
            if not agent.alive or agent.reached_goal or agent.stationary:
                continue

            # Check immediate neighbors the agent might move to
            # use the UNFILTERED neighbour list: get_neighbors() already drops
            # closed nodes, which would make this check always empty
            neighbors = agent.get_all_neighbors()
            blocked_neighbors = [n for n in neighbors if n in newly_blocked]

            if blocked_neighbors:
                # Path ahead is compromised — force re-plan
                for bn in blocked_neighbors:
                    agent.blacklist.add(bn)
                agent._taboo_reset()
                agent.snn.reset_momentum()
                agent._steps_without_progress = agent.MAX_STEPS_WITHOUT_PROGRESS

                self._agent_reroute_events[i] += 1
                print(f"    ↳ Agent {i:2d} PATH INVALIDATED at step {step}: "
                      f"{len(blocked_neighbors)} neighbor(s) blocked. "
                      f"Reroute #{self._agent_reroute_events[i]}")

    # -----------------------------------------------------------------------
    # Near-goal stuck rescue (from SwarmSimulation v3)
    # -----------------------------------------------------------------------

    def _rescue_near_goal_stuck(self, agent):
        if agent._last_dist_to_goal is None:
            return
        if agent._last_dist_to_goal > 300:
            return
        if agent._steps_without_progress < 10:
            return

        neighbors = agent.get_neighbors()
        if not neighbors:
            return

        goal_data = agent.graph._nodes[agent.goal_node]
        agent._taboo_reset()
        agent.blacklist.clear()
        agent.snn.reset_momentum()
        agent._steps_without_progress = 0

        best = min(
            neighbors,
            key=lambda n: haversine_distance(
                agent.graph._nodes[n]['lat'], agent.graph._nodes[n]['lon'],
                goal_data['lat'], goal_data['lon']
            )
        )
        agent._execute_move(best,
                            agent.graph._nodes[agent.current_node],
                            goal_data)

    # -----------------------------------------------------------------------
    # Shared memory helpers (from SwarmSimulation v3)
    # -----------------------------------------------------------------------

    def _deposit_pheromones(self, agent):
        path_len = len(agent.path)
        ph       = self.shared_memory["pheromones"]
        quality  = max(0.5, 1.0 - path_len / 400.0)
        for i, node in enumerate(agent.path):
            pos_factor = i / max(path_len - 1, 1)
            strength   = self.PHEROMONE_DEPOSIT * pos_factor * quality
            ph[node]   = min(1.0, ph.get(node, 0.0) + strength)

    def _evaporate_pheromones(self):
        # In place: agents hold a view onto this same dict.
        ph = self.shared_memory["pheromones"]
        for k in list(ph):
            v = ph[k] * self.PHEROMONE_DECAY
            if v > 0.01:
                ph[k] = v
            else:
                del ph[k]

    def _distance(self, n1, n2):
        a, b = self.graph._nodes[n1], self.graph._nodes[n2]
        return haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    # -----------------------------------------------------------------------
    # Summary — includes adaptation metrics
    # -----------------------------------------------------------------------

    def navigators(self):
        return [a for a in self.agents if a.role == Role.NAVIGATOR.value]

    def summary(self):
        navs    = self.navigators()
        total   = len(navs)
        reached = sum(a.reached_goal for a in navs)
        alive   = sum(a.alive and not a.reached_goal for a in navs)
        dead    = sum(not a.alive and not a.reached_goal for a in navs)

        print("\n" + "=" * 60)
        print("DYNAMIC SWARM SUMMARY")
        print("=" * 60)
        print(f"Total navigators: {total}  (swarm size {len(self.agents)})")
        print(f"Reached goal  : {reached} ({100*reached/total:.1f}%)")
        print(f"Still alive   : {alive}   ({100*alive/total:.1f}%)")
        print(f"Dead          : {dead}   ({100*dead/total:.1f}%)")
        print("=" * 60)

        # Disaster progression
        print("\nDISASTER PROGRESSION:")
        eng_summary = self.engine.summary()
        for k, v in eng_summary.items():
            print(f"  {k:<30}: {v}")

        print(f"\n  Total disaster events logged : {len(self.disaster_log)}")
        event_types: dict = {}
        for _, ev in self.disaster_log:
            event_types[ev.event_type] = event_types.get(ev.event_type, 0) + 1
        for etype, cnt in sorted(event_types.items()):
            print(f"    {etype:<22}: {cnt}")

        # Adaptation metrics
        total_reroutes = sum(self._agent_reroute_events.values())
        total_adapt_lag = sum(self._agent_adaptation_lag.values())
        active_agents  = [i for i, a in enumerate(self.agents)
                          if a.role == Role.NAVIGATOR.value
                          and self._agent_reroute_events[i] > 0]

        print("\nADAPTATION METRICS:")
        print(f"  Total reroute events         : {total_reroutes}")
        if active_agents:
            mean_reroutes = sum(
                self._agent_reroute_events[i] for i in active_agents
            ) / len(active_agents)
            print(f"  Agents affected by reroutes  : {len(active_agents)}/{total}")
            print(f"  Mean reroutes per agent      : {mean_reroutes:.1f}")
            if total_reroutes > 0:
                mean_lag = total_adapt_lag / total_reroutes
                print(f"  Mean adaptation lag (steps)  : {mean_lag:.1f}")
        else:
            print("  No agents required rerouting.")

        if reached > 0:
            path_lens  = [len(a.path) for a in navs if a.reached_goal]
            goal_steps = list(self._goal_step.values())
            reroutes_s = [self._agent_reroute_events[i]
                          for i, a in enumerate(self.agents)
                          if a.role == Role.NAVIGATOR.value and a.reached_goal]

            print("\nSUCCESSFUL AGENTS:")
            print(f"  Path length : min={min(path_lens)}  max={max(path_lens)}  "
                  f"mean={sum(path_lens)/len(path_lens):.1f}")
            if goal_steps:
                print(f"  Goal step   : first={min(goal_steps)}  last={max(goal_steps)}  "
                      f"mean={sum(goal_steps)/len(goal_steps):.1f}")
            print(f"  Reroutes    : min={min(reroutes_s)}  max={max(reroutes_s)}  "
                  f"mean={sum(reroutes_s)/len(reroutes_s):.1f}")

            print("\n  Per-agent breakdown:")
            for i, agent in enumerate(self.agents):
                if agent.role == Role.NAVIGATOR.value and agent.reached_goal:
                    step     = self._goal_step.get(i, "?")
                    reroutes = self._agent_reroute_events[i]
                    lag      = self._agent_adaptation_lag[i]
                    bl       = len(agent.blacklist)
                    print(f"    Agent {i:2d}: {len(agent.path):4d} steps | "
                          f"goal@{step} | reroutes={reroutes} | "
                          f"adapt_lag={lag} | blacklisted={bl}")

        ph_nodes = len(self.shared_memory["pheromones"])
        print(f"\nPheromone trail: {ph_nodes} nodes marked at run end")
        cs = self.bus.stats()
        print(f"Communication  : {cs['transmissions']} tx, {cs['deliveries']} rx, "
              f"{cs['bytes_tx']} B tx, mean delay {cs['mean_delay_steps']:.2f} steps")
        if self.coordination_log:
            print(f"Coordination   : {len(self.coordination_log)} events "
                  f"({self.coord_cfg.mode})")
        print("=" * 60)

    def adaptation_metrics(self) -> dict:
        """
        Machine-readable adaptation metrics for the evaluator / paper tables.
        Navigators only: scouts and hubs are support roles, not tasks to succeed at.
        """
        nav_idx = [i for i, a in enumerate(self.agents)
                   if a.role == Role.NAVIGATOR.value]
        total = len(nav_idx)
        reached_idx = [i for i in nav_idx if self.agents[i].reached_goal]
        reroutes = sum(self._agent_reroute_events[i] for i in nav_idx)
        lag = sum(self._agent_adaptation_lag[i] for i in nav_idx)

        return {
            "total_agents":          total,
            "reached_goal":          len(reached_idx),
            "success_rate":          round(len(reached_idx) / total, 3) if total else 0,
            "total_reroute_events":  reroutes,
            "mean_reroutes_per_agent": round(reroutes / total, 2) if total else 0,
            "mean_adaptation_lag":   round(lag / max(reroutes, 1), 2),
            "mean_path_length": round(
                sum(len(self.agents[i].path) for i in reached_idx) / len(reached_idx),
                1,
            ) if reached_idx else None,
            "mean_goal_step": round(
                sum(self._goal_step[i] for i in reached_idx if i in self._goal_step)
                / len(reached_idx),
                1,
            ) if reached_idx else None,
            "disaster_events": len(self.disaster_log),
            "disaster_summary": self.engine.summary(),
        }

    # -----------------------------------------------------------------------
    # Communication / role metrics
    # -----------------------------------------------------------------------

    def info_propagation(self) -> dict:
        """
        How fast hazards spread from the agent that first sensed them.
        For each hazard node and each OTHER alive agent: when did that agent
        learn it (told directly, or readable on the board), and did it by run end?
        """
        delays, informed, pairs = [], 0, 0
        for nid, (t0, detector) in self.detection_log.items():
            for j, a in enumerate(self.agents):
                if j == detector or not a.alive:
                    continue
                pairs += 1
                times = []
                if nid in a.learned_at:
                    times.append(a.learned_at[nid][0])
                if self.comm.shared_memory and nid in self.board_visible:
                    times.append(self.board_visible[nid])
                if times:
                    informed += 1
                    delays.append(max(0, min(times) - t0))
        return {
            "hazards_detected":  len(self.detection_log),
            "agent_hazard_pairs": pairs,
            "fraction_informed": round(informed / pairs, 4) if pairs else None,
            "mean_delay_steps":  round(sum(delays) / len(delays), 3) if delays else None,
        }

    def contributions(self) -> list:
        """Per-agent contribution record (inputs to per-agent contribution analysis)."""
        stats = self.bus.stats()
        first = {}
        for _, (_, idx) in self.detection_log.items():
            first[idx] = first.get(idx, 0) + 1
        out = []
        for i, a in enumerate(self.agents):
            out.append({
                "agent": i, "role": a.role, "alive": a.alive,
                "reached_goal": a.reached_goal if a.role == Role.NAVIGATOR.value else None,
                "first_detections": first.get(i, 0),
                "tx": stats["tx_by_sender"].get(i, 0),
                "bytes_tx": stats["bytes_by_sender"].get(i, 0),
                "msgs_received": a.stats["msgs_received"],
                "blocked_learned_self": a.stats["blocked_learned_self"],
                "blocked_learned_msg": a.stats["blocked_learned_msg"],
                "nodes_visited": len(a.visited),
                "cells_covered": len(a.cell_counts),
                "path_len": len(a.path),
                "replans": a.stats["replans"],
                "plan_invalidations": a.stats["plan_invalidations"],
                "goal_changes": a.stats["goal_changes"],
            })
        return out

    def metrics(self) -> dict:
        from dataclasses import asdict
        from collections import Counter
        return {
            "config": {
                "comm": asdict(self.comm),
                "roles": dict(Counter(a.role for a in self.agents)),
                "specialised_roles": self._specialised,
                "coordination": self.coord_cfg.mode,
                "shelters": len(self._shelters) if self._shelters else 0,
            },
            "task": self.adaptation_metrics(),
            "comms": self.bus.stats(),
            "info_propagation": self.info_propagation(),
            "coordination_events": list(self.coordination_log),
            "contributions": self.contributions(),
        }
