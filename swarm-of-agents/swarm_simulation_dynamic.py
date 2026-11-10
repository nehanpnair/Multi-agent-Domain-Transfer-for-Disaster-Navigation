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
    ):
        self.graph      = graph
        self.num_agents = num_agents
        self.min_start_goal_distance = min_start_goal_distance
        self.max_start_goal_distance = max_start_goal_distance

        self._rng = random.Random(seed)
        if seed is not None:
            random.seed(seed)

        # ---- Disaster engine ----
        self.engine = DynamicDisasterEngine(
            graph            = graph,
            disaster_type    = disaster_type,
            centre           = disaster_centre,
            radius_m         = disaster_radius_m,
            severity         = disaster_severity,
            seed             = seed,
            wind_bearing_deg = wind_bearing_deg,
        )

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

        # Apply t=0 mainshock, THEN spawn agents so spawn distances
        # reflect the post-disaster graph passability.
        init_ev = self.engine.initialise()
        self.disaster_log.append((0, init_ev))
        print(f"\n{init_ev}")

        self._spawn_agents()

    # -----------------------------------------------------------------------
    # Spawn (identical logic to SwarmSimulation v3, uses self._rng)
    # -----------------------------------------------------------------------

    def _spawn_agents(self):
        nodes = list(self.graph._nodes.keys())

        for i in range(self.num_agents):
            placed, attempts = False, 0
            while attempts < 200:
                attempts += 1
                start = self._rng.choice(nodes)
                goal  = self._rng.choice(nodes)
                if start == goal:
                    continue
                sd   = self.graph._nodes[start]
                gd   = self.graph._nodes[goal]
                dist = haversine_distance(sd['lat'], sd['lon'], gd['lat'], gd['lon'])
                if (self.min_start_goal_distance <= dist <= self.max_start_goal_distance
                        and self.graph._adj.get(start)
                        and self.graph._adj.get(goal)):
                    placed = True
                    break

            if not placed:
                candidates = [n for n in nodes if self.graph._adj.get(n)]
                if len(candidates) >= 2:
                    start, goal = self._rng.sample(candidates, 2)
                else:
                    start, goal = self._rng.sample(nodes, 2)

            agent = SNNAgent(self.graph, start, goal)
            agent.shared_memory = self.shared_memory
            self.agents.append(agent)

            self._agent_reroute_events[i] = 0
            self._agent_adaptation_lag[i] = 0

            sd   = self.graph._nodes[start]
            gd   = self.graph._nodes[goal]
            dist = haversine_distance(sd['lat'], sd['lon'], gd['lat'], gd['lon'])
            print(f"Agent {i:2d} | Start: {start} → Goal: {goal} | {dist:.0f}m")

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

    def run(self, steps: int = 200):
        for t in range(1, steps + 1):
            print(f"\n{'='*60}")
            print(f"STEP {t}")
            print(f"{'='*60}")

            # 1. Advance disaster — BEFORE agents move
            new_events = self.engine.tick(t)
            for ev in new_events:
                self.disaster_log.append((t, ev))
                print(f"  ⚡ DISASTER: {ev}")

            self.current_step = t

            # 2. Agents adjacent to newly blocked nodes are alerted. Shared memory
            #    is NOT updated from the engine: the swarm only learns about
            #    hazards that an agent has itself sensed (see _publish_observation).
            if new_events:
                self._invalidate_blocked_paths(t, new_events)

            # 3. Move agents
            active_count = 0
            for i, agent in enumerate(self.agents):
                if not agent.alive or agent.reached_goal:
                    continue

                active_count += 1
                self._apply_shared_memory(agent)
                self._rescue_near_goal_stuck(agent)

                prev_node = agent.current_node
                prev_progress = agent._steps_without_progress

                agent.step()

                # Track adaptation lag — steps the agent spent in stall handler
                # after a reroute event
                if (self._agent_reroute_events[i] > 0
                        and agent._steps_without_progress > prev_progress):
                    self._agent_adaptation_lag[i] += 1

                if agent.current_node != prev_node:
                    cd  = self.graph._nodes[agent.current_node]
                    gd  = self.graph._nodes[agent.goal_node]
                    dtg = haversine_distance(cd['lat'], cd['lon'], gd['lat'], gd['lon'])
                    status = "ALIVE"
                    if not agent.alive:      status = "DEAD"
                    elif agent.reached_goal: status = "GOAL!"
                    print(f"  Agent {i:2d}: {prev_node} → {agent.current_node} | "
                          f"DTG: {dtg:6.0f}m | {status}")

                self._publish_observation(i, agent)

                if agent.reached_goal:
                    self._deposit_pheromones(agent)
                    self._goal_step[i] = t

            self._evaporate_pheromones()

            active_after = sum(1 for a in self.agents if a.alive and not a.reached_goal)
            print(f"\n  Active agents: {active_after}")

            if active_after == 0:
                print("\n  All agents finished.")
                break

            self._share_information()

    # -----------------------------------------------------------------------
    # Dynamic disaster integration
    # -----------------------------------------------------------------------

    def _publish_observation(self, idx, agent):
        """
        Publish only what `agent` physically sensed this step (1-hop neighbours
        and its own node). This replaces the old engine-to-shared-memory push,
        which gave the swarm perfect knowledge of the whole disaster.
        """
        obs = agent.last_observation
        for nid in obs["blocked_nodes"]:
            self.shared_memory["blocked_nodes"].add(nid)
            self.detection_log.setdefault(nid, (self.current_step, idx))
        for nid, dmg in obs["danger"].items():
            prev = self.shared_memory["danger_nodes"].get(nid, 0.0)
            self.shared_memory["danger_nodes"][nid] = max(prev, dmg)

        # the node the agent now stands on
        node   = agent.current_node
        hazard = agent.graph.get_node_damage(node)
        if hazard > agent.BLOCK_DAMAGE_THRESHOLD:
            self.shared_memory["blocked_nodes"].add(node)
            self.detection_log.setdefault(node, (self.current_step, idx))
        if hazard > 0.0:
            self.shared_memory["danger_nodes"][node] = max(
                hazard, self.shared_memory["danger_nodes"].get(node, 0.0))

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
            if not agent.alive or agent.reached_goal:
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

    def _apply_shared_memory(self, agent):
        if not hasattr(agent, 'shared_memory') or agent.shared_memory is None:
            agent.shared_memory = self.shared_memory

    def _deposit_pheromones(self, agent):
        path_len = len(agent.path)
        ph       = self.shared_memory["pheromones"]
        quality  = max(0.5, 1.0 - path_len / 400.0)
        for i, node in enumerate(agent.path):
            pos_factor = i / max(path_len - 1, 1)
            strength   = self.PHEROMONE_DEPOSIT * pos_factor * quality
            ph[node]   = min(1.0, ph.get(node, 0.0) + strength)

    def _evaporate_pheromones(self):
        ph = self.shared_memory["pheromones"]
        self.shared_memory["pheromones"] = {
            k: v * self.PHEROMONE_DECAY
            for k, v in ph.items()
            if v * self.PHEROMONE_DECAY > 0.01
        }

    def _share_information(self):
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            for j, b in enumerate(self.agents):
                if j <= i or not b.alive:
                    continue
                if self._distance(a.current_node, b.current_node) < 500:
                    all_nodes = (set(a.local_memory["danger_nodes"]) |
                                 set(b.local_memory["danger_nodes"]))
                    for node in all_nodes:
                        ha = a.local_memory["danger_nodes"].get(node, 0.0)
                        hb = b.local_memory["danger_nodes"].get(node, 0.0)
                        merged = max(ha, hb) * 0.95
                        if merged > 0.0:
                            a.local_memory["danger_nodes"][node] = merged
                            b.local_memory["danger_nodes"][node] = merged

    def _distance(self, n1, n2):
        a, b = self.graph._nodes[n1], self.graph._nodes[n2]
        return haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    # -----------------------------------------------------------------------
    # Summary — includes adaptation metrics
    # -----------------------------------------------------------------------

    def summary(self):
        total   = len(self.agents)
        reached = sum(a.reached_goal for a in self.agents)
        alive   = sum(a.alive and not a.reached_goal for a in self.agents)
        dead    = sum(not a.alive and not a.reached_goal for a in self.agents)

        print("\n" + "=" * 60)
        print("DYNAMIC SWARM SUMMARY")
        print("=" * 60)
        print(f"Total agents  : {total}")
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
        active_agents  = [i for i in range(total) if self._agent_reroute_events[i] > 0]

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
            path_lens  = [len(a.path) for a in self.agents if a.reached_goal]
            goal_steps = list(self._goal_step.values())
            reroutes_s = [self._agent_reroute_events[i]
                          for i, a in enumerate(self.agents) if a.reached_goal]

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
                if agent.reached_goal:
                    step     = self._goal_step.get(i, "?")
                    reroutes = self._agent_reroute_events[i]
                    lag      = self._agent_adaptation_lag[i]
                    bl       = len(agent.blacklist)
                    print(f"    Agent {i:2d}: {len(agent.path):4d} steps | "
                          f"goal@{step} | reroutes={reroutes} | "
                          f"adapt_lag={lag} | blacklisted={bl}")

        ph_nodes = len(self.shared_memory["pheromones"])
        print(f"\nPheromone trail: {ph_nodes} nodes marked at run end")
        print("=" * 60)

    def adaptation_metrics(self) -> dict:
        """
        Machine-readable adaptation metrics for the evaluator / paper tables.
        """
        total = len(self.agents)
        reached_idx = [i for i, a in enumerate(self.agents) if a.reached_goal]

        return {
            "total_agents":          total,
            "reached_goal":          len(reached_idx),
            "success_rate":          round(len(reached_idx) / total, 3) if total else 0,
            "total_reroute_events":  sum(self._agent_reroute_events.values()),
            "mean_reroutes_per_agent": round(
                sum(self._agent_reroute_events.values()) / total, 2
            ) if total else 0,
            "mean_adaptation_lag":   round(
                sum(self._agent_adaptation_lag.values()) /
                max(sum(self._agent_reroute_events.values()), 1),
                2,
            ),
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
