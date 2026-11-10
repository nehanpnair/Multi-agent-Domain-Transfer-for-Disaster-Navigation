import math
import random


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class SNNDecision:
    """
    Potential-field scorer — v2 fixes applied:

    F4  temperature ladder: 1.8 (far) → 0.5 (mid) → 0.05 (near) → greedy (final)
        Restores decision diversity that temperature=0.3 was killing.
    F2  w_pheromone=8.0 added so shared trail actually influences decisions.
        (Feature is now fed in from build_features via feat['pheromone'].)
    Tuning:
        w_progress reduced 15→9 to stop greedy tunneling into dead-end corridors.
        w_explore raised 0.5→2.0 to widen the search cone.
        exploration_rate raised 0.05→0.12.
    """

    def __init__(self, rng=None):
        # Own random stream (a random.Random). Falls back to the global module
        # so existing scripts behave exactly as before.
        self.rng = rng if rng is not None else random
        self.w_goal       = 10.0
        self.w_hazard     = -8.0
        self.w_visited    = -1.5   # applied as log-penalty now, see score_neighbor
        self.w_backtrack  = -6.0
        self.w_progress   = 9.0    # FIX: was 15.0 — greedy tunneling
        self.w_explore    = 2.0    # FIX: was 0.5
        self.w_taboo      = -200.0
        self.w_blacklist  = -500.0
        self.w_two_hop    = 30.0
        self.w_momentum   = 4.0
        self.w_pheromone  = 8.0    # FIX F2: was 0.0 (feature existed but had no weight)
        # Knowledge-driven terms (fed by what the agent has sensed or been told)
        self.w_lookahead  = -6.0   # mean risk of the candidate's onward exits (known blocked=1)
        self.w_dead_end   = -25.0  # every onward exit of the candidate is known blocked
        self.w_plan       = 12.0   # candidate is the next hop of the agent's own plan

        self.momentum_decay   = 0.7
        self.last_direction   = None
        self.exploration_rate = 0.12   # FIX: was 0.05

    def reset_momentum(self):
        self.last_direction = None

    def apply_weights(self, overrides: dict):
        """Override scorer parameters by name (used by role profiles).
        Unknown names raise, so a typo in a role spec cannot silently do nothing."""
        for k, v in (overrides or {}).items():
            if not hasattr(self, k):
                raise AttributeError(f"SNNDecision has no parameter {k!r}")
            setattr(self, k, v)

    def compute_momentum_score(self, current_pos, next_pos):
        if self.last_direction is None:
            return 0.0
        dlat = next_pos['lat'] - current_pos['lat']
        dlon = next_pos['lon'] - current_pos['lon']
        mag  = math.sqrt(dlat**2 + dlon**2)
        if mag < 1e-9:
            return 0.0
        dot = (dlat / mag) * self.last_direction[0] + (dlon / mag) * self.last_direction[1]
        return dot * self.w_momentum

    def update_momentum(self, current_pos, next_pos):
        dlat = next_pos['lat'] - current_pos['lat']
        dlon = next_pos['lon'] - current_pos['lon']
        mag  = math.sqrt(dlat**2 + dlon**2)
        if mag < 1e-9:
            return
        nd = (dlat / mag, dlon / mag)
        if self.last_direction is None:
            self.last_direction = nd
        else:
            self.last_direction = (
                self.momentum_decay * self.last_direction[0] + (1 - self.momentum_decay) * nd[0],
                self.momentum_decay * self.last_direction[1] + (1 - self.momentum_decay) * nd[1],
            )

    def score_neighbor(self, feat, current_pos, goal_pos):
        d        = feat['dist']
        in_final = d < 80
        in_near  = d < 200
        in_mid   = d < 800

        # 1. Goal attraction
        dist_norm = min(d, 5000.0) / 5000.0
        base_goal = (1.0 - dist_norm) * self.w_goal
        if in_final:
            goal_score = base_goal * (1.0 + (80 - d) / 80 * 8.0)
        elif in_near:
            goal_score = base_goal * 2.5
        else:
            goal_score = base_goal

        # 2. Progress (reduced weight to avoid greedy tunneling — FIX)
        progress_score = feat.get('progress', 0.0) * self.w_progress

        # 3. Hazard
        haz_mult     = 0.1 if in_final else (0.5 if in_near else 1.0)
        hazard_score = feat['hazard'] * self.w_hazard * haz_mult

        # 4. Visited — soft log penalty so repeat visits compound (FIX: was binary 0/1/2)
        raw_visit    = feat.get('visited', 0.0)          # now a float from log1p
        vis_mult     = 0.3 if in_near else 1.0
        visited_score = raw_visit * self.w_visited * vis_mult

        # 5. Backtrack
        bt_mult        = 0.4 if in_near else 1.0
        backtrack_score = feat.get('is_prev_node', 0.0) * self.w_backtrack * bt_mult

        # 6. Exploration (suppressed near goal)
        explore_score = 0.0 if in_near else feat.get('is_new', 0.0) * self.w_explore

        # 7. Momentum
        mom_score = 0.0
        if 'node_data' in feat and current_pos is not None:
            mom_score = self.compute_momentum_score(current_pos, feat['node_data'])
            if in_near:
                mom_score *= 0.5

        # 8. Taboo
        taboo_score = feat.get('taboo', 0.0) * self.w_taboo

        # 9. Blacklist (near-infinite)
        bl_score = feat.get('blacklist', 0.0) * self.w_blacklist

        # 10. Two-hop goal bonus
        two_hop_score = feat.get('two_hop_goal', 0.0) * self.w_two_hop

        # 11. Pheromone trail bonus — FIX F2: was never scored
        pheromone_score = feat.get('pheromone', 0.0) * self.w_pheromone

        # 12. Knowledge terms. Dead-end avoidance is dropped in the final approach
        #     where the goal itself is the exit.
        look_score = feat.get('lookahead', 0.0) * self.w_lookahead
        if not in_final:
            look_score += feat.get('dead_end', 0.0) * self.w_dead_end
        plan_score = feat.get('on_plan', 0.0) * self.w_plan

        return (goal_score + progress_score + hazard_score + visited_score +
                backtrack_score + explore_score + mom_score +
                taboo_score + bl_score + two_hop_score + pheromone_score +
                look_score + plan_score)

    def softmax_selection(self, scores, temperature):
        max_s = max(scores)
        exp_s = [math.exp((s - max_s) / temperature) for s in scores]
        tot   = sum(exp_s)
        probs = [e / tot for e in exp_s]
        r     = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(scores) - 1

    def decide(self, neighbor_features, current_pos=None, goal_pos=None, goal_node=None):
        if not neighbor_features:
            return None

        # Always grab goal if adjacent
        if goal_node:
            for f in neighbor_features:
                if f['node'] == goal_node:
                    return goal_node

        dist_to_goal = float('inf')
        if goal_pos and current_pos:
            dist_to_goal = haversine_distance(
                current_pos['lat'], current_pos['lon'],
                goal_pos['lat'],    goal_pos['lon']
            )

        scores = [self.score_neighbor(f, current_pos, goal_pos) for f in neighbor_features]

        # FIX F4: temperature ladder — restores decision diversity at distance
        # Pure greedy in final 50m
        if dist_to_goal < 50:
            return neighbor_features[scores.index(max(scores))]['node']

        # Near-deterministic in final 200m
        if dist_to_goal < 200:
            return neighbor_features[self.softmax_selection(scores, temperature=0.05)]['node']

        # Focused but not locked in 200–800m band
        if dist_to_goal < 800:
            return neighbor_features[self.softmax_selection(scores, temperature=0.5)]['node']

        # Exploration zone: higher temperature + explicit exploration rate
        if self.rng.random() < self.exploration_rate:
            unvisited = [f for f in neighbor_features if f.get('is_new', 0.0) > 0.5]
            pool = unvisited if unvisited else neighbor_features
            return self.rng.choice(pool)['node']

        return neighbor_features[self.softmax_selection(scores, temperature=1.8)]['node']
