"""
feasibility.py
--------------
Task sampling that only hands out tasks an agent could actually complete,
plus the seeding helpers that keep agents statistically independent.

Why this exists
---------------
Spawning random (start, goal) pairs and then counting every non-arrival as a
swarm failure mixes two different things: agents that were bad at a feasible
task, and agents that were given an impossible one (goal sealed off by damage,
start inside a destroyed node, route far longer than the step budget). This
module removes the second category at t=0 and records the optimal route for
every task, so later metrics can report path stretch against a real optimum.

Passability semantics are exactly the agent's (see SNNAgent.is_passable):
a move u -> v is legal iff the edge is open AND damage(v) <= threshold.

Seeding
-------
Every agent gets two private random streams derived from (seed, index):
one for choosing its task and one for its decisions. Agent i's task and
behaviour therefore do not change when the swarm has more or fewer agents.
"""

import heapq
import random
from collections import deque
from typing import Optional

from snn_core_v2 import haversine_distance

BLOCK_DAMAGE_THRESHOLD = 0.6   # keep in sync with SNNAgent.BLOCK_DAMAGE_THRESHOLD


class SpawnError(RuntimeError):
    """Raised when no feasible task can be found under the given constraints."""


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _stream(seed, kind: str, index: int) -> random.Random:
    """Deterministic named stream. String seeds are hashed deterministically
    by Python (independent of PYTHONHASHSEED), so runs reproduce across
    processes and machines. seed=None gives a non-reproducible stream."""
    if seed is None:
        return random.Random()
    return random.Random(f"{seed}:{kind}:{index}")


def agent_rng(seed, index: int) -> random.Random:
    """Stream for agent `index`'s decisions."""
    return _stream(seed, "agent", index)


def spawn_rng(seed, index: int) -> random.Random:
    """Stream for choosing agent `index`'s task."""
    return _stream(seed, "spawn", index)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

def is_open(graph, u: int, v: int, thr: float = BLOCK_DAMAGE_THRESHOLD) -> bool:
    return graph.is_edge_passable(u, v) and graph.get_node_damage(v) <= thr


def passable_components(graph, thr: float = BLOCK_DAMAGE_THRESHOLD) -> dict:
    """
    Label connected components of the currently passable network.
    Returns {node_id: component_id}; destroyed nodes (damage > thr) are absent.
    One pass over the graph (~0.5 s for Tokyo); compute once per scenario state.
    """
    comp: dict = {}
    n = 0
    for s in graph._nodes:
        if s in comp or graph.get_node_damage(s) > thr:
            continue
        comp[s] = n
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v, _ in graph._adj.get(u, ()):
                if v in comp or graph.get_node_damage(v) > thr:
                    continue
                if not graph.is_edge_passable(u, v):
                    continue
                comp[v] = n
                dq.append(v)
        n += 1
    return comp


def optimal_route(graph, start: int, goal: int,
                  thr: float = BLOCK_DAMAGE_THRESHOLD) -> Optional[dict]:
    """
    Shortest passable route by distance (early-exit Dijkstra).
    Returns {"dist_m", "hops"} or None if the goal cannot be reached.
    """
    if start == goal:
        return {"dist_m": 0.0, "hops": 0}
    dist = {start: 0.0}
    hops = {start: 0}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == goal:
            return {"dist_m": d, "hops": hops[u]}
        for v, edge in graph._adj.get(u, ()):
            if not is_open(graph, u, v, thr):
                continue
            nd = d + edge["length"]
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                hops[v] = hops[u] + 1
                heapq.heappush(heap, (nd, v))
    return None


# ---------------------------------------------------------------------------
# Task validation / sampling
# ---------------------------------------------------------------------------

def _straight_m(graph, a: int, b: int) -> float:
    na, nb = graph._nodes[a], graph._nodes[b]
    return haversine_distance(na["lat"], na["lon"], nb["lat"], nb["lon"])


def validate_task(graph, start: int, goal: int, comp: dict = None,
                  max_optimal_hops: Optional[int] = None,
                  thr: float = BLOCK_DAMAGE_THRESHOLD) -> Optional[dict]:
    """
    Return task info if (start, goal) is feasible right now, else None.
    Feasible = both nodes intact, same passable component, and (optionally)
    optimal route no longer than `max_optimal_hops`.
    """
    if start == goal or start not in graph._nodes or goal not in graph._nodes:
        return None
    if comp is None:
        comp = passable_components(graph, thr)
    if start not in comp or goal not in comp or comp[start] != comp[goal]:
        return None
    route = optimal_route(graph, start, goal, thr)
    if route is None:
        return None
    if max_optimal_hops is not None and route["hops"] > max_optimal_hops:
        return None
    return {
        "start": start, "goal": goal,
        "straight_m": _straight_m(graph, start, goal),
        "optimal_m": route["dist_m"],
        "optimal_hops": route["hops"],
    }


def sample_feasible_task(graph, rng: random.Random, comp: dict,
                         candidates: list,
                         min_dist: float = 1000, max_dist: float = 4000,
                         max_optimal_hops: Optional[int] = None,
                         max_attempts: int = 2000,
                         thr: float = BLOCK_DAMAGE_THRESHOLD) -> dict:
    """Draw (start, goal) until one is feasible. Raises SpawnError otherwise."""
    for _ in range(max_attempts):
        s = rng.choice(candidates)
        g = rng.choice(candidates)
        if s == g or comp[s] != comp[g]:
            continue
        if not (min_dist <= _straight_m(graph, s, g) <= max_dist):
            continue
        info = validate_task(graph, s, g, comp, max_optimal_hops, thr)
        if info is not None:
            return info
    raise SpawnError(
        f"No feasible task in {max_attempts} attempts "
        f"(distance {min_dist:.0f}-{max_dist:.0f} m, max_optimal_hops="
        f"{max_optimal_hops}). The scenario may be too damaged, or the "
        f"constraints too tight.")


def candidate_nodes(graph, comp: dict) -> list:
    """Intact nodes that have at least one neighbour, in stable order."""
    return [n for n in graph._nodes if n in comp and graph._adj.get(n)]


def generate_tasks(graph, n: int, seed=None,
                   min_dist: float = 1000, max_dist: float = 4000,
                   max_optimal_hops: Optional[int] = None,
                   max_attempts: int = 2000) -> list:
    """
    Generate n feasible tasks for the graph's CURRENT damage state.
    Task i depends only on (seed, i): generating 1 or 20 tasks gives the same
    first task. Use this to share identical tasks between experiment arms
    (single agent / independent / swarm / ablations).
    """
    comp = passable_components(graph)
    cands = candidate_nodes(graph, comp)
    return [
        sample_feasible_task(graph, spawn_rng(seed, i), comp, cands,
                             min_dist, max_dist, max_optimal_hops, max_attempts)
        for i in range(n)
    ]
