"""
test_fixes_8_9_10.py
--------------------
Checks for:
  8  per-agent RNG        -> reproducible, and independent of swarm size
  9  graph run isolation  -> forks never leak damage; reuse == fresh load
  10 feasible spawning    -> every task solvable at t=0; old spawner was not

Run:  python test_fixes_8_9_10.py [path/to/tokyo_graph.json]
Exit code is non-zero if any check fails.
"""

import contextlib
import io
import os
import random
import sys
import time

from hybrid_spatial_index import HybridTokyoGraph, load_cached
from dynamic_disaster_engine import DynamicDisasterEngine
from swarm_simulation_dynamic import DynamicSwarmSimulation
from feasibility import generate_tasks, validate_task, passable_components, SpawnError
from snn_core_v2 import haversine_distance

GRAPH = (sys.argv[1] if len(sys.argv) > 1
         else os.environ.get("TOKYO_GRAPH", "tokyo_graph.json"))
CENTRE = (35.68, 139.76)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""),
          flush=True)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def make_sim(graph, dtype="earthquake", n=5, seed=3, steps=120, cls=DynamicSwarmSimulation,
             **kw):
    def go():
        sim = cls(graph, num_agents=n, disaster_type=dtype, seed=seed, **kw)
        sim.run(steps=steps)
        return sim
    return quiet(go)


def paths(sim):
    return [list(a.path) for a in sim.agents]


class IsolatedSim(DynamicSwarmSimulation):
    """Agents cannot influence each other (no shared memory, messages or
    pheromones). With this, agent i must behave identically at any swarm size."""
    def _spawn_agents(self):
        super()._spawn_agents()
        for a in self.agents:
            a.shared_memory = {"blocked_nodes": set(), "danger_nodes": {}, "pheromones": {}}
    def _publish_observation(self, idx, agent): pass
    def _share_information(self): pass
    def _deposit_pheromones(self, agent): pass


def common_prefix_equal(a, b):
    return all(x == y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
print("\n=== load ===", flush=True)
t0 = time.time()
g0 = load_cached(GRAPH)
print(f"  first load+index: {time.time()-t0:.1f}s", flush=True)

# ---------------------------------------------------------------------------
print("\n=== FIX 9: graph run isolation ===", flush=True)

t0 = time.time(); gx = load_cached(GRAPH); fork_s = time.time() - t0
check("subsequent load is a cheap fork", fork_s < 1.0, f"{fork_s*1000:.1f} ms")
check("fresh fork has no damage", gx.damage_fingerprint() == (0, 0, 0.0))

gA = load_cached(GRAPH)
simA = make_sim(gA, "earthquake", seed=3)
fpA = gA.damage_fingerprint()
check("run damages its own fork", fpA[0] > 0, f"fingerprint={fpA}")

gB = load_cached(GRAPH)
check("new fork after a run is undamaged (no leakage)", gB.damage_fingerprint() == (0, 0, 0.0))
base_fp = load_cached(GRAPH).damage_fingerprint()
check("cache base graph untouched", base_fp == (0, 0, 0.0))

simB = make_sim(gB, "earthquake", seed=3)
check("same seed on a reused graph reproduces the run exactly", paths(simA) == paths(simB))

# concurrent forks do not interfere
gC, gD = load_cached(GRAPH), load_cached(GRAPH)
make_sim(gC, "flood", seed=3)
check("damaging one fork leaves a sibling untouched", gD.damage_fingerprint() == (0, 0, 0.0))
simD = make_sim(gD, "earthquake", seed=3)
check("sibling run unaffected by other fork's disaster", paths(simD) == paths(simA))

# reuse == genuinely fresh load
t0 = time.time()
gF = HybridTokyoGraph(GRAPH); quiet(gF.build_index)
print(f"  (fresh load+index for comparison took {time.time()-t0:.1f}s)", flush=True)
simF = make_sim(gF, "earthquake", seed=3)
check("forked run == run on a freshly loaded graph", paths(simA) == paths(simF))

# ---------------------------------------------------------------------------
print("\n=== FIX 8: per-agent RNG ===", flush=True)

simC = make_sim(load_cached(GRAPH), "earthquake", seed=4)
check("different seeds give different runs", paths(simC) != paths(simA))

gT = load_cached(GRAPH)
quiet(DynamicDisasterEngine(gT, "earthquake", CENTRE, 2000, 0.7, seed=3).initialise)
t1 = generate_tasks(gT, 1, seed=11)
t10 = generate_tasks(gT, 10, seed=11)
t20 = generate_tasks(gT, 20, seed=11)
check("task i does not depend on swarm size",
      t1[0] == t10[0] == t20[0] and t10 == t20[:10])

res = {}
for n in (1, 10, 20):
    res[n] = paths(make_sim(load_cached(GRAPH), "earthquake", n=n, seed=5, cls=IsolatedSim))
same_1_10 = res[1][0] == res[10][0]
same_10_20 = all(res[10][i] == res[20][i] for i in range(10))
check("isolated agent 0 identical in swarms of 1 and 10", same_1_10,
      f"path length {len(res[1][0])}")
check("isolated agents 0-9 identical in swarms of 10 and 20", same_10_20)

# control: put all agents back on ONE shared stream (the old behaviour) and the
# same comparison should break, proving the test can detect the problem
class SharedStreamSim(IsolatedSim):
    def _spawn_agents(self):
        super()._spawn_agents()
        shared = random.Random(self.seed)
        for a in self.agents:
            a.rng = shared
            a.snn.rng = shared
ctl = {n: paths(make_sim(load_cached(GRAPH), "earthquake", n=n, seed=5, cls=SharedStreamSim))
       for n in (1, 10)}
check("control: a single shared stream breaks agent independence",
      ctl[1][0] != ctl[10][0],
      "agent 0 path differs between N=1 and N=10" if ctl[1][0] != ctl[10][0]
      else "no difference observed (test has no power on this seed)")

# ---------------------------------------------------------------------------
print("\n=== FIX 10: feasible spawning ===", flush=True)


def independent_reachable(graph, s, g, thr=0.6):
    """Separate BFS implementation used only to audit the spawner."""
    if graph.get_node_damage(s) > thr or graph.get_node_damage(g) > thr:
        return False
    seen, stack = {s}, [s]
    while stack:
        u = stack.pop()
        if u == g:
            return True
        for v, _ in graph._adj.get(u, []):
            if v not in seen and graph.is_edge_passable(u, v) and graph.get_node_damage(v) <= thr:
                seen.add(v); stack.append(v)
    return False


def legacy_task(graph, rng, nodes, lo=1000, hi=4000):
    """The pre-fix spawner: random pair within a distance band, nothing else."""
    for _ in range(200):
        s, g = rng.choice(nodes), rng.choice(nodes)
        if s == g:
            continue
        a, b = graph._nodes[s], graph._nodes[g]
        if lo <= haversine_distance(a["lat"], a["lon"], b["lat"], b["lon"]) <= hi \
                and graph._adj.get(s) and graph._adj.get(g):
            return (s, g)
    raise RuntimeError("legacy sampler failed")


# harsh scenario so infeasibility is visible
gH = load_cached(GRAPH)
quiet(DynamicDisasterEngine(gH, "earthquake", CENTRE, 3000, 0.95, seed=1).initialise)
nodes = list(gH._nodes)
rng = random.Random(0)
legacy = [legacy_task(gH, rng, nodes) for _ in range(300)]
comp = passable_components(gH)
bad = sum(1 for s, g in legacy if validate_task(gH, s, g, comp) is None)
print(f"  legacy spawner, harsh quake: {bad}/300 tasks infeasible at t=0 "
      f"({100*bad/300:.1f}%)", flush=True)
check("legacy spawner does produce infeasible tasks (the bug)", bad > 0)

t0 = time.time()
tasks = generate_tasks(gH, 20, seed=2)
spawn_s = time.time() - t0
check("all 20 spawned tasks pass an independent reachability audit",
      all(independent_reachable(gH, t["start"], t["goal"]) for t in tasks),
      f"{spawn_s:.1f}s incl. component labelling")
check("optimal route is never shorter than straight line",
      all(t["optimal_m"] >= t["straight_m"] - 1 for t in tasks))
check("distance band respected",
      all(1000 <= t["straight_m"] <= 4000 for t in tasks))

# explicit tasks
infeasible = next(((s, g) for s, g in legacy if validate_task(gH, s, g, comp) is None), None)
if infeasible:
    def build(**kw):
        return DynamicSwarmSimulation(gH.fork(), num_agents=1, disaster_type="earthquake",
                                      seed=1, disaster_centre=CENTRE, disaster_radius_m=3000,
                                      disaster_severity=0.95, tasks=[infeasible], **kw)
    try:
        quiet(build); raised = False
    except ValueError:
        raised = True
    check("explicit infeasible task is rejected by default", raised)
    try:
        quiet(build, require_feasible=False); allowed = True
    except ValueError:
        allowed = False
    check("...but allowed with require_feasible=False", allowed)

try:
    generate_tasks(gH, 1, seed=1, min_dist=5e6, max_dist=6e6, max_attempts=50)
    check("impossible constraints raise SpawnError", False)
except SpawnError:
    check("impossible constraints raise SpawnError", True)

# shared tasks across arms
gS = load_cached(GRAPH)
simS = make_sim(gS, "earthquake", n=10, seed=8)
simS1 = make_sim(load_cached(GRAPH), "earthquake", n=1, seed=8, tasks=simS.tasks[:1])
check("tasks can be reused across experiment arms",
      simS1.tasks[0]["start"] == simS.tasks[0]["start"]
      and simS1.tasks[0]["goal"] == simS.tasks[0]["goal"])

# ---------------------------------------------------------------------------
print("\n=== end-to-end: legacy vs feasible tasks (5 seeds x 4 disasters, 10 agents) ===",
      flush=True)
print(f"  {'disaster':<11}{'legacy infeasible':>19}{'legacy success':>16}{'feasible success':>18}")
tot_bad = tot_n = 0
for dtype in ("earthquake", "flood", "fire", "compound"):
    lb = ls = fs = n_tasks = 0
    for seed in range(5):
        g = load_cached(GRAPH)
        sim = make_sim(g, dtype, n=10, seed=seed)           # feasible spawner
        fs += sum(a.reached_goal for a in sim.agents)

        g2 = load_cached(GRAPH)
        quiet(DynamicDisasterEngine(g2, dtype, CENTRE, 2000, 0.7, seed=seed).initialise)
        c2 = passable_components(g2)
        r = random.Random(f"legacy:{seed}")
        lt = [legacy_task(g2, r, list(g2._nodes)) for _ in range(10)]
        lb += sum(1 for s, gl in lt if validate_task(g2, s, gl, c2) is None)
        simL = make_sim(load_cached(GRAPH), dtype, n=10, seed=seed, tasks=lt,
                        require_feasible=False)
        ls += sum(a.reached_goal for a in simL.agents)
        n_tasks += 10
    tot_bad += lb; tot_n += n_tasks
    print(f"  {dtype:<11}{lb:>6}/{n_tasks:<3} ({100*lb/n_tasks:4.1f}%){100*ls/n_tasks:>13.0f}%"
          f"{100*fs/n_tasks:>17.0f}%", flush=True)
check("legacy tasks include infeasible ones across scenarios", tot_bad > 0,
      f"{tot_bad}/{tot_n} overall")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed", flush=True)
sys.exit(0 if all(RESULTS) else 1)
