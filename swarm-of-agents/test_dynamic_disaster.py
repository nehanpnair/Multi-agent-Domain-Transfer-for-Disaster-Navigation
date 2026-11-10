"""
test_dynamic_disaster.py
------------------------
Test harness for the dynamic disaster swarm experiments.

Modes
-----
1. run_single()
   One simulation with a chosen disaster type. Good for debugging and
   quick inspection of agent behaviour.

2. run_experiment()
   n_runs independent simulations of one disaster type. Aggregates
   success rate, path length, reroute events, adaptation lag — the
   metrics that go directly into the research paper tables.

3. run_comparative_study()
   Runs run_experiment() for ALL four disaster types
   (earthquake, flood, fire, compound) under identical conditions and
   prints a side-by-side comparison table.
   This is the main experiment for the paper.

Usage
-----
    python test_dynamic_disaster.py
    python test_dynamic_disaster.py --mode single   --type fire      --agents 10 --steps 200
    python test_dynamic_disaster.py --mode experiment --type compound --runs 5
    python test_dynamic_disaster.py --mode compare  --agents 10 --steps 200 --runs 5
"""

import sys
import os
import random
import argparse

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.insert(0, root_dir)

from hybrid_spatial_index import HybridTokyoGraph, load_cached
from swarm_simulation_dynamic import DynamicSwarmSimulation

# ---------------------------------------------------------------------------
# Graph path — adjust to your local setup
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

def _find_graph_path() -> str:
    """TOKYO_GRAPH env var wins; otherwise look in the usual places."""
    candidates = [
        os.environ.get("TOKYO_GRAPH"),
        os.path.join(script_dir, "tokyo_graph.json"),
        os.path.join(script_dir, "..", "dataset", "tokyo_graph.json"),
        os.path.join(script_dir, "..", "dataset", "tokyo_graph", "tokyo_graph.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "tokyo_graph.json not found. Unzip dataset/tokyo_graph.zip next to this "
        "script or set the TOKYO_GRAPH environment variable.")

GRAPH_PATH = _find_graph_path()

# ---------------------------------------------------------------------------
# Shared disaster parameters — same across all experiment modes so results
# are directly comparable
# ---------------------------------------------------------------------------
DISASTER_CENTRE   = (35.68, 139.76)
DISASTER_RADIUS_M = 2000
DISASTER_SEVERITY = 0.7
WIND_BEARING      = 45.0   # NE wind for fire scenarios


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------

def load_fresh_graph() -> "HybridTokyoGraph":
    """
    Undamaged graph for one run. The file is parsed and indexed once per
    process; each call returns an independent fork, so damage cannot leak
    between runs and there is no ~17 s reload per run.
    """
    return load_cached(GRAPH_PATH)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    disaster_type:  str   = "earthquake",
    num_agents:     int   = 10,
    steps:          int   = 200,
    min_dist:       float = 1000,
    max_dist:       float = 4000,
    seed:           int   = None,
    verbose:        bool  = True,
) -> dict:
    """
    Run one dynamic simulation and return the adaptation_metrics() dict.
    """
    print(f"\n{'#'*60}")
    print(f"  DYNAMIC SIMULATION — {disaster_type.upper()}")
    print(f"  Agents: {num_agents}  Steps: {steps}  Seed: {seed}")
    print(f"  Centre: {DISASTER_CENTRE}  Severity: {DISASTER_SEVERITY}  "
          f"Radius: {DISASTER_RADIUS_M}m")
    print(f"{'#'*60}\n")

    graph = load_fresh_graph()

    sim = DynamicSwarmSimulation(
        graph                   = graph,
        num_agents              = num_agents,
        min_start_goal_distance = min_dist,
        max_start_goal_distance = max_dist,
        disaster_type           = disaster_type,
        disaster_centre         = DISASTER_CENTRE,
        disaster_radius_m       = DISASTER_RADIUS_M,
        disaster_severity       = DISASTER_SEVERITY,
        wind_bearing_deg        = WIND_BEARING,
        seed                    = seed,
    )

    sim.run(steps=steps)

    if verbose:
        sim.summary()

    return sim.adaptation_metrics()


# ---------------------------------------------------------------------------
# Multi-run experiment (one disaster type)
# ---------------------------------------------------------------------------

def run_experiment(
    disaster_type:  str   = "earthquake",
    n_runs:         int   = 5,
    num_agents:     int   = 10,
    steps:          int   = 200,
    min_dist:       float = 1000,
    max_dist:       float = 4000,
) -> dict:
    """
    Run n_runs independent simulations of one disaster type.
    Returns aggregate statistics dict.
    """
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {disaster_type.upper()} — {n_runs} runs × {num_agents} agents × {steps} steps")
    print(f"{'#'*60}\n")

    results = []
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{n_runs}  (seed={i*42})")
        print(f"{'='*60}")
        m = run_single(
            disaster_type = disaster_type,
            num_agents    = num_agents,
            steps         = steps,
            min_dist      = min_dist,
            max_dist      = max_dist,
            seed          = i * 42,
            verbose       = False,
        )
        results.append(m)
        print(f"  → success={m['reached_goal']}/{m['total_agents']} "
              f"({m['success_rate']:.0%})  "
              f"reroutes={m['total_reroute_events']}  "
              f"adapt_lag={m['mean_adaptation_lag']:.1f}")

    # Aggregate
    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def _std(key):
        vals = [r[key] for r in results if r[key] is not None]
        if len(vals) < 2:
            return 0.0
        mu = sum(vals) / len(vals)
        return round((sum((v-mu)**2 for v in vals) / len(vals)) ** 0.5, 3)

    agg = {
        "disaster_type":          disaster_type,
        "n_runs":                 n_runs,
        "num_agents":             num_agents,
        "steps":                  steps,
        "mean_success_rate":      _mean("success_rate"),
        "std_success_rate":       _std("success_rate"),
        "mean_reroutes":          _mean("mean_reroutes_per_agent"),
        "std_reroutes":           _std("mean_reroutes_per_agent"),
        "mean_adaptation_lag":    _mean("mean_adaptation_lag"),
        "std_adaptation_lag":     _std("mean_adaptation_lag"),
        "mean_path_length":       _mean("mean_path_length"),
        "mean_goal_step":         _mean("mean_goal_step"),
        "mean_disaster_events":   _mean("disaster_events"),
    }

    _print_experiment_summary(agg, results)
    return agg


def _print_experiment_summary(agg: dict, results: list):
    print(f"\n{'#'*60}")
    print(f"AGGREGATE — {agg['disaster_type'].upper()} "
          f"({agg['n_runs']} runs × {agg['num_agents']} agents)")
    print(f"{'#'*60}")
    sr_pct = round((agg['mean_success_rate'] or 0) * 100, 1)
    sr_std = round((agg['std_success_rate'] or 0) * 100, 1)
    print(f"  Success rate     : {sr_pct:.1f}% ± {sr_std:.1f}%")
    print(f"  Reroutes/agent   : {agg['mean_reroutes']:.2f} ± {agg['std_reroutes']:.2f}")
    print(f"  Adaptation lag   : {agg['mean_adaptation_lag']:.1f} ± {agg['std_adaptation_lag']:.1f} steps")
    if agg['mean_path_length']:
        print(f"  Mean path length : {agg['mean_path_length']:.1f} steps")
    if agg['mean_goal_step']:
        print(f"  Mean goal step   : {agg['mean_goal_step']:.1f}")
    print(f"  Disaster events  : {agg['mean_disaster_events']:.1f} per run (avg)")

    # Per-run table
    print(f"\n  {'Run':<5} {'Success':<10} {'Reroutes':<12} {'Lag':<10} {'PathLen':<10} {'DisEvents'}")
    print("  " + "-"*60)
    for i, r in enumerate(results):
        sr   = f"{r['success_rate']:.0%}"
        rr   = f"{r['mean_reroutes_per_agent']:.2f}"
        lag  = f"{r['mean_adaptation_lag']:.1f}"
        pl   = f"{r['mean_path_length']:.1f}" if r['mean_path_length'] else "-"
        devs = str(r['disaster_events'])
        print(f"  {i+1:<5} {sr:<10} {rr:<12} {lag:<10} {pl:<10} {devs}")
    print(f"{'#'*60}\n")


# ---------------------------------------------------------------------------
# Comparative study — all four disaster types
# ---------------------------------------------------------------------------

def run_comparative_study(
    n_runs:     int = 5,
    num_agents: int = 10,
    steps:      int = 200,
    min_dist:   float = 1000,
    max_dist:   float = 4000,
) -> dict:
    """
    Run all four disaster types under identical conditions and print a
    side-by-side comparison table for the research paper.

    Returns dict keyed by disaster_type → aggregate metrics dict.
    """
    disaster_types = ["earthquake", "flood", "fire", "compound"]
    all_agg = {}

    for dtype in disaster_types:
        agg = run_experiment(
            disaster_type = dtype,
            n_runs        = n_runs,
            num_agents    = num_agents,
            steps         = steps,
            min_dist      = min_dist,
            max_dist      = max_dist,
        )
        all_agg[dtype] = agg

    # Side-by-side comparison table
    print("\n" + "="*80)
    print("  COMPARATIVE STUDY — SNN ADAPTATION ACROSS DISASTER TYPES")
    print("="*80)
    print(f"\n  Conditions: {n_runs} runs × {num_agents} agents × {steps} steps")
    print(f"  Disaster: severity={DISASTER_SEVERITY}, radius={DISASTER_RADIUS_M}m, "
          f"centre={DISASTER_CENTRE}")
    print()

    # Header
    col_w = 14
    header = (f"  {'Metric':<28}" +
              "".join(f"{dt.upper():<{col_w}}" for dt in disaster_types))
    print(header)
    print("  " + "-" * (28 + col_w * len(disaster_types)))

    def row(label, key, fmt="{:.1f}", pct=False):
        vals = []
        for dt in disaster_types:
            v = all_agg[dt].get(key)
            if v is None:
                vals.append("-".center(col_w))
            elif pct:
                vals.append(f"{v*100:.1f}%".ljust(col_w))
            else:
                vals.append(fmt.format(v).ljust(col_w))
        print(f"  {label:<28}" + "".join(vals))

    row("Success rate",        "mean_success_rate",    pct=True)
    row("Std (success)",       "std_success_rate",     pct=True)
    row("Reroutes/agent",      "mean_reroutes",        "{:.2f}")
    row("Std (reroutes)",      "std_reroutes",         "{:.2f}")
    row("Adaptation lag (steps)", "mean_adaptation_lag", "{:.1f}")
    row("Std (adapt lag)",     "std_adaptation_lag",   "{:.1f}")
    row("Mean path length",    "mean_path_length",     "{:.1f}")
    row("Mean goal step",      "mean_goal_step",       "{:.1f}")
    row("Disaster events/run", "mean_disaster_events", "{:.1f}")

    print("  " + "-" * (28 + col_w * len(disaster_types)))
    print("\n  Interpretation guide:")
    print("    Success rate    : fraction of agents that reached their goal")
    print("    Reroutes/agent  : mean times an agent's path was invalidated by new damage")
    print("    Adaptation lag  : mean steps spent in stall/cycle handler after a reroute")
    print("    Path length     : mean steps taken by successful agents")
    print("    Goal step       : mean simulation step at which goal was reached")
    print("    Disaster events : total events fired (aftershocks/flood expansions/etc.)")
    print("=" * 80 + "\n")

    return all_agg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dynamic disaster SNN experiment runner")
    p.add_argument("--mode",   choices=["single", "experiment", "compare"],
                   default="single")
    p.add_argument("--type",   choices=["earthquake", "flood", "fire", "compound"],
                   default="earthquake",   help="Disaster type (single/experiment mode)")
    p.add_argument("--agents", type=int, default=10,   help="Number of agents")
    p.add_argument("--steps",  type=int, default=200,  help="Max simulation steps")
    p.add_argument("--runs",   type=int, default=5,    help="Runs per disaster type (experiment/compare)")
    p.add_argument("--seed",   type=int, default=None, help="RNG seed (single mode)")
    p.add_argument("--min-dist", type=float, default=1000)
    p.add_argument("--max-dist", type=float, default=4000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "single":
        run_single(
            disaster_type = args.type,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
            seed          = args.seed,
        )
    elif args.mode == "experiment":
        run_experiment(
            disaster_type = args.type,
            n_runs        = args.runs,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
        )
    elif args.mode == "compare":
        run_comparative_study(
            n_runs     = args.runs,
            num_agents = args.agents,
            steps      = args.steps,
            min_dist   = args.min_dist,
            max_dist   = args.max_dist,
        )

    elif args.mode == "experiment":
        run_experiment(
            disaster_type = args.type,
            n_runs        = args.runs,
            num_agents    = args.agents,
            steps         = args.steps,
            min_dist      = args.min_dist,
            max_dist      = args.max_dist,
        )

    elif args.mode == "compare":
        run_comparative_study(
            n_runs     = args.runs,
            num_agents = args.agents,
            steps      = args.steps,
            min_dist   = args.min_dist,
            max_dist   = args.max_dist,
        )
