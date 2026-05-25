import time
import numpy as np
import networkx as nx
from tabulate import tabulate
from typing import Dict, List, Tuple

from graph_generator import (
    generate_urban_graph,
    generate_scaled_graph,
    get_graph_info,
    CRITERIA,
)
from ahp import ahp_weights, get_default_comparison_matrix
from gini_weights import gini_weights
from pipeline import (
    fuse_weights,
    dijkstra_simple,
    dijkstra_multicriteria,
    run_pipeline_full,
    run_pipeline_ahp_hurwicz,
    run_pipeline_gini_hurwicz,
)


def compare_approaches(
    G: nx.DiGraph,
    source: int,
    target: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    verbose: bool = True,
) -> Dict:
    A = get_default_comparison_matrix()
    ws, ci, cr, is_consistent = ahp_weights(A, verbose=False)

    wo = gini_weights(G, verbose=False)

    results = {}

    t0 = time.perf_counter()
    path1, cost1, det1 = dijkstra_simple(G, source, target)
    t1 = time.perf_counter()
    results["Dijkstra"] = {
        "path": path1,
        "cost": cost1,
        "details": det1,
        "time_ms": (t1 - t0) * 1000,
        "path_length": len(path1),
    }

    t0 = time.perf_counter()
    path2, cost2, det2 = run_pipeline_ahp_hurwicz(G, source, target, ws, alpha)
    t1 = time.perf_counter()
    results["AHP+Hurwicz+Dijkstra"] = {
        "path": path2,
        "cost": cost2,
        "details": det2,
        "time_ms": (t1 - t0) * 1000,
        "path_length": len(path2),
    }

    t0 = time.perf_counter()
    path3, cost3, det3 = run_pipeline_gini_hurwicz(G, source, target, wo, alpha)
    t1 = time.perf_counter()
    results["Gini+Hurwicz+Dijkstra"] = {
        "path": path3,
        "cost": cost3,
        "details": det3,
        "time_ms": (t1 - t0) * 1000,
        "path_length": len(path3),
    }

    t0 = time.perf_counter()
    path4, cost4, det4 = run_pipeline_full(G, source, target, ws, wo, alpha, beta)
    t1 = time.perf_counter()
    results["AHP+Gini+Hurwicz+Dijkstra"] = {
        "path": path4,
        "cost": cost4,
        "details": det4,
        "time_ms": (t1 - t0) * 1000,
        "path_length": len(path4),
    }

    if verbose:
        _print_comparison(results, ws, wo, alpha, beta, ci, cr)

    return results


def _print_comparison(results, ws, wo, alpha, beta, ci, cr):
    print("\n" + "=" * 80)
    print("  COMPARATIVE ANALYSIS OF ROUTE OPTIMIZATION APPROACHES")
    print("  Domain: Urban Road Navigation")
    print("=" * 80)

    print(f"\n  AHP Weights (Ws):  {np.round(ws, 4)}  (CR = {cr:.4f})")
    print(f"  Gini Weights (Wo): {np.round(wo, 4)}")
    print(f"  alpha (Hurwicz):   {alpha}")
    print(f"  beta (Fusion):     {beta}")

    headers = ["Approach", "Path", "#Edges", "Cost", "Time (ms)"]
    rows = []
    for name, res in results.items():
        path_str = "->".join(map(str, res["path"][:8]))
        if len(res["path"]) > 8:
            path_str += f"->...-> {res['path'][-1]}"
        rows.append([
            name,
            path_str,
            res["path_length"] - 1 if res["path_length"] > 0 else 0,
            f"{res['cost']:.4f}",
            f"{res['time_ms']:.2f}",
        ])

    print(f"\n{'-' * 80}")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    print(f"\n{'-' * 80}")
    print("  Per-Criterion Performance (Midpoint Totals Along Path)")
    print(f"{'-' * 80}")

    crit_headers = ["Approach"] + [c.replace("_", " ").title() for c in CRITERIA]
    crit_rows = []
    for name, res in results.items():
        if "criterion_midpoints" in res["details"]:
            row = [name]
            for c in CRITERIA:
                row.append(f"{res['details']['criterion_midpoints'][c]:.3f}")
            crit_rows.append(row)

    print(tabulate(crit_rows, headers=crit_headers, tablefmt="fancy_grid"))
    print()


def run_scalability_experiment(
    sizes: List[int] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    avg_degree: int = 6,
    n_trials: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    if sizes is None:
        sizes = [10, 20, 50, 100, 200, 500, 1000]

    rng = np.random.RandomState(seed)

    approaches = [
        "Dijkstra",
        "AHP+Hurwicz+Dijkstra",
        "Gini+Hurwicz+Dijkstra",
        "AHP+Gini+Hurwicz+Dijkstra",
    ]

    exp_results = {
        approach: {"sizes": [], "times": [], "costs": [], "path_lengths": [], "edges": []}
        for approach in approaches
    }

    A = get_default_comparison_matrix()
    ws, _, _, _ = ahp_weights(A)

    if verbose:
        print("\n" + "=" * 80)
        print("  SCALABILITY EXPERIMENT: Small N -> Large N")
        print("=" * 80)

    for n in sizes:
        if verbose:
            print(f"\n  N = {n} nodes ...", end=" ", flush=True)

        G = generate_scaled_graph(n, avg_degree=avg_degree, seed=seed)
        info = get_graph_info(G)

        if verbose:
            print(f"(|E| = {info['edges']}, density = {info['density']:.4f})")

        wo = gini_weights(G)

        W_full = fuse_weights(ws, wo, beta)

        nodes = list(G.nodes())
        for trial in range(n_trials):
            while True:
                s = rng.choice(nodes)
                t = rng.choice(nodes)
                if s != t and nx.has_path(G, s, t):
                    break

            for approach in approaches:
                t0 = time.perf_counter()

                if approach == "Dijkstra":
                    path, cost, det = dijkstra_simple(G, s, t)
                elif approach == "AHP+Hurwicz+Dijkstra":
                    path, cost, det = run_pipeline_ahp_hurwicz(G, s, t, ws, alpha)
                elif approach == "Gini+Hurwicz+Dijkstra":
                    path, cost, det = run_pipeline_gini_hurwicz(G, s, t, wo, alpha)
                else:
                    path, cost, det = dijkstra_multicriteria(G, s, t, W_full, alpha)

                elapsed = (time.perf_counter() - t0) * 1000

                exp_results[approach]["sizes"].append(n)
                exp_results[approach]["times"].append(elapsed)
                exp_results[approach]["costs"].append(cost)
                exp_results[approach]["path_lengths"].append(len(path))
                exp_results[approach]["edges"].append(info["edges"])

    if verbose:
        _print_scalability_summary(exp_results, sizes, n_trials)

    return exp_results


def _print_scalability_summary(exp_results, sizes, n_trials):
    print(f"\n{'=' * 80}")
    print("  SCALABILITY SUMMARY (average over trials)")
    print(f"{'=' * 80}\n")

    approaches = list(exp_results.keys())

    headers = ["N (nodes)"] + approaches
    rows = []
    for n in sizes:
        row = [n]
        for approach in approaches:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            times = [exp_results[approach]["times"][i] for i in indices]
            row.append(f"{np.mean(times):.3f}")
        rows.append(row)

    print("  Avg Execution Time (ms):")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    print(f"\n  Avg Composite Path Cost:")
    rows = []
    for n in sizes:
        row = [n]
        for approach in approaches:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            costs = [exp_results[approach]["costs"][i] for i in indices]
            row.append(f"{np.mean(costs):.4f}")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    print(f"\n  Avg Path Length (nodes):")
    rows = []
    for n in sizes:
        row = [n]
        for approach in approaches:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            lengths = [exp_results[approach]["path_lengths"][i] for i in indices]
            row.append(f"{np.mean(lengths):.1f}")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    print()


def sensitivity_alpha(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    wo: np.ndarray,
    beta: float = 0.5,
    alphas: List[float] = None,
    verbose: bool = True,
) -> Dict:
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    W = fuse_weights(ws, wo, beta)
    results = {"alpha": [], "cost": [], "path": [], "path_length": []}
    per_criterion = {c: [] for c in CRITERIA}

    for a in alphas:
        path, cost, det = dijkstra_multicriteria(G, source, target, W, a)
        results["alpha"].append(a)
        results["cost"].append(cost)
        results["path"].append(path)
        results["path_length"].append(len(path))

        for c in CRITERIA:
            per_criterion[c].append(det["criterion_midpoints"][c])

    results["per_criterion"] = per_criterion

    if verbose:
        print(f"\n{'=' * 60}")
        print("  SENSITIVITY ANALYSIS: alpha (Hurwicz Optimism)")
        print(f"{'=' * 60}")
        headers = ["alpha", "Cost", "#Edges"] + [c.replace("_", " ").title() for c in CRITERIA]
        rows = []
        for i, a in enumerate(alphas):
            row = [
                f"{a:.1f}",
                f"{results['cost'][i]:.4f}",
                results["path_length"][i] - 1,
            ]
            for c in CRITERIA:
                row.append(f"{per_criterion[c][i]:.3f}")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    return results


def sensitivity_beta(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    wo: np.ndarray,
    alpha: float = 0.5,
    betas: List[float] = None,
    verbose: bool = True,
) -> Dict:
    if betas is None:
        betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {"beta": [], "cost": [], "path": [], "path_length": [], "weights": []}
    per_criterion = {c: [] for c in CRITERIA}

    for b in betas:
        W = fuse_weights(ws, wo, b)
        path, cost, det = dijkstra_multicriteria(G, source, target, W, alpha)
        results["beta"].append(b)
        results["cost"].append(cost)
        results["path"].append(path)
        results["path_length"].append(len(path))
        results["weights"].append(W.copy())

        for c in CRITERIA:
            per_criterion[c].append(det["criterion_midpoints"][c])

    results["per_criterion"] = per_criterion

    if verbose:
        print(f"\n{'=' * 60}")
        print("  SENSITIVITY ANALYSIS: beta (Weight Fusion)")
        print(f"{'=' * 60}")
        headers = ["beta", "Cost", "#Edges"] + [c.replace("_", " ").title() for c in CRITERIA]
        rows = []
        for i, b in enumerate(betas):
            row = [
                f"{b:.1f}",
                f"{results['cost'][i]:.4f}",
                results["path_length"][i] - 1,
            ]
            for c in CRITERIA:
                row.append(f"{per_criterion[c][i]:.3f}")
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    return results


if __name__ == "__main__":
    G = generate_urban_graph(30, connectivity=0.2, seed=42)
    compare_approaches(G, source=0, target=29, alpha=0.5, beta=0.5)

    run_scalability_experiment(
        sizes=[10, 20, 50, 100, 200],
        alpha=0.5,
        beta=0.5,
        n_trials=3,
    )
