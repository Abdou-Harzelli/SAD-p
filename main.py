#!/usr/bin/env python3

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_generator import generate_urban_graph, generate_scaled_graph, get_graph_info, CRITERIA
from ahp import ahp_weights, get_default_comparison_matrix, get_alternative_comparison_matrices
from gini_weights import gini_weights
from pipeline import fuse_weights, dijkstra_simple, dijkstra_multicriteria
from experiments import (
    compare_approaches,
    run_scalability_experiment,
    sensitivity_alpha,
    sensitivity_beta,
)
from visualizations import (
    plot_graph_with_paths,
    plot_scalability,
    plot_sensitivity_alpha,
    plot_sensitivity_beta,
    plot_criterion_comparison,
    plot_weight_comparison,
    plot_ahp_matrix,
)


def main():
    print("+" + "=" * 78 + "+")
    print("|  Multi-Criteria Dynamic Route Optimization -- Full Analysis" + " " * 18 + "|")
    print("|  Domain: Urban Road Navigation" + " " * 47 + "|")
    print("+" + "=" * 78 + "+")

    ALPHA = 0.5
    BETA = 0.5
    SEED = 42
    N_DEMO = 40
    CONNECTIVITY = 0.15

    print("\n" + "#" * 80)
    print("  PART 1: STEP-BY-STEP PIPELINE DEMONSTRATION")
    print("#" * 80)

    G = generate_urban_graph(N_DEMO, connectivity=CONNECTIVITY, seed=SEED)
    info = get_graph_info(G)
    print(f"\n  Graph Generated: |V| = {info['nodes']}, |E| = {info['edges']}, "
          f"density = {info['density']:.4f}")
    print(f"  Strongly connected: {info['strongly_connected']}")
    print(f"  Avg out-degree: {info['avg_out_degree']:.1f}")

    source, target = 0, N_DEMO - 1

    print("\n" + "-" * 60)
    print("  STEP 1: Subjective Weighting (AHP)")
    print("-" * 60)
    A = get_default_comparison_matrix()
    ws, ci, cr, is_consistent = ahp_weights(A, verbose=True)

    plot_ahp_matrix(A, ws, cr)

    print("-" * 60)
    print("  STEP 2: Objective Weighting (Gini Index)")
    print("-" * 60)
    wo = gini_weights(G, verbose=True)

    print("-" * 60)
    print("  STEP 3: Weight Fusion (beta = {})".format(BETA))
    print("-" * 60)
    W = fuse_weights(ws, wo, beta=BETA, verbose=True)

    plot_weight_comparison(ws, wo, W, beta=BETA)

    print("-" * 60)
    print("  STEPS 4-5: alpha-Hurwicz Risk Resolution + Dijkstra Optimization")
    print("-" * 60)
    print(f"  alpha = {ALPHA} (Hurwicz optimism)")
    print(f"  Source = {source}, Target = {target}")

    print("\n" + "#" * 80)
    print("  PART 2: COMPARATIVE ANALYSIS OF ALL 4 APPROACHES")
    print("#" * 80)

    results = compare_approaches(G, source, target, alpha=ALPHA, beta=BETA, verbose=True)

    paths = {name: res["path"] for name, res in results.items()}
    plot_graph_with_paths(G, paths, source, target)

    plot_criterion_comparison(results)

    print("\n" + "#" * 80)
    print("  PART 3: SENSITIVITY ANALYSIS")
    print("#" * 80)

    alpha_results = sensitivity_alpha(G, source, target, ws, wo, beta=BETA)
    plot_sensitivity_alpha(alpha_results)

    beta_results = sensitivity_beta(G, source, target, ws, wo, alpha=ALPHA)
    plot_sensitivity_beta(beta_results)

    print("\n" + "#" * 80)
    print("  PART 4: SCALABILITY EXPERIMENTS  G(V,E), |V|=N, |E|=M")
    print("#" * 80)

    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]
    exp_results = run_scalability_experiment(
        sizes=sizes,
        alpha=ALPHA,
        beta=BETA,
        avg_degree=6,
        n_trials=5,
        seed=SEED,
    )

    plot_scalability(exp_results, sizes, n_trials=5)

    print("\n" + "#" * 80)
    print("  PART 5: SUMMARY & CONCLUSIONS")
    print("#" * 80)

    print("""
   +-------------------------------------------------------------------------+
   |                        KEY FINDINGS                                     |
   +-------------------------------------------------------------------------+
   |                                                                         |
   |  1. PLAIN DIJKSTRA optimizes only distance -- ignores safety, time,    |
   |     and congestion, leading to potentially unsafe or slow routes.       |
   |                                                                         |
   |  2. AHP + HURWICZ + DIJKSTRA incorporates subjective expert judgment   |
   |     and handles uncertainty, producing balanced routes based on the     |
   |     decision-maker's priorities.                                        |
   |                                                                         |
   |  3. GINI + HURWICZ + DIJKSTRA is purely data-driven -- criteria with   |
   |     more variance (more discriminating power) get higher weight,       |
   |     which is unbiased but may not match the user's priorities.         |
   |                                                                         |
   |  4. FULL PIPELINE (AHP + Gini + Hurwicz + Dijkstra) combines the      |
   |     best of both: subjective priorities + objective data analysis +    |
   |     uncertainty handling, producing the most robust optimal routes.    |
   |                                                                         |
   |  5. SCALABILITY: All approaches have similar O((V+E) log V) time      |
   |     complexity; the overhead of multi-criteria cost computation is     |
   |     constant per edge and does not change the asymptotic behavior.    |
   |                                                                         |
   |  6. alpha-HURWICZ provides a natural dial between optimistic and       |
   |     pessimistic decision-making under uncertainty.                      |
   |                                                                         |
   |  7. beta-FUSION allows smooth interpolation between subjective         |
   |     expertise (AHP) and objective data analysis (Gini).                |
   |                                                                         |
   +-------------------------------------------------------------------------+
    """)

    print("  All figures saved to: results/")
    print("  " + "=" * 60)
    print("  Analysis complete.")
    print()


if __name__ == "__main__":
    main()
