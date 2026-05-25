#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Multi-Criteria Dynamic Route Optimization                                  ║
║  ─────────────────────────────────────────                                  ║
║  Hybrid AHP + Gini + α-Hurwicz + Dijkstra Pipeline                         ║
║  Domain: Urban Road Navigation                                              ║
║                                                                              ║
║  5-Step Pipeline:                                                            ║
║    1. Subjective Weighting (AHP)                                             ║
║    2. Objective Weighting (Gini Index)                                       ║
║    3. Weight Fusion (β coefficient)                                          ║
║    4. Risk Resolution (α-Hurwicz)                                            ║
║    5. Cost Calculation & Optimization (Dijkstra)                             ║
║                                                                              ║
║  Compares 4 approaches:                                                      ║
║    A) Plain Dijkstra (baseline)                                              ║
║    B) AHP + α-Hurwicz + Dijkstra                                            ║
║    C) Gini + α-Hurwicz + Dijkstra                                           ║
║    D) AHP + Gini + α-Hurwicz + Dijkstra (full pipeline)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import numpy as np

# Add project root to path
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
    """Run the complete multi-criteria routing analysis."""
    print("╔" + "═" * 78 + "╗")
    print("║  Multi-Criteria Dynamic Route Optimization — Full Analysis" + " " * 19 + "║")
    print("║  Domain: Urban Road Navigation" + " " * 47 + "║")
    print("╚" + "═" * 78 + "╝")

    # ── Parameters ────────────────────────────────────────────────────────
    ALPHA = 0.5      # Hurwicz optimism coefficient
    BETA = 0.5       # Weight fusion coefficient
    SEED = 42
    N_DEMO = 40      # Node count for the demo graph
    CONNECTIVITY = 0.15

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: STEP-BY-STEP PIPELINE DEMONSTRATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("  PART 1: STEP-BY-STEP PIPELINE DEMONSTRATION")
    print("▓" * 80)

    # Generate demo graph
    G = generate_urban_graph(N_DEMO, connectivity=CONNECTIVITY, seed=SEED)
    info = get_graph_info(G)
    print(f"\n  Graph Generated: |V| = {info['nodes']}, |E| = {info['edges']}, "
          f"density = {info['density']:.4f}")
    print(f"  Strongly connected: {info['strongly_connected']}")
    print(f"  Avg out-degree: {info['avg_out_degree']:.1f}")

    source, target = 0, N_DEMO - 1

    # ── Step 1: AHP ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 1: Subjective Weighting (AHP)")
    print("─" * 60)
    A = get_default_comparison_matrix()
    ws, ci, cr, is_consistent = ahp_weights(A, verbose=True)

    # Plot AHP analysis
    plot_ahp_matrix(A, ws, cr)

    # ── Step 2: Gini ─────────────────────────────────────────────────────
    print("─" * 60)
    print("  STEP 2: Objective Weighting (Gini Index)")
    print("─" * 60)
    wo = gini_weights(G, verbose=True)

    # ── Step 3: Fusion ───────────────────────────────────────────────────
    print("─" * 60)
    print("  STEP 3: Weight Fusion (β = {})".format(BETA))
    print("─" * 60)
    W = fuse_weights(ws, wo, beta=BETA, verbose=True)

    # Plot weight comparison
    plot_weight_comparison(ws, wo, W, beta=BETA)

    # ── Step 4 & 5: Hurwicz + Dijkstra ───────────────────────────────────
    print("─" * 60)
    print("  STEPS 4-5: α-Hurwicz Risk Resolution + Dijkstra Optimization")
    print("─" * 60)
    print(f"  α = {ALPHA} (Hurwicz optimism)")
    print(f"  Source = {source}, Target = {target}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: COMPARATIVE ANALYSIS (4 APPROACHES)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("  PART 2: COMPARATIVE ANALYSIS OF ALL 4 APPROACHES")
    print("▓" * 80)

    results = compare_approaches(G, source, target, alpha=ALPHA, beta=BETA, verbose=True)

    # Plot graph with all paths
    paths = {name: res["path"] for name, res in results.items()}
    plot_graph_with_paths(G, paths, source, target)

    # Plot per-criterion comparison
    plot_criterion_comparison(results)

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("  PART 3: SENSITIVITY ANALYSIS")
    print("▓" * 80)

    # α sensitivity
    alpha_results = sensitivity_alpha(G, source, target, ws, wo, beta=BETA)
    plot_sensitivity_alpha(alpha_results)

    # β sensitivity
    beta_results = sensitivity_beta(G, source, target, ws, wo, alpha=ALPHA)
    plot_sensitivity_beta(beta_results)

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: SCALABILITY EXPERIMENTS (Small N → Large N)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("  PART 4: SCALABILITY EXPERIMENTS  G(V,E), |V|=N, |E|=M")
    print("▓" * 80)

    sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]
    exp_results = run_scalability_experiment(
        sizes=sizes,
        alpha=ALPHA,
        beta=BETA,
        avg_degree=6,
        n_trials=5,
        seed=SEED,
    )

    # Plot scalability
    plot_scalability(exp_results, sizes, n_trials=5)

    # ══════════════════════════════════════════════════════════════════════
    # PART 5: SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("  PART 5: SUMMARY & CONCLUSIONS")
    print("▓" * 80)

    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        KEY FINDINGS                                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  1. PLAIN DIJKSTRA optimizes only distance — ignores safety, time,     │
  │     and congestion, leading to potentially unsafe or slow routes.       │
  │                                                                         │
  │  2. AHP + HURWICZ + DIJKSTRA incorporates subjective expert judgment   │
  │     and handles uncertainty, producing balanced routes based on the     │
  │     decision-maker's priorities.                                        │
  │                                                                         │
  │  3. GINI + HURWICZ + DIJKSTRA is purely data-driven — criteria with    │
  │     more variance (more discriminating power) get higher weight,       │
  │     which is unbiased but may not match the user's priorities.         │
  │                                                                         │
  │  4. FULL PIPELINE (AHP + Gini + Hurwicz + Dijkstra) combines the      │
  │     best of both: subjective priorities + objective data analysis +    │
  │     uncertainty handling, producing the most robust optimal routes.    │
  │                                                                         │
  │  5. SCALABILITY: All approaches have similar O((V+E) log V) time      │
  │     complexity; the overhead of multi-criteria cost computation is     │
  │     constant per edge and does not change the asymptotic behavior.    │
  │                                                                         │
  │  6. α-HURWICZ provides a natural dial between optimistic and          │
  │     pessimistic decision-making under uncertainty.                      │
  │                                                                         │
  │  7. β-FUSION allows smooth interpolation between subjective           │
  │     expertise (AHP) and objective data analysis (Gini).                │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    print("  All figures saved to: results/")
    print("  " + "═" * 60)
    print("  Analysis complete. ✓")
    print()


if __name__ == "__main__":
    main()
