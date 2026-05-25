"""
Step 2: Objective Weighting using Gini Index.

The Gini coefficient measures the inequality/dispersion of each criterion
across all edges of the graph. Higher variance → higher information content
→ higher objective weight.

This ensures that criteria with more discriminating power get boosted,
regardless of the subjective AHP preferences.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List

from graph_generator import CRITERIA


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of an array of values.

    Gini = 0  → perfect equality (all values identical)
    Gini = 1  → maximum inequality

    Formula (mean absolute difference):
        G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2 · n² · x̄)

    Parameters
    ----------
    values : np.ndarray
        1D array of non-negative values.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0 or values.sum() == 0:
        return 0.0

    # Sort for efficient computation
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    mean_val = sorted_vals.mean()

    if mean_val == 0:
        return 0.0

    # Efficient Gini using sorted values
    # G = (2 * Σᵢ (i+1) * xᵢ) / (n * Σᵢ xᵢ) - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * sorted_vals.sum()) - (n + 1.0) / n

    return max(0.0, gini)


def gini_weights(G: nx.DiGraph, verbose: bool = False) -> np.ndarray:
    """
    Compute objective weights for each criterion based on Gini index.

    For each criterion, we compute the Gini coefficient across all edges
    using the midpoint value (x_min + x_max) / 2 as the representative value.
    Higher Gini → more variance → more informative → higher weight.

    The weights are normalized to sum to 1.

    Parameters
    ----------
    G : nx.DiGraph
        Graph with multi-criteria edge attributes.
    verbose : bool
        Print intermediate results.

    Returns
    -------
    np.ndarray
        Objective weight vector Wo = [wo1, wo2, ..., won].
    """
    edges = list(G.edges(data=True))
    n_edges = len(edges)

    if n_edges == 0:
        return np.ones(len(CRITERIA)) / len(CRITERIA)

    gini_values = []

    for criterion in CRITERIA:
        # Extract midpoint values for this criterion across all edges
        midpoints = np.array([
            (data[f"{criterion}_min"] + data[f"{criterion}_max"]) / 2.0
            for _, _, data in edges
        ])

        g = gini_coefficient(midpoints)
        gini_values.append(g)

    gini_values = np.array(gini_values)

    # Normalize: higher Gini → higher weight
    total = gini_values.sum()
    if total == 0:
        weights = np.ones(len(CRITERIA)) / len(CRITERIA)
    else:
        weights = gini_values / total

    if verbose:
        print("=" * 50)
        print("Gini Index Analysis (Objective Weighting)")
        print("=" * 50)
        for i, criterion in enumerate(CRITERIA):
            print(f"  {criterion:15s}: Gini = {gini_values[i]:.4f}  →  Wo = {weights[i]:.4f}")
        print(f"  Sum of weights: {weights.sum():.4f}")
        print()

    return weights


def entropy_weights(G: nx.DiGraph, verbose: bool = False) -> np.ndarray:
    """
    Alternative objective weighting using Shannon Entropy method.
    Provided for comparison, but Gini is the primary method per the spec.

    Parameters
    ----------
    G : nx.DiGraph
        Graph with multi-criteria edge attributes.
    verbose : bool
        Print intermediate results.

    Returns
    -------
    np.ndarray
        Objective weight vector.
    """
    edges = list(G.edges(data=True))
    n_edges = len(edges)

    if n_edges == 0:
        return np.ones(len(CRITERIA)) / len(CRITERIA)

    entropies = []
    for criterion in CRITERIA:
        midpoints = np.array([
            (data[f"{criterion}_min"] + data[f"{criterion}_max"]) / 2.0
            for _, _, data in edges
        ])

        # Normalize to proportions
        total = midpoints.sum()
        if total == 0:
            entropies.append(0.0)
            continue

        proportions = midpoints / total
        proportions = proportions[proportions > 0]  # avoid log(0)

        # Shannon entropy
        e = -np.sum(proportions * np.log(proportions)) / np.log(n_edges)
        entropies.append(e)

    entropies = np.array(entropies)
    # Weight = 1 - entropy (lower entropy → more diverse → higher weight)
    divergence = 1.0 - entropies
    total = divergence.sum()

    if total == 0:
        weights = np.ones(len(CRITERIA)) / len(CRITERIA)
    else:
        weights = divergence / total

    if verbose:
        print("=" * 50)
        print("Entropy Analysis (Alternative Objective Weighting)")
        print("=" * 50)
        for i, criterion in enumerate(CRITERIA):
            print(f"  {criterion:15s}: Entropy = {entropies[i]:.4f}  →  Wo = {weights[i]:.4f}")
        print()

    return weights


if __name__ == "__main__":
    from graph_generator import generate_urban_graph

    G = generate_urban_graph(30, connectivity=0.2, seed=42)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    wo = gini_weights(G, verbose=True)
    print(f"Objective Weight Vector (Gini): {np.round(wo, 4)}")
