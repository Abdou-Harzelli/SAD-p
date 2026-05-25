import numpy as np
import networkx as nx
from typing import Tuple, List

from graph_generator import CRITERIA


def gini_coefficient(values: np.ndarray) -> float:

    values = np.asarray(values, dtype=float)

    if len(values) == 0 or values.sum() == 0:
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    mean_val = sorted_vals.mean()

    if mean_val == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * sorted_vals.sum()) - (n + 1.0) / n

    return max(0.0, gini)


def gini_weights(G: nx.DiGraph, verbose: bool = False) -> np.ndarray:

    edges = list(G.edges(data=True))
    n_edges = len(edges)

    if n_edges == 0:
        return np.ones(len(CRITERIA)) / len(CRITERIA)

    gini_values = []

    for criterion in CRITERIA:
        midpoints = np.array([
            (data[f"{criterion}_min"] + data[f"{criterion}_max"]) / 2.0
            for _, _, data in edges
        ])

        g = gini_coefficient(midpoints)
        gini_values.append(g)

    gini_values = np.array(gini_values)

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
            print(f"  {criterion:15s}: Gini = {gini_values[i]:.4f}    Wo = {weights[i]:.4f}")
        print(f"  Sum of weights: {weights.sum():.4f}")
        print()

    return weights


def entropy_weights(G: nx.DiGraph, verbose: bool = False) -> np.ndarray:

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

        total = midpoints.sum()
        if total == 0:
            entropies.append(0.0)
            continue

        proportions = midpoints / total
        proportions = proportions[proportions > 0]

        e = -np.sum(proportions * np.log(proportions)) / np.log(n_edges)
        entropies.append(e)

    entropies = np.array(entropies)
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
            print(f"  {criterion:15s}: Entropy = {entropies[i]:.4f}    Wo = {weights[i]:.4f}")
        print()

    return weights


if __name__ == "__main__":
    from graph_generator import generate_urban_graph

    G = generate_urban_graph(30, connectivity=0.2, seed=42)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    wo = gini_weights(G, verbose=True)
    print(f"Objective Weight Vector (Gini): {np.round(wo, 4)}")
