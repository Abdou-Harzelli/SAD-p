"""
Steps 3-5: Weight Fusion, Hurwicz Risk Resolution, and Dijkstra Optimization.

This module ties everything together:
  Step 3: Weight Fusion        — Blend AHP (subjective) and Gini (objective) weights
  Step 4: α-Hurwicz            — Resolve uncertainty intervals on each edge
  Step 5: Cost + Dijkstra      — Compute composite edge costs and find optimal path
"""

import heapq
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

from graph_generator import CRITERIA, NUM_CRITERIA


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       STEP 3: WEIGHT FUSION                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fuse_weights(
    ws: np.ndarray,
    wo: np.ndarray,
    beta: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fuse subjective (AHP) and objective (Gini) weight vectors.

    W = β · Ws + (1 - β) · Wo

    Parameters
    ----------
    ws : np.ndarray
        Subjective weight vector from AHP.
    wo : np.ndarray
        Objective weight vector from Gini Index.
    beta : float
        Blending coefficient in [0, 1].
        β = 1 → fully subjective; β = 0 → fully objective.
    verbose : bool
        Print intermediate results.

    Returns
    -------
    np.ndarray
        Comprehensive weight vector W.
    """
    assert len(ws) == len(wo), "Weight vectors must have same length."
    assert 0 <= beta <= 1, "Beta must be in [0, 1]."

    W = beta * ws + (1 - beta) * wo

    # Re-normalize to ensure sum = 1 (should already be close)
    W = W / W.sum()

    if verbose:
        print("=" * 50)
        print(f"Weight Fusion (β = {beta})")
        print("=" * 50)
        for i, c in enumerate(CRITERIA):
            print(f"  {c:15s}: Ws={ws[i]:.4f}  Wo={wo[i]:.4f}  →  W={W[i]:.4f}")
        print(f"  Sum = {W.sum():.4f}")
        print()

    return W


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    STEP 4: α-HURWICZ RISK RESOLUTION                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def hurwicz_value(x_min: float, x_max: float, alpha: float) -> float:
    """
    Compute the Hurwicz criterion value for a single interval.

    V = α · x_min + (1 - α) · x_max

    Parameters
    ----------
    x_min : float
        Best-case value.
    x_max : float
        Worst-case value.
    alpha : float
        Optimism coefficient in [0, 1].
        α = 1 → fully optimistic (use best case)
        α = 0 → fully pessimistic (use worst case)

    Returns
    -------
    float
        Expected performance value V.
    """
    return alpha * x_min + (1 - alpha) * x_max


def compute_hurwicz_edge(
    edge_data: Dict,
    alpha: float,
) -> np.ndarray:
    """
    Compute Hurwicz values for all criteria of a single edge.

    Parameters
    ----------
    edge_data : dict
        Edge attributes containing {criterion}_min and {criterion}_max.
    alpha : float
        Optimism coefficient.

    Returns
    -------
    np.ndarray
        Array of Hurwicz values V = [V1, V2, ..., Vn] for the edge.
    """
    V = np.zeros(NUM_CRITERIA)
    for k, criterion in enumerate(CRITERIA):
        x_min = edge_data[f"{criterion}_min"]
        x_max = edge_data[f"{criterion}_max"]
        V[k] = hurwicz_value(x_min, x_max, alpha)
    return V


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              STEP 5: COST CALCULATION & DIJKSTRA                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_edge_cost(
    edge_data: Dict,
    W: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute the composite cost Cij for a single edge.

    Cij = Σₖ wk · Vij_k

    where Vij_k is the Hurwicz value for criterion k on edge (i, j).

    Parameters
    ----------
    edge_data : dict
        Edge attributes.
    W : np.ndarray
        Comprehensive weight vector.
    alpha : float
        Hurwicz optimism coefficient.

    Returns
    -------
    float
        Composite edge cost.
    """
    V = compute_hurwicz_edge(edge_data, alpha)
    return float(np.dot(W, V))


def dijkstra_multicriteria(
    G: nx.DiGraph,
    source: int,
    target: int,
    W: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    """
    Run Dijkstra's algorithm with multi-criteria composite costs.

    Parameters
    ----------
    G : nx.DiGraph
        The urban road graph.
    source : int
        Source node.
    target : int
        Target node.
    W : np.ndarray
        Comprehensive weight vector (from fusion step).
    alpha : float
        Hurwicz optimism coefficient.

    Returns
    -------
    path : List[int]
        Ordered list of nodes in the optimal path.
    total_cost : float
        Total composite cost of the path.
    details : dict
        Per-criterion breakdown of the path.
    """
    n = G.number_of_nodes()

    # Initialize distances
    dist = {node: float("inf") for node in G.nodes()}
    prev = {node: None for node in G.nodes()}
    dist[source] = 0.0

    # Priority queue: (distance, node)
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in G.successors(u):
            if v in visited:
                continue

            edge_data = G[u][v]
            cost = compute_edge_cost(edge_data, W, alpha)

            new_dist = dist[u] + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    if path[0] != source:
        return [], float("inf"), {}

    # Compute per-criterion breakdown
    details = _compute_path_details(G, path, alpha)
    details["composite_cost"] = dist[target]

    return path, dist[target], details


def dijkstra_simple(
    G: nx.DiGraph,
    source: int,
    target: int,
) -> Tuple[List[int], float, Dict]:
    """
    Run plain Dijkstra using only the 'weight' attribute (average distance).

    This serves as the BASELINE comparison.

    Parameters
    ----------
    G : nx.DiGraph
    source : int
    target : int

    Returns
    -------
    path : List[int]
    total_cost : float
    details : dict
    """
    n = G.number_of_nodes()
    dist = {node: float("inf") for node in G.nodes()}
    prev = {node: None for node in G.nodes()}
    dist[source] = 0.0

    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in G.successors(u):
            if v in visited:
                continue

            weight = G[u][v]["weight"]
            new_dist = dist[u] + weight

            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    if path[0] != source:
        return [], float("inf"), {}

    details = _compute_path_details(G, path, alpha=0.5)
    details["composite_cost"] = dist[target]

    return path, dist[target], details


def _compute_path_details(
    G: nx.DiGraph,
    path: List[int],
    alpha: float,
) -> Dict:
    """Compute per-criterion totals along a path."""
    totals = {c: 0.0 for c in CRITERIA}
    hurwicz_totals = {c: 0.0 for c in CRITERIA}

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = G[u][v]
        for c in CRITERIA:
            mid = (data[f"{c}_min"] + data[f"{c}_max"]) / 2.0
            totals[c] += mid
            hurwicz_totals[c] += hurwicz_value(data[f"{c}_min"], data[f"{c}_max"], alpha)

    return {
        "path_length": len(path),
        "num_edges": len(path) - 1,
        "criterion_midpoints": totals,
        "criterion_hurwicz": hurwicz_totals,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     COMBINED PIPELINE FUNCTIONS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_pipeline_full(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    wo: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    """
    Full pipeline: AHP + Gini + α-Hurwicz + Dijkstra.

    Steps:
      3. Fuse weights W = β·Ws + (1-β)·Wo
      4-5. Run Dijkstra with Hurwicz-resolved costs
    """
    W = fuse_weights(ws, wo, beta)
    return dijkstra_multicriteria(G, source, target, W, alpha)


def run_pipeline_ahp_hurwicz(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    """
    Pipeline: AHP + α-Hurwicz + Dijkstra (no Gini).

    Uses only subjective weights from AHP.
    """
    W = ws / ws.sum()  # ensure normalized
    return dijkstra_multicriteria(G, source, target, W, alpha)


def run_pipeline_gini_hurwicz(
    G: nx.DiGraph,
    source: int,
    target: int,
    wo: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    """
    Pipeline: Gini + α-Hurwicz + Dijkstra (no AHP).

    Uses only objective weights from Gini.
    """
    W = wo / wo.sum()  # ensure normalized
    return dijkstra_multicriteria(G, source, target, W, alpha)


if __name__ == "__main__":
    from graph_generator import generate_urban_graph
    from ahp import ahp_weights, get_default_comparison_matrix
    from gini_weights import gini_weights

    # Generate graph
    G = generate_urban_graph(20, connectivity=0.25, seed=42)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Step 1: AHP
    A = get_default_comparison_matrix()
    ws, _, _, _ = ahp_weights(A, verbose=True)

    # Step 2: Gini
    wo = gini_weights(G, verbose=True)

    # Step 3: Fusion
    W = fuse_weights(ws, wo, beta=0.5, verbose=True)

    # Steps 4-5: Dijkstra
    source, target = 0, G.number_of_nodes() - 1
    path, cost, details = dijkstra_multicriteria(G, source, target, W, alpha=0.5)
    print(f"Optimal path: {path}")
    print(f"Composite cost: {cost:.4f}")
    print(f"Path details: {details}")
