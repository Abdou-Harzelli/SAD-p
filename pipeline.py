import heapq
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

from graph_generator import CRITERIA, NUM_CRITERIA


def fuse_weights(
    ws: np.ndarray,
    wo: np.ndarray,
    beta: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    assert len(ws) == len(wo), "Weight vectors must have same length."
    assert 0 <= beta <= 1, "Beta must be in [0, 1]."

    W = beta * ws + (1 - beta) * wo

    W = W / W.sum()

    if verbose:
        print("=" * 50)
        print(f"Weight Fusion (beta = {beta})")
        print("=" * 50)
        for i, c in enumerate(CRITERIA):
            print(f"  {c:15s}: Ws={ws[i]:.4f}  Wo={wo[i]:.4f}  ->  W={W[i]:.4f}")
        print(f"  Sum = {W.sum():.4f}")
        print()

    return W


def hurwicz_value(x_min: float, x_max: float, alpha: float) -> float:
    return alpha * x_min + (1 - alpha) * x_max


def compute_hurwicz_edge(
    edge_data: Dict,
    alpha: float,
) -> np.ndarray:
    V = np.zeros(NUM_CRITERIA)
    for k, criterion in enumerate(CRITERIA):
        x_min = edge_data[f"{criterion}_min"]
        x_max = edge_data[f"{criterion}_max"]
        V[k] = hurwicz_value(x_min, x_max, alpha)
    return V


def compute_edge_cost(
    edge_data: Dict,
    W: np.ndarray,
    alpha: float,
) -> float:
    V = compute_hurwicz_edge(edge_data, alpha)
    return float(np.dot(W, V))


def dijkstra_multicriteria(
    G: nx.DiGraph,
    source: int,
    target: int,
    W: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
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

            edge_data = G[u][v]
            cost = compute_edge_cost(edge_data, W, alpha)

            new_dist = dist[u] + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    if path[0] != source:
        return [], float("inf"), {}

    details = _compute_path_details(G, path, alpha)
    details["composite_cost"] = dist[target]

    return path, dist[target], details


def dijkstra_simple(
    G: nx.DiGraph,
    source: int,
    target: int,
) -> Tuple[List[int], float, Dict]:
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


def run_pipeline_full(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    wo: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    W = fuse_weights(ws, wo, beta)
    return dijkstra_multicriteria(G, source, target, W, alpha)


def run_pipeline_ahp_hurwicz(
    G: nx.DiGraph,
    source: int,
    target: int,
    ws: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    W = ws / ws.sum()
    return dijkstra_multicriteria(G, source, target, W, alpha)


def run_pipeline_gini_hurwicz(
    G: nx.DiGraph,
    source: int,
    target: int,
    wo: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[List[int], float, Dict]:
    W = wo / wo.sum()
    return dijkstra_multicriteria(G, source, target, W, alpha)


if __name__ == "__main__":
    from graph_generator import generate_urban_graph
    from ahp import ahp_weights, get_default_comparison_matrix
    from gini_weights import gini_weights

    G = generate_urban_graph(20, connectivity=0.25, seed=42)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    A = get_default_comparison_matrix()
    ws, _, _, _ = ahp_weights(A, verbose=True)

    wo = gini_weights(G, verbose=True)

    W = fuse_weights(ws, wo, beta=0.5, verbose=True)

    source, target = 0, G.number_of_nodes() - 1
    path, cost, details = dijkstra_multicriteria(G, source, target, W, alpha=0.5)
    print(f"Optimal path: {path}")
    print(f"Composite cost: {cost:.4f}")
    print(f"Path details: {details}")
