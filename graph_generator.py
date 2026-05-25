"""
Graph Generator for Urban Road Network Simulation.

Domain: Urban Road Navigation
Criteria:
  1. Distance (km)
  2. Travel Time (minutes)
  3. Safety Risk (accident rate index 0-1)
  4. Congestion Level (0-1)

Each edge has interval-valued attributes [x_min, x_max] to model uncertainty.
"""

import random
import math
import networkx as nx
import numpy as np
from typing import Tuple, Dict, List


# ─── Criteria Metadata ───────────────────────────────────────────────────────
CRITERIA = ["distance", "travel_time", "safety_risk", "congestion"]
NUM_CRITERIA = len(CRITERIA)


def generate_urban_graph(
    n_nodes: int,
    connectivity: float = 0.3,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Generate a random directed urban road graph with multi-criteria edge attributes.

    Each edge carries interval-valued data [x_min, x_max] for every criterion,
    representing best-case and worst-case scenarios under uncertainty.

    Parameters
    ----------
    n_nodes : int
        Number of intersection nodes |V| = N.
    connectivity : float
        Probability of an edge between any two nodes (Erdős–Rényi model).
        Adjusted to ensure the graph is connected.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        Directed graph with edge attributes for each criterion.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # Place nodes on a 2D grid to get realistic distances
    positions = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(n_nodes)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    nx.set_node_attributes(G, positions, "pos")

    # ── Create edges using Erdős–Rényi model ──────────────────────────────
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and rng.random() < connectivity:
                dist = _euclidean(positions[i], positions[j])
                attrs = _generate_edge_criteria(dist, rng)
                G.add_edge(i, j, **attrs)

    # ── Ensure strong connectivity ────────────────────────────────────────
    _ensure_connectivity(G, positions, rng)

    return G


def generate_scaled_graph(
    n_nodes: int,
    avg_degree: int = 6,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Generate a graph with controlled average degree for scalability experiments.

    For large N, using Erdős–Rényi with fixed connectivity creates O(N²) edges.
    This generator keeps |E| = O(N · avg_degree) for fair comparison.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    avg_degree : int
        Target average out-degree per node.
    seed : int
        Random seed.

    Returns
    -------
    nx.DiGraph
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    positions = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(n_nodes)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    nx.set_node_attributes(G, positions, "pos")

    # For each node, connect to `avg_degree` nearest neighbours
    all_nodes = list(range(n_nodes))
    for i in all_nodes:
        distances = []
        for j in all_nodes:
            if i != j:
                d = _euclidean(positions[i], positions[j])
                distances.append((j, d))
        distances.sort(key=lambda x: x[1])

        # Connect to nearest neighbours + some random ones for variety
        n_near = max(1, avg_degree // 2)
        n_rand = max(1, avg_degree - n_near)

        neighbours = [x[0] for x in distances[:n_near]]
        remaining = [x[0] for x in distances[n_near:]]
        if remaining:
            n_rand = min(n_rand, len(remaining))
            neighbours += list(rng.choice(remaining, size=n_rand, replace=False))

        for j in neighbours:
            if not G.has_edge(i, j):
                dist = _euclidean(positions[i], positions[j])
                attrs = _generate_edge_criteria(dist, rng)
                G.add_edge(i, j, **attrs)

    _ensure_connectivity(G, positions, rng)
    return G


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _generate_edge_criteria(
    euclidean_dist: float, rng: np.random.RandomState
) -> Dict:
    """
    Generate interval-valued [x_min, x_max] criteria for one edge.

    The criteria model real-world uncertainty:
    - Distance:     nearly deterministic (small interval)
    - Travel Time:  moderate uncertainty (traffic variation)
    - Safety Risk:  high uncertainty (random events)
    - Congestion:   high uncertainty (time-dependent)
    """
    # ── 1. Distance (km) ─ nearly deterministic ──────────────────────────
    base_dist = euclidean_dist / 10.0  # scale to 0–14 km range
    dist_min = base_dist * rng.uniform(0.95, 1.0)
    dist_max = base_dist * rng.uniform(1.0, 1.05)

    # ── 2. Travel Time (minutes) ─ moderate uncertainty ──────────────────
    # base speed: 30–60 km/h  →  time = dist / speed * 60
    speed = rng.uniform(30, 60)
    base_time = (base_dist / speed) * 60
    time_min = base_time * rng.uniform(0.8, 1.0)
    time_max = base_time * rng.uniform(1.0, 1.5)

    # ── 3. Safety Risk (0–1) ─ high uncertainty ──────────────────────────
    base_risk = rng.beta(2, 5)  # skewed towards lower risk
    risk_min = max(0.0, base_risk * rng.uniform(0.5, 0.9))
    risk_max = min(1.0, base_risk * rng.uniform(1.1, 2.0))

    # ── 4. Congestion (0–1) ─ high uncertainty ───────────────────────────
    base_cong = rng.beta(2, 3)
    cong_min = max(0.0, base_cong * rng.uniform(0.4, 0.8))
    cong_max = min(1.0, base_cong * rng.uniform(1.2, 2.5))

    return {
        "distance_min": dist_min,
        "distance_max": dist_max,
        "travel_time_min": time_min,
        "travel_time_max": time_max,
        "safety_risk_min": risk_min,
        "safety_risk_max": risk_max,
        "congestion_min": cong_min,
        "congestion_max": cong_max,
        # Also store a single deterministic distance for plain Dijkstra
        "weight": (dist_min + dist_max) / 2.0,
    }


def _ensure_connectivity(G: nx.DiGraph, positions: Dict, rng: np.random.RandomState):
    """Ensure the graph is strongly connected by adding bridge edges."""
    if nx.is_strongly_connected(G):
        return

    # Get strongly connected components and bridge them
    components = list(nx.strongly_connected_components(G))
    for idx in range(len(components) - 1):
        c1 = list(components[idx])
        c2 = list(components[idx + 1])
        # Pick closest pair of nodes between the two components
        best_pair = None
        best_dist = float("inf")
        for u in c1[:10]:  # limit search for performance
            for v in c2[:10]:
                d = _euclidean(positions[u], positions[v])
                if d < best_dist:
                    best_dist = d
                    best_pair = (u, v)

        if best_pair:
            u, v = best_pair
            dist = _euclidean(positions[u], positions[v])
            # Add edges in both directions
            for a, b in [(u, v), (v, u)]:
                if not G.has_edge(a, b):
                    attrs = _generate_edge_criteria(dist, rng)
                    G.add_edge(a, b, **attrs)


def get_graph_info(G: nx.DiGraph) -> Dict:
    """Return summary statistics of the graph."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "strongly_connected": nx.is_strongly_connected(G),
        "avg_out_degree": sum(d for _, d in G.out_degree()) / G.number_of_nodes(),
    }


if __name__ == "__main__":
    G = generate_urban_graph(20, connectivity=0.25, seed=42)
    info = get_graph_info(G)
    print("=== Urban Road Graph ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\nSample edge (0→1): {dict(G[0][1]) if G.has_edge(0, 1) else 'no edge'}")
