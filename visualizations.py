import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List, Optional

from graph_generator import CRITERIA

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLORS = {
    "Dijkstra": "#636EFA",
    "AHP+Hurwicz+Dijkstra": "#EF553B",
    "Gini+Hurwicz+Dijkstra": "#00CC96",
    "AHP+Gini+Hurwicz+Dijkstra": "#AB63FA",
}

MARKERS = {
    "Dijkstra": "o",
    "AHP+Hurwicz+Dijkstra": "s",
    "Gini+Hurwicz+Dijkstra": "^",
    "AHP+Gini+Hurwicz+Dijkstra": "D",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_graph_with_paths(
    G: nx.DiGraph,
    paths: Dict[str, List[int]],
    source: int,
    target: int,
    title: str = "Urban Road Network -- Path Comparison",
    filename: str = "graph_paths.png",
):
    ensure_output_dir()
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    pos = nx.get_node_attributes(G, "pos")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, edge_color="#cccccc",
                           width=0.5, arrows=True, arrowsize=5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color="#cccccc", alpha=0.5)

    arc_radii = [-0.15, -0.05, 0.05, 0.15]
    line_styles = ["solid", "dashed", "dotted", "dashdot"]
    line_widths = [3.0, 2.8, 3.5, 2.5]

    path_tuples = {}
    for name, path in paths.items():
        key = tuple(path) if path else ()
        path_tuples.setdefault(key, []).append(name)

    for idx, (name, path) in enumerate(paths.items()):
        if not path or len(path) < 2:
            continue

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        color = COLORS.get(name, "#333333")
        rad = arc_radii[idx % len(arc_radii)]
        ls = line_styles[idx % len(line_styles)]
        lw = line_widths[idx % len(line_widths)]

        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=color, width=lw, alpha=0.85,
            arrows=True, arrowsize=14,
            connectionstyle=f"arc3,rad={rad}",
            style=ls,
        )

        nx.draw_networkx_nodes(
            G, pos, nodelist=path, ax=ax,
            node_size=80, node_color=color, alpha=0.65,
            edgecolors="white", linewidths=1.0,
        )

    nx.draw_networkx_nodes(
        G, pos, nodelist=[source], ax=ax,
        node_size=250, node_color="#2ECC71", edgecolors="black",
        linewidths=2, label="Source",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=[target], ax=ax,
        node_size=250, node_color="#E74C3C", edgecolors="black",
        linewidths=2, label="Target", node_shape="*",
    )

    nx.draw_networkx_labels(
        G, pos, labels={source: f"S={source}", target: f"T={target}"},
        ax=ax, font_size=10, font_weight="bold",
    )

    legend_elements = []
    for idx, name in enumerate(paths):
        if paths[name]:
            legend_elements.append(
                Line2D([0], [0], color=COLORS[name],
                       lw=line_widths[idx % len(line_widths)],
                       linestyle=line_styles[idx % len(line_styles)],
                       label=name)
            )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
               markersize=10, label='Source')
    )
    legend_elements.append(
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#E74C3C',
               markersize=12, label='Target')
    )

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9,
              fontsize=9, edgecolor="#cccccc")

    overlap_lines = []
    for route, names in path_tuples.items():
        if len(names) > 1 and route:
            route_str = "->".join(str(n) for n in route)
            overlap_lines.append(f"Same route ({route_str}):\n  " + "\n  ".join(names))
    if overlap_lines:
        note = "Overlapping Paths\n" + "\n".join(overlap_lines)
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD",
                          edgecolor="#FFC107", alpha=0.9))

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.15)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_scalability(
    exp_results: Dict,
    sizes: List[int],
    n_trials: int = 3,
    filename_prefix: str = "scalability",
):
    ensure_output_dir()
    approaches = list(exp_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for approach in approaches:
        avg_times = []
        std_times = []
        for n in sizes:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            times = [exp_results[approach]["times"][i] for i in indices]
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))

        ax.errorbar(
            sizes, avg_times, yerr=std_times,
            marker=MARKERS[approach], color=COLORS[approach],
            label=approach, linewidth=2, markersize=7, capsize=3,
        )

    ax.set_xlabel("Number of Nodes (N)", fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontweight="bold")
    ax.set_title("Execution Time vs Graph Size", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1]
    for approach in approaches:
        avg_costs = []
        for n in sizes:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            costs = [exp_results[approach]["costs"][i] for i in indices]
            avg_costs.append(np.mean(costs))

        ax.plot(
            sizes, avg_costs,
            marker=MARKERS[approach], color=COLORS[approach],
            label=approach, linewidth=2, markersize=7,
        )

    ax.set_xlabel("Number of Nodes (N)", fontweight="bold")
    ax.set_ylabel("Avg Composite Cost", fontweight="bold")
    ax.set_title("Path Cost vs Graph Size", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for approach in approaches:
        avg_lengths = []
        for n in sizes:
            indices = [i for i, s in enumerate(exp_results[approach]["sizes"]) if s == n]
            lengths = [exp_results[approach]["path_lengths"][i] for i in indices]
            avg_lengths.append(np.mean(lengths))

        ax.plot(
            sizes, avg_lengths,
            marker=MARKERS[approach], color=COLORS[approach],
            label=approach, linewidth=2, markersize=7,
        )

    ax.set_xlabel("Number of Nodes (N)", fontweight="bold")
    ax.set_ylabel("Avg Path Length (nodes)", fontweight="bold")
    ax.set_title("Path Length vs Graph Size", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Scalability Analysis: G(V, E), |V|=N", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_sensitivity_alpha(
    results: Dict,
    filename: str = "sensitivity_alpha.png",
):
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    alphas = results["alpha"]
    costs = results["cost"]

    ax = axes[0]
    ax.plot(alphas, costs, "o-", color="#AB63FA", linewidth=2.5, markersize=8)
    ax.fill_between(alphas, costs, alpha=0.15, color="#AB63FA")
    ax.set_xlabel("alpha (Optimism Coefficient)", fontweight="bold")
    ax.set_ylabel("Composite Path Cost", fontweight="bold")
    ax.set_title("Path Cost vs Hurwicz alpha", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.annotate("Pessimistic", xy=(0, costs[0]), fontsize=9, color="gray")
    ax.annotate("Optimistic", xy=(1, costs[-1]), fontsize=9, color="gray",
                ha="right")

    ax = axes[1]
    crit_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]
    for i, c in enumerate(CRITERIA):
        vals = results["per_criterion"][c]
        ax.plot(alphas, vals, "o-", color=crit_colors[i], linewidth=2,
                markersize=6, label=c.replace("_", " ").title())

    ax.set_xlabel("alpha (Optimism Coefficient)", fontweight="bold")
    ax.set_ylabel("Criterion Value (Path Total)", fontweight="bold")
    ax.set_title("Per-Criterion Sensitivity to alpha", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Sensitivity Analysis: alpha-Hurwicz Parameter", fontweight="bold",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_sensitivity_beta(
    results: Dict,
    filename: str = "sensitivity_beta.png",
):
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    betas = results["beta"]
    costs = results["cost"]

    ax = axes[0]
    ax.plot(betas, costs, "s-", color="#EF553B", linewidth=2.5, markersize=8)
    ax.fill_between(betas, costs, alpha=0.15, color="#EF553B")
    ax.set_xlabel("beta (Fusion Coefficient)", fontweight="bold")
    ax.set_ylabel("Composite Path Cost", fontweight="bold")
    ax.set_title("Path Cost vs Fusion beta", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.annotate("Fully Objective\n(Gini only)", xy=(0, costs[0]), fontsize=8, color="gray")
    ax.annotate("Fully Subjective\n(AHP only)", xy=(1, costs[-1]), fontsize=8,
                color="gray", ha="right")

    ax = axes[1]
    crit_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]
    for i, c in enumerate(CRITERIA):
        vals = results["per_criterion"][c]
        ax.plot(betas, vals, "s-", color=crit_colors[i], linewidth=2,
                markersize=6, label=c.replace("_", " ").title())

    ax.set_xlabel("beta (Fusion Coefficient)", fontweight="bold")
    ax.set_ylabel("Criterion Value (Path Total)", fontweight="bold")
    ax.set_title("Per-Criterion Sensitivity to beta", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Sensitivity Analysis: beta Weight Fusion Parameter", fontweight="bold",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_criterion_comparison(
    results: Dict,
    filename: str = "criterion_comparison.png",
):
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 6))

    approaches = list(results.keys())
    n_approaches = len(approaches)
    n_criteria = len(CRITERIA)

    x = np.arange(n_criteria)
    width = 0.18

    for idx, approach in enumerate(approaches):
        if "criterion_midpoints" not in results[approach].get("details", {}):
            continue
        vals = [results[approach]["details"]["criterion_midpoints"][c] for c in CRITERIA]
        offset = (idx - n_approaches / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=approach, color=COLORS.get(approach, "#999"),
                      edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Criteria", fontweight="bold")
    ax.set_ylabel("Path Total Value", fontweight="bold")
    ax.set_title("Per-Criterion Performance Comparison", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in CRITERIA])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_weight_comparison(
    ws: np.ndarray,
    wo: np.ndarray,
    W: np.ndarray,
    beta: float = 0.5,
    filename: str = "weight_comparison.png",
):
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(CRITERIA))
    width = 0.25

    bars1 = ax.bar(x - width, ws, width, label="AHP (Subjective)",
                   color="#636EFA", edgecolor="white")
    bars2 = ax.bar(x, wo, width, label="Gini (Objective)",
                   color="#00CC96", edgecolor="white")
    bars3 = ax.bar(x + width, W, width, label=f"Fused (beta={beta})",
                   color="#AB63FA", edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Criteria", fontweight="bold")
    ax.set_ylabel("Weight", fontweight="bold")
    ax.set_title("Weight Comparison: Subjective vs Objective vs Fused", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in CRITERIA])
    ax.legend()
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(0, max(ws.max(), wo.max(), W.max()) * 1.25)

    fig.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def plot_ahp_matrix(
    comparison_matrix: np.ndarray,
    weights: np.ndarray,
    cr: float,
    filename: str = "ahp_analysis.png",
):
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    labels = [c.replace("_", "\n").title() for c in CRITERIA]
    im = ax.imshow(comparison_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(CRITERIA)))
    ax.set_yticks(range(len(CRITERIA)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(CRITERIA)):
        for j in range(len(CRITERIA)):
            val = comparison_matrix[i][j]
            text = f"{val:.2f}" if val >= 1 else f"1/{1/val:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if val > 3 else "black")

    ax.set_title(f"AHP Pairwise Comparison Matrix\n(CR = {cr:.4f})",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]
    bars = ax.barh(range(len(CRITERIA)), weights, color=colors, edgecolor="white")

    ax.set_yticks(range(len(CRITERIA)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in CRITERIA])
    ax.set_xlabel("Weight", fontweight="bold")
    ax.set_title("Derived Priority Weights", fontweight="bold")
    ax.grid(True, alpha=0.15, axis="x")

    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
                f"{w:.4f}", va="center", fontsize=10)

    fig.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath
