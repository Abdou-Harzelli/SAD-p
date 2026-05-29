# Multi-Criteria Dynamic Route Optimization

## Hybrid Approach Based on AHP and Dijkstra's Algorithm for Urban Road Navigation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Domain of Application](#3-domain-of-application)
4. [How to Run](#10-how-to-run)

---

## 1. Introduction

Finding the best route between two points is one of the most studied problems in computer science. The most well-known solution is Dijkstra's algorithm, which finds the shortest path based on a single criterion, usually distance.

However, in real-world navigation, the "best" route is not necessarily the shortest one. A driver also cares about travel time, road safety, and traffic congestion. A route that is short in distance may be dangerous, slow due to traffic, or pass through a high-accident zone.

This project improves Dijkstra's algorithm by introducing a 5-step pipeline that considers multiple criteria simultaneously and handles real-world uncertainty. The pipeline combines:

- The Analytic Hierarchy Process (AHP) for capturing expert judgment
- The Gini Index for data-driven objective weighting
- The Hurwicz criterion for handling uncertainty under incomplete information
- Dijkstra's algorithm for computing the optimal path

The result is a system that finds routes which are not just short, but also fast, safe, and congestion-free.

---

## 2. Problem Statement

Standard Dijkstra's algorithm optimizes a single criterion: distance. It ignores all other factors that affect route quality. This leads to paths that may be geometrically short but practically poor, being dangerous, congested, or slow.

The goal of this project is to build a multi-criteria route optimization system that:

- Considers multiple criteria simultaneously (distance, time, safety, congestion)
- Incorporates both expert opinion and data analysis for weighting
- Handles uncertainty in real-world conditions (traffic varies, accidents are random)
- Finds the mathematically optimal path considering all of the above

We compare four approaches to demonstrate how each component of the pipeline contributes to the final result:

| Number | Approach | Components Used |
|--------|----------|----------------|
| 1 | Plain Dijkstra | Distance only (baseline) |
| 2 | AHP + Hurwicz + Dijkstra | Expert weights + uncertainty handling |
| 3 | Gini + Hurwicz + Dijkstra | Data-driven weights + uncertainty handling |
| 4 | AHP + Gini + Hurwicz + Dijkstra | Full pipeline (all components) |

---

## 3. Domain of Application

The chosen domain is Urban Road Navigation. We model a city's road network as a directed graph G(V, E) where:

- V = set of intersection nodes (|V| = N)
- E = set of road segments connecting intersections (|E| = M)

Each road segment (edge) carries four criteria, and each criterion has an interval [x_min, x_max] representing uncertainty:

| Criterion | Description | Unit | Uncertainty Level |
|-----------|-------------|------|-------------------|
| Distance | Physical length of the road | Kilometers | Low (roads rarely change length) |
| Travel Time | Duration to traverse the road | Minutes | Moderate (varies with traffic flow) |
| Safety Risk | Probability of accidents | Index from 0 to 1 | High (random, unpredictable events) |
| Congestion | Level of traffic congestion | Index from 0 to 1 | High (changes with time of day) |

The intervals capture real-world uncertainty. For example, a road's travel time might be 5 minutes in the best case (no traffic) but 12 minutes in the worst case (rush hour). The system does not assume perfect knowledge; instead, it explicitly models this uncertainty.

---


## 4. How to Run

### Requirements

- Python 3.8 or higher
- NumPy
- NetworkX
- Matplotlib
- Tabulate

### Install Dependencies

On Kali Linux or other Debian-based systems:

```bash
sudo apt install python3-numpy python3-networkx python3-matplotlib python3-tabulate
```

On other systems:

```bash
pip install numpy networkx matplotlib tabulate
```

### Run the Jupyter Notebook Version

```bash
sudo apt install jupyter-notebook
cd multicriteria-routing
jupyter notebook multicriteria_routing.ipynb
```

Then click "Kernel" then "Restart and Run All" to execute all cells.

