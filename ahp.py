"""
Step 1: Analytic Hierarchy Process (AHP) — Subjective Weighting.

Implements pairwise comparison matrices and eigenvector-based weight derivation
for the four urban navigation criteria:
  1. Distance
  2. Travel Time
  3. Safety Risk
  4. Congestion

Also includes consistency checking (CI, CR).
"""

import numpy as np
from typing import Tuple, Optional


# Saaty's Random Index table for n = 1..10
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def ahp_weights(
    comparison_matrix: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Compute AHP weights from a pairwise comparison matrix.

    Uses the principal eigenvector method (Saaty, 1980).

    Parameters
    ----------
    comparison_matrix : np.ndarray
        Square reciprocal matrix of pairwise comparisons.
        Element a[i][j] represents the importance of criterion i over j.
        Must satisfy a[i][j] = 1 / a[j][i].
    verbose : bool
        Print intermediate results.

    Returns
    -------
    weights : np.ndarray
        Normalized subjective weight vector Ws = [ws1, ws2, ..., wsn].
    ci : float
        Consistency Index.
    cr : float
        Consistency Ratio (< 0.10 is acceptable).
    is_consistent : bool
        True if CR < 0.10.
    """
    n = comparison_matrix.shape[0]
    assert comparison_matrix.shape == (n, n), "Matrix must be square."

    # ── Eigenvector Method ────────────────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)

    # Find the principal (largest real) eigenvalue
    real_parts = eigenvalues.real
    max_idx = np.argmax(real_parts)
    lambda_max = real_parts[max_idx]

    # Principal eigenvector (normalize to sum = 1)
    principal_vector = np.abs(eigenvectors[:, max_idx].real)
    weights = principal_vector / principal_vector.sum()

    # ── Consistency Check ─────────────────────────────────────────────────
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    is_consistent = cr < 0.10

    if verbose:
        print("=" * 50)
        print("AHP Analysis")
        print("=" * 50)
        print(f"  Comparison Matrix:\n{comparison_matrix}")
        print(f"  λ_max = {lambda_max:.4f}")
        print(f"  Weights (Ws) = {weights}")
        print(f"  CI = {ci:.4f}")
        print(f"  CR = {cr:.4f} {'✓ Consistent' if is_consistent else '✗ INCONSISTENT'}")
        print()

    return weights, ci, cr, is_consistent


def get_default_comparison_matrix() -> np.ndarray:
    """
    Default AHP comparison matrix for urban navigation.

    Priority ranking (for a safety-conscious driver):
      Safety Risk > Travel Time > Distance > Congestion

    Pairwise comparison scale (Saaty 1-9):
      1 = Equal importance
      3 = Moderate importance
      5 = Strong importance
      7 = Very strong importance
      9 = Extreme importance

    Criteria order: [distance, travel_time, safety_risk, congestion]
    """
    # Safety > Time > Distance > Congestion
    A = np.array([
        #  Dist  Time  Safety  Cong
        [1,     1/3,  1/5,    3  ],   # Distance
        [3,     1,    1/3,    5  ],   # Travel Time
        [5,     3,    1,      7  ],   # Safety Risk
        [1/3,   1/5,  1/7,   1   ],   # Congestion
    ], dtype=float)

    return A


def get_alternative_comparison_matrices() -> dict:
    """
    Return several AHP comparison matrices representing different driver profiles.
    """
    profiles = {}

    # ── Speed-focused driver ──────────────────────────────────────────────
    profiles["speed_focused"] = np.array([
        [1,     1/3,  3,     3  ],
        [3,     1,    5,     5  ],
        [1/3,   1/5,  1,     1  ],
        [1/3,   1/5,  1,     1  ],
    ], dtype=float)

    # ── Safety-focused driver ─────────────────────────────────────────────
    profiles["safety_focused"] = np.array([
        [1,     1/3,  1/7,   1  ],
        [3,     1,    1/5,   3  ],
        [7,     5,    1,     9  ],
        [1,     1/3,  1/9,   1  ],
    ], dtype=float)

    # ── Balanced driver ───────────────────────────────────────────────────
    profiles["balanced"] = np.array([
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
    ], dtype=float)

    return profiles


if __name__ == "__main__":
    print("─── Default Profile (Safety-Conscious) ───")
    A = get_default_comparison_matrix()
    weights, ci, cr, ok = ahp_weights(A, verbose=True)

    print("\n─── Alternative Profiles ───")
    for name, matrix in get_alternative_comparison_matrices().items():
        print(f"\n  Profile: {name}")
        w, ci, cr, ok = ahp_weights(matrix, verbose=False)
        print(f"    Ws = {np.round(w, 4)}")
        print(f"    CR = {cr:.4f} {'✓' if ok else '✗'}")
