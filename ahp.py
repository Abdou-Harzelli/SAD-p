import numpy as np
from typing import Tuple, Optional


RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def ahp_weights(
    comparison_matrix: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float, bool]:
    n = comparison_matrix.shape[0]
    assert comparison_matrix.shape == (n, n), "Matrix must be square."

    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)

    real_parts = eigenvalues.real
    max_idx = np.argmax(real_parts)
    lambda_max = real_parts[max_idx]

    principal_vector = np.abs(eigenvectors[:, max_idx].real)
    weights = principal_vector / principal_vector.sum()

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    is_consistent = cr < 0.10

    if verbose:
        print("=" * 50)
        print("AHP Analysis")
        print("=" * 50)
        print(f"  Comparison Matrix:\n{comparison_matrix}")
        print(f"  lambda_max = {lambda_max:.4f}")
        print(f"  Weights (Ws) = {weights}")
        print(f"  CI = {ci:.4f}")
        print(f"  CR = {cr:.4f} {'Consistent' if is_consistent else 'INCONSISTENT'}")
        print()

    return weights, ci, cr, is_consistent


def get_default_comparison_matrix() -> np.ndarray:
    A = np.array([
        [1,     1/3,  1/5,    3  ],
        [3,     1,    1/3,    5  ],
        [5,     3,    1,      7  ],
        [1/3,   1/5,  1/7,   1   ],
    ], dtype=float)

    return A


def get_alternative_comparison_matrices() -> dict:
    profiles = {}

    profiles["speed_focused"] = np.array([
        [1,     1/3,  3,     3  ],
        [3,     1,    5,     5  ],
        [1/3,   1/5,  1,     1  ],
        [1/3,   1/5,  1,     1  ],
    ], dtype=float)

    profiles["safety_focused"] = np.array([
        [1,     1/3,  1/7,   1  ],
        [3,     1,    1/5,   3  ],
        [7,     5,    1,     9  ],
        [1,     1/3,  1/9,   1  ],
    ], dtype=float)

    profiles["balanced"] = np.array([
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
        [1,     1,    1,     1  ],
    ], dtype=float)

    return profiles


if __name__ == "__main__":
    print("--- Default Profile (Safety-Conscious) ---")
    A = get_default_comparison_matrix()
    weights, ci, cr, ok = ahp_weights(A, verbose=True)

    print("\n--- Alternative Profiles ---")
    for name, matrix in get_alternative_comparison_matrices().items():
        print(f"\n  Profile: {name}")
        w, ci, cr, ok = ahp_weights(matrix, verbose=False)
        print(f"    Ws = {np.round(w, 4)}")
        print(f"    CR = {cr:.4f} {'OK' if ok else 'FAIL'}")
