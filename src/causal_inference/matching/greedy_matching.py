import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def greedy_match(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    distance_metric: str = "euclidean",
    k: int = 1,
) -> list[tuple[int, list[int]]]:
    """
    Perform greedy matching between treated and control groups.

    Parameters:
        treated (pd.DataFrame): DataFrame of treated units.
        control (pd.DataFrame): DataFrame of control units.
        covariates (list[str]): List of column names used for matching.
        distance_metric (str): Distance metric to use (default: 'euclidean').
        k (int): Number of control units to match to each treated unit (default: 1).
    Returns:
        list[tuple[int, int]]: Matched pairs as tuples (treated_index, control_index)
    """
    treated_cov = treated[covariates].to_numpy()
    control_cov = control[covariates].to_numpy()
    distances = cdist(treated_cov, control_cov, metric=distance_metric)

    matched_pairs = []
    used_controls = set()

    for i, row in enumerate(distances):
        available_idx = [j for j in range(len(control_cov)) if j not in used_controls]
        if len(available_idx) < k:
            break  # Not enough control units left

        best_matches = np.argsort(row[available_idx])[:k]  # Get top-k closest matches
        matched_controls = [control.index[available_idx[j]] for j in best_matches]

        matched_pairs.append((treated.index[i], matched_controls))
        used_controls.update(matched_controls)

    return matched_pairs
