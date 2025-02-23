import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def greedy_match(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    distance_metric: str = "mahalanobis",
) -> list[tuple[int, int]]:
    """
    Perform greedy matching between treated and control groups.

    Parameters:
        treated (pd.DataFrame): DataFrame of treated units.
        control (pd.DataFrame): DataFrame of control units.
        covariates (list[str]): List of column names used for matching.
        distance_metric (str): Distance metric to use (default: 'euclidean').

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
        if not available_idx:
            break  # No more controls available

        best_match_idx = available_idx[np.argmin(row[available_idx])]
        matched_pairs.append((treated.index[i], control.index[best_match_idx]))
        used_controls.add(best_match_idx)

    return matched_pairs
