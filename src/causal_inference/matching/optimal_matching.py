import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def optimal_match(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    distance_metric: str = "euclidean",
    k: int = 1,
) -> list[tuple[int, list[int]]]:
    """
    Perform optimal matching using the Hungarian algorithm.

    Parameters:
        treated (pd.DataFrame): DataFrame of treated units.
        control (pd.DataFrame): DataFrame of control units.
        covariates (List[str]): List of covariate column names used for matching.
        distance_metric (str): Distance metric to use (default: "euclidean").
        k (int): Number of control units to match to each treated unit (default: 1).

    Returns:
        List[Tuple[int, int]]: Matched pairs as tuples (treated_index, control_index).
    """
    treated_cov = treated[covariates].to_numpy()
    control_cov = control[covariates].to_numpy()

    # Compute pairwise distances
    distance_matrix = cdist(treated_cov, control_cov, metric=distance_metric)

    # Solve the assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create a mapping of treated units to their closest control matches
    treated_control_map: dict[int, list[int]] = {i: [] for i in treated.index}

    # Sort distances for each treated unit
    for i in range(len(treated)):
        sorted_controls = np.argsort(distance_matrix[i])[
            :k
        ]  # Get top-k closest matches
        treated_control_map[treated.index[i]] = [
            control.index[j] for j in sorted_controls
        ]

    # Convert to list of tuples
    matched_pairs = [(key, treated_control_map[key]) for key in treated_control_map]
    return matched_pairs
