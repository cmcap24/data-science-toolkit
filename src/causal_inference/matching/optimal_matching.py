import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def optimal_match(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    distance_metric: str = "mahalanobis",
) -> list[tuple[int, int]]:
    """
    Perform optimal matching using the Hungarian algorithm.

    Parameters:
        treated (pd.DataFrame): DataFrame of treated units.
        control (pd.DataFrame): DataFrame of control units.
        covariates (List[str]): List of covariate column names used for matching.
        distance_metric (str): Distance metric to use (default: "mahalanobis").

    Returns:
        List[Tuple[int, int]]: Matched pairs as tuples (treated_index, control_index).
    """
    treated_cov = treated[covariates].to_numpy()
    control_cov = control[covariates].to_numpy()

    # Compute pairwise distances
    distance_matrix = cdist(treated_cov, control_cov, metric=distance_metric)

    # Solve the assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Return matched pairs (treated index, control index)
    matched_pairs = [
        (treated.index[i], control.index[j]) for i, j in zip(row_ind, col_ind)
    ]
    return matched_pairs
