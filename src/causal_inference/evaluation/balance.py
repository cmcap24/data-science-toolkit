import pandas as pd


def compute_balance_stats(
    data: pd.DataFrame, matched_pairs: list[tuple[int, int]], covariates: list[str]
) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMD) before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (List[Tuple[int, int]]):
            List of (treated_idx, control_idx) tuples.
        covariates (List[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: Summary of balance statistics (SMD before and after matching).
    """
    treated_idx, control_idx = zip(*matched_pairs)
    treated_matched = data.loc[list(treated_idx), covariates]
    control_matched = data.loc[list(control_idx), covariates]

    before_matching = data.groupby("treat")[covariates].mean()

    # Compute Standardized Mean Difference (SMD)
    smd_before = (before_matching.loc[1] - before_matching.loc[0]) / data[
        covariates
    ].std()
    smd_after = (treated_matched.mean() - control_matched.mean()) / data[
        covariates
    ].std()

    balance_summary = pd.DataFrame(
        {"Before Matching": smd_before, "After Matching": smd_after}
    ).T

    return balance_summary
