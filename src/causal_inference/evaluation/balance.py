import pandas as pd
from scipy.stats import ks_2samp


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
    )
    return balance_summary.T


def compute_variance_ratios(
    data: pd.DataFrame, matched_pairs: list[tuple[int, int]], covariates: list[str]
) -> pd.DataFrame:
    """
    Compute variance ratios before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (List[Tuple[int, int]]):
            List of (treated_idx, control_idx) tuples.
        covariates (List[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: Variance ratios before and after matching.
    """
    treated_idx, control_idx = zip(*matched_pairs)
    treated_matched = data.loc[list(treated_idx), covariates]
    control_matched = data.loc[list(control_idx), covariates]

    before_variance = data.groupby("treat")[covariates].var()
    after_variance = pd.DataFrame(
        {
            "Treated (Matched)": treated_matched.var(),
            "Control (Matched)": control_matched.var(),
        }
    )

    variance_ratios = pd.DataFrame(
        {
            "Before Matching": before_variance.loc[1] / before_variance.loc[0],
            "After Matching": after_variance["Treated (Matched)"]
            / after_variance["Control (Matched)"],
        }
    ).T

    return variance_ratios


def compute_ks_test(
    data: pd.DataFrame, matched_pairs: list[tuple[int, int]], covariates: list[str]
) -> pd.DataFrame:
    """
    Compute Kolmogorov-Smirnov (KS) test statistics before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (List[Tuple[int, int]]):
            List of (treated_idx, control_idx) tuples.
        covariates (List[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: KS statistics before and after matching.
    """
    treated_idx, control_idx = zip(*matched_pairs)
    treated_matched = data.loc[list(treated_idx), covariates]
    control_matched = data.loc[list(control_idx), covariates]

    ks_stats = {}
    for cov in covariates:
        ks_before = ks_2samp(
            data[data["treat"] == 1][cov], data[data["treat"] == 0][cov]
        ).statistic
        ks_after = ks_2samp(treated_matched[cov], control_matched[cov]).statistic
        ks_stats[cov] = {"Before Matching": ks_before, "After Matching": ks_after}

    return pd.DataFrame(ks_stats).T
