import pandas as pd
from scipy.stats import ks_2samp


def get_matched_groups(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts the treated and averaged control groups from matched pairs.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Matched treated and averaged control DataFrames.
    """
    treated_idx, control_idx_lists = zip(*matched_pairs)

    # Matched treated dataset
    treated_matched = data.loc[list(treated_idx), covariates]

    # Compute the mean covariates for each treated unit's matched controls
    control_averaged = pd.DataFrame(
        {
            cov: [data.loc[ctrl_idxs, cov].mean() for ctrl_idxs in control_idx_lists]
            for cov in covariates
        }
    )
    control_averaged.index = treated_matched.index  # Align indices

    return treated_matched, control_averaged


def compute_balance_stats(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> pd.DataFrame:
    """
    Compute summary statistics and standardized mean differences
    (SMD) before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: Summary of before and after treatment statistics including means,
            standard deviations, and SMD.
    """
    treated_matched, control_averaged = get_matched_groups(
        data, matched_pairs, covariates
    )

    # Compute statistics before matching
    stats_before = data.groupby("treat")[covariates].agg(["mean", "std"])

    # Extract proper indexing from multi-index columns
    means_before = stats_before.xs("mean", axis=1, level=1)
    # stds_before = stats_before.xs("std", axis=1, level=1)

    # Compute statistics after matching
    means_after = pd.DataFrame(
        {
            "Treated Mean (Matched)": treated_matched.mean(),
            "Control Mean (Matched)": control_averaged.mean(),
        },
        index=covariates,
    )

    stds_after = pd.DataFrame(
        {
            "Treated Std (Matched)": treated_matched.std(),
            "Control Std (Matched)": control_averaged.std(),
        },
        index=covariates,
    )

    # Compute SMD
    smd_before = (means_before.loc[1] - means_before.loc[0]) / data[covariates].std()
    smd_after = (treated_matched.mean() - control_averaged.mean()) / data[
        covariates
    ].std()

    smd_before_data = pd.DataFrame(smd_before, columns=["SMD Before"])
    smd_after_data = pd.DataFrame(smd_after, columns=["SMD After"])

    # Combine into one summary DataFrame (ensuring aligned index)
    balance_summary = pd.concat(
        [
            # means_before.rename(columns=lambda x: f"{x} (Before)"),
            # stds_before.rename(columns=lambda x: f"{x} Std (Before)"),
            means_after,
            stds_after,
            smd_before_data,
            smd_after_data,
        ],
        axis=1,
    )

    return balance_summary


def compute_variance_ratios(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> pd.DataFrame:
    """
    Compute variance ratios before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: Variance ratios before and after matching.
    """
    treated_matched, control_averaged = get_matched_groups(
        data, matched_pairs, covariates
    )

    # Compute variance before matching
    variance_before = data.groupby("treat")[covariates].var()

    # Compute variance after matching
    variance_after = pd.DataFrame(
        {
            "Treated (Matched)": treated_matched.var(),
            "Control (Matched)": control_averaged.var(),
        }
    )

    # Compute variance ratios
    variance_ratios = pd.DataFrame(
        {
            "Before Matching": variance_before.loc[1] / variance_before.loc[0],
            "After Matching": variance_after["Treated (Matched)"]
            / variance_after["Control (Matched)"],
        }
    )

    return variance_ratios


def compute_ks_test(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> pd.DataFrame:
    """
    Compute Kolmogorov-Smirnov (KS) test statistics before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.

    Returns:
        pd.DataFrame: KS statistics before and after matching.
    """
    treated_matched, control_averaged = get_matched_groups(
        data, matched_pairs, covariates
    )

    # Flatten all control matches
    all_control_indices = [
        idx for sublist in [pair[1] for pair in matched_pairs] for idx in sublist
    ]
    control_matched_flattened = data.loc[all_control_indices, covariates]

    ks_stats = {}
    for cov in covariates:
        ks_before = ks_2samp(
            data[data["treat"] == 1][cov], data[data["treat"] == 0][cov]
        ).statistic
        ks_after = ks_2samp(
            treated_matched[cov], control_matched_flattened[cov]
        ).statistic
        ks_stats[cov] = {"Before Matching": ks_before, "After Matching": ks_after}

    return pd.DataFrame(ks_stats).T
