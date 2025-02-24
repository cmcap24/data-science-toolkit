import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def get_matched_distributions(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts treated and averaged control distributions for visualization.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Matched treated and averaged control DataFrames for plotting.
    """
    treated_idx, control_idx_lists = zip(*matched_pairs)
    treated_matched = data.loc[list(treated_idx), covariates]

    control_averaged = pd.DataFrame(
        {
            cov: [data.loc[ctrl_idxs, cov].mean() for ctrl_idxs in control_idx_lists]
            for cov in covariates
        }
    )
    control_averaged.index = treated_matched.index

    return treated_matched, control_averaged


def plot_covariate_balance(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> None:
    """
    Plot distributions of covariates before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.
    """
    treated_matched, control_averaged = get_matched_distributions(
        data, matched_pairs, covariates
    )

    fig, axes = plt.subplots(len(covariates), 2, figsize=(12, 3 * len(covariates)))

    for i, cov in enumerate(covariates):
        # Before matching
        sns.kdeplot(
            data[data["treat"] == 1][cov], label="Treated", ax=axes[i, 0], fill=True
        )
        sns.kdeplot(
            data[data["treat"] == 0][cov], label="Control", ax=axes[i, 0], fill=True
        )
        axes[i, 0].set_title(f"{cov} - Before Matching")

        # After matching
        sns.kdeplot(
            treated_matched[cov], label="Treated (Matched)", ax=axes[i, 1], fill=True
        )
        sns.kdeplot(
            control_averaged[cov], label="Control (Matched)", ax=axes[i, 1], fill=True
        )
        axes[i, 1].set_title(f"{cov} - After Matching")

    plt.tight_layout()
    plt.show()


def plot_qq_balance(
    data: pd.DataFrame,
    matched_pairs: list[tuple[int, list[int]]],
    covariates: list[str],
) -> None:
    """
    Generate QQ plots to compare covariate distributions before and after matching.

    Parameters:
        data (pd.DataFrame): Original dataset.
        matched_pairs (list[tuple[int, list[int]]]):
            Matched pairs (each treated index matched to a list of control indices).
        covariates (list[str]): Covariates used for matching.
    """
    treated_matched, control_averaged = get_matched_distributions(
        data, matched_pairs, covariates
    )

    fig, axes = plt.subplots(len(covariates), 2, figsize=(12, 3 * len(covariates)))

    for i, cov in enumerate(covariates):
        # Before matching
        stats.probplot(data[data["treat"] == 1][cov], dist="norm", plot=axes[i, 0])
        stats.probplot(data[data["treat"] == 0][cov], dist="norm", plot=axes[i, 0])
        axes[i, 0].set_title(f"QQ Plot: {cov} - Before Matching")

        # After matching
        stats.probplot(treated_matched[cov], dist="norm", plot=axes[i, 1])
        stats.probplot(control_averaged[cov], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f"QQ Plot: {cov} - After Matching")

    plt.tight_layout()
    plt.show()
