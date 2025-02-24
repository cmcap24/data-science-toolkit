import pandas as pd

from src.causal_inference.evaluation.balance import (
    compute_balance_stats,
    compute_ks_test,
    compute_variance_ratios,
)

# Create Dummy data
df = pd.DataFrame(
    {
        "treat": [1, 1, 0, 0],
        "age": [25, 30, 26, 29],
        "educ": [12, 16, 12, 16],
        "re74": [20000, 25000, 19000, 24000],
        "re75": [18000, 22000, 17000, 21000],
    }
)


def test_compute_balance_stats() -> None:
    matched_pairs = [(0, 2), (1, 3)]  # Simulated matching

    balance_summary = compute_balance_stats(
        df, matched_pairs, ["age", "educ", "re74", "re75"]
    )

    # Ensure the DataFrame has expected shape
    assert balance_summary.shape == (
        2,
        4,
    ), """
    Balance summary should have two rows (before and after matching) and four columns
    """
    assert (
        not balance_summary.isnull().values.any()
    ), "Balance summary should not have NaN values"


def test_compute_variance_ratios() -> None:
    matched_pairs = [(0, 2), (1, 3)]

    variance_ratios = compute_variance_ratios(
        df, matched_pairs, ["age", "educ", "re74", "re75"]
    )

    assert variance_ratios.shape == (2, 4), "Incorrect shape for variance ratios"
    assert (
        not variance_ratios.isnull().values.any()
    ), "Variance ratios contain NaN values"


def test_compute_ks_test() -> None:
    matched_pairs = [(0, 2), (1, 3)]

    ks_test_results = compute_ks_test(
        df, matched_pairs, ["age", "educ", "re74", "re75"]
    )

    assert ks_test_results.shape == (4, 2), "Incorrect shape for KS test results"
