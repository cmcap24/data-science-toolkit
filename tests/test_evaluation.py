import pandas as pd

from src.causal_inference.evaluation.balance import compute_balance_stats


def test_compute_balance_stats() -> None:
    # Create dummy data
    df = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "age": [25, 30, 26, 29],
            "educ": [12, 16, 12, 16],
            "re74": [20000, 25000, 19000, 24000],
            "re75": [18000, 22000, 17000, 21000],
        }
    )

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
