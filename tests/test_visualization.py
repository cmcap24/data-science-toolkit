import pandas as pd
import pytest

from src.causal_inference.visualization.matching import plot_covariate_balance


@pytest.mark.mpl_image_compare  # type: ignore
def test_plot_covariate_balance() -> None:
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

    matched_pairs = [(0, 2), (1, 3)]
    covariates = ["age", "educ", "re74", "re75"]

    # Run the function to check for exceptions
    try:
        plot_covariate_balance(df, matched_pairs, covariates)
    except Exception as e:
        pytest.fail(f"Plotting function raised an error: {e}")
