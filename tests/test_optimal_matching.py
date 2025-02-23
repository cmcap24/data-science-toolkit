import pandas as pd

from src.causal_inference.matching.optimal_matching import optimal_match


def test_optimal_match() -> None:
    # Dummy dataset
    treated = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [25, 30],
            "educ": [12, 16],
            "re74": [20000, 25000],
            "re75": [18000, 22000],
        }
    )
    control = pd.DataFrame(
        {
            "id": [3, 4, 5],
            "age": [26, 29, 35],
            "educ": [12, 16, 14],
            "re74": [19000, 24000, 26000],
            "re75": [17000, 21000, 23000],
        }
    )

    treated.set_index("id", inplace=True)
    control.set_index("id", inplace=True)

    covariates = ["age", "educ", "re74", "re75"]

    matched_pairs = optimal_match(treated, control, covariates)

    # Check if correct number of pairs are matched
    assert len(matched_pairs) == len(
        treated
    ), "Optimal matching did not match all treated units"
    assert all(
        isinstance(pair, tuple) for pair in matched_pairs
    ), "Matched pairs should be tuples"
