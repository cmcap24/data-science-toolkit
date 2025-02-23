import pandas as pd

from src.causal_inference.matching.greedy_matching import greedy_match


def test_greedy_match() -> None:
    # Create dummy data
    treated_data = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [25, 30],
            "educ": [12, 16],
            "re74": [20000, 25000],
            "re75": [18000, 22000],
        }
    )
    control_data = pd.DataFrame(
        {
            "id": [3, 4, 5],
            "age": [26, 29, 35],
            "educ": [12, 16, 14],
            "re74": [19000, 24000, 26000],
            "re75": [17000, 21000, 23000],
        }
    )

    treated_data.set_index("id", inplace=True)
    control_data.set_index("id", inplace=True)

    covariates = ["age", "educ", "re74", "re75"]

    matched_pairs = greedy_match(treated_data, control_data, covariates)

    # Check if correct number of pairs are matched
    assert len(matched_pairs) == len(
        treated_data
    ), "Mismatch in number of matched pairs"
    assert all(
        isinstance(pair, tuple) for pair in matched_pairs
    ), "Matched pairs should be tuples"
