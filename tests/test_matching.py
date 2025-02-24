import pandas as pd

from src.causal_inference.matching.greedy_matching import greedy_match
from src.causal_inference.matching.optimal_matching import optimal_match

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
        "id": [3, 4, 5, 6],
        "age": [26, 29, 35, 31],
        "educ": [12, 16, 14, 15],
        "re74": [19000, 24000, 26000, 23000],
        "re75": [17000, 21000, 23000, 20000],
    }
)
treated.set_index("id", inplace=True)
control.set_index("id", inplace=True)
covariates = ["age", "educ", "re74", "re75"]


def test_greedy_match() -> None:
    # Many-to-one (matching 2 controls per treated)
    matched_pairs = greedy_match(treated, control, covariates, k=2)

    assert len(matched_pairs) == len(treated)
    assert all(len(pair[1]) == 2 for pair in matched_pairs)


def test_optimal_match() -> None:
    matched_pairs = optimal_match(treated, control, covariates, k=2)

    assert len(matched_pairs) == len(treated), "Mismatch in number of treated units"
    assert all(
        len(pair[1]) == 2 for pair in matched_pairs
    ), "Each treated should be matched to 2 controls"
