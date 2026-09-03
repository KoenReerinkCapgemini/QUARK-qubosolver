"""Tests for the QUARK Pasqal adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from quark.core import Data, Failed
from quark.interface_types import Qubo, SampleDistribution

from quark_plugin_pasqal import QuboToPasqal


def make_solution() -> SimpleNamespace:
    return SimpleNamespace(
        bitstrings=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        counts=torch.tensor([3, 2], dtype=torch.int32),
        probabilities=torch.tensor([0.6, 0.4]),
        costs=torch.tensor([0.0, 1.0]),
    )


def test_postprocess_converts_counts_to_sample_distribution() -> None:
    module = QuboToPasqal()
    result = module.postprocess(Data(make_solution()))

    assert isinstance(result, Data)
    assert isinstance(result.data, SampleDistribution)
    assert result.data.as_list() == [("00", 3.0), ("10", 2.0)]
    assert result.data.nbshots == 5


def test_postprocess_converts_deterministic_solution() -> None:
    solution = SimpleNamespace(
        bitstrings=torch.tensor([[0, 0]], dtype=torch.int32),
        counts=None,
        probabilities=None,
    )

    result = QuboToPasqal().postprocess(Data(solution))

    assert isinstance(result, Data)
    assert result.data.as_list() == [("00", 1.0)]
    assert result.data.nbshots == 1


def test_postprocess_uses_probabilities_when_counts_do_not_match() -> None:
    solution = SimpleNamespace(
        bitstrings=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        counts=torch.tensor([5], dtype=torch.int32),
        probabilities=torch.tensor([0.6, 0.4]),
    )

    result = QuboToPasqal().postprocess(Data(solution))

    assert isinstance(result, Data)
    np.testing.assert_allclose(
        [probability for _, probability in result.data.as_list()],
        [0.6, 0.4],
    )
    assert result.data.nbshots == 5


def test_postprocess_rejects_missing_solution() -> None:
    result = QuboToPasqal().postprocess(None)

    assert isinstance(result, Failed)
    assert result.reason == "Pasqal solver returned no solution"


def test_postprocess_rejects_malformed_solution() -> None:
    solution = SimpleNamespace(
        bitstrings=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        counts=torch.tensor([1], dtype=torch.int32),
        probabilities=None,
    )

    result = QuboToPasqal().postprocess(Data(solution))

    assert isinstance(result, Failed)
    assert "neither matching counts nor probabilities" in result.reason


def test_invalid_input_returns_failed() -> None:
    result = QuboToPasqal().preprocess("not a qubo")

    assert isinstance(result, Failed)
    assert "Expected Qubo" in result.reason


def test_invalid_matrix_returns_failed() -> None:
    qubo = Qubo.from_matrix(np.ones((2, 3)))
    result = QuboToPasqal().preprocess(qubo)

    assert isinstance(result, Failed)
    assert "square" in result.reason


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "negative off-diagonal"),
    ],
)
def test_invalid_quantum_matrix_returns_failed(matrix, message) -> None:
    result = QuboToPasqal(use_quantum=True).preprocess(Qubo.from_matrix(matrix))

    assert isinstance(result, Failed)
    assert message in result.reason


def test_quantum_solver_rejects_more_than_80_variables() -> None:
    matrix = np.zeros((81, 81))

    result = QuboToPasqal(use_quantum=True).preprocess(Qubo.from_matrix(matrix))

    assert isinstance(result, Failed)
    assert "at most 80 variables" in result.reason


def test_preprocess_solves_qubo_and_records_metrics(monkeypatch) -> None:
    solution = make_solution()

    class FakeQuboSolver:
        def __init__(self, instance, config) -> None:
            assert instance.coefficients.shape == (2, 2)
            assert config.use_quantum is False

        def solve(self):
            return solution

    monkeypatch.setattr("quark_plugin_pasqal.qubo_to_pasqal.QuboSolver", FakeQuboSolver)
    module = QuboToPasqal(use_quantum=False)

    result = module.preprocess(Qubo.from_matrix(np.diag([-1.0, -1.0])))

    assert isinstance(result, Data)
    assert result.data.data is solution
    metrics = module.get_metrics()

    assert metrics["qubo_size"] == 2
    assert metrics["num_samples"] == 2
    assert metrics["best_cost"] == 0.0
    assert metrics["best_bitstrings"] == ["00"]
    assert metrics["runtime_s"] is not None

    postprocessed = module.postprocess(result)
    assert isinstance(postprocessed, Data)
    assert postprocessed.data.as_list() == [("00", 3.0), ("10", 2.0)]


def test_preprocess_returns_failed_when_solver_raises(monkeypatch) -> None:
    class FailingQuboSolver:
        def __init__(self, instance, config) -> None:
            pass

        def solve(self):
            raise RuntimeError("solver unavailable")

    monkeypatch.setattr("quark_plugin_pasqal.qubo_to_pasqal.QuboSolver", FailingQuboSolver)

    result = QuboToPasqal(use_quantum=False).preprocess(Qubo.from_matrix(np.zeros((2, 2))))

    assert isinstance(result, Failed)
    assert result.reason == "Pasqal QUBO solve failed: solver unavailable"
