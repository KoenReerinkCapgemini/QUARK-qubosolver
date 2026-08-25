"""Tests for the QUARK Pasqal adapter."""

from types import SimpleNamespace

import numpy as np
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


def test_invalid_input_returns_failed() -> None:
    result = QuboToPasqal().preprocess("not a qubo")

    assert isinstance(result, Failed)
    assert "Expected Qubo" in result.reason


def test_invalid_matrix_returns_failed() -> None:
    qubo = Qubo.from_matrix(np.ones((2, 3)))
    result = QuboToPasqal().preprocess(qubo)

    assert isinstance(result, Failed)
    assert "square" in result.reason


def test_metrics_are_recorded_after_postprocess() -> None:
    module = QuboToPasqal()
    module._solution = make_solution()
    module.qubo_size = 2
    module.num_samples = 2
    module.best_cost = 0.0
    module.runtime_s = 0.01

    metrics = module.get_metrics()

    assert metrics["qubo_size"] == 2
    assert metrics["num_samples"] == 2
    assert metrics["best_cost"] == 0.0
