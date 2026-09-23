"""Tests for the LP-to-QUBO converter."""

import numpy as np
import pytest
from qiskit_optimization import QuadraticProgram

from quark.core import Data, Failed
from quark.interface_types.lp import LP
from quark_plugin_pasqal import LpToQuboConverter


BINARY_LP = """
Minimize
 obj: x + 2 y
Subject To
 c1: x + y >= 1
Binaries
 x y
End
"""


def test_qiskit_parser_converts_binary_lp() -> None:
    result = LpToQuboConverter(penalty_factor=10).preprocess(LP.from_str(BINARY_LP))

    assert isinstance(result, Data)
    np.testing.assert_allclose(result.data.as_matrix(), [[-9.0, 10.0], [10.0, -8.0]])


def test_qubo_matrix_is_symmetric() -> None:
    program = QuadraticProgram()
    program.binary_var("x")
    program.binary_var("y")
    program.minimize(linear={"x": 1.0}, quadratic={("x", "y"): 3.0, ("y", "y"): 2.0})

    matrix = LpToQuboConverter._qubo_matrix(program)

    np.testing.assert_allclose(matrix, [[1.0, 3.0], [3.0, 2.0]])


def test_preprocess_binary_lp_returns_penalized_qubo_and_metrics() -> None:
    converter = LpToQuboConverter(penalty_factor=10)

    result = converter.preprocess(LP.from_str(BINARY_LP))

    assert isinstance(result, Data)
    np.testing.assert_allclose(result.data.as_matrix(), [[-9.0, 10.0], [10.0, -8.0]])
    assert converter.get_metrics()["original_num_variables"] == 2
    assert converter.get_metrics()["qubo_num_variables"] == 2
    assert converter.runtime_s is not None


def test_preprocess_rejects_non_lp_input() -> None:
    result = LpToQuboConverter().preprocess("not an LP")

    assert isinstance(result, Failed)
    assert result.reason == "Expected LP, got str"


def test_preprocess_reports_invalid_lp() -> None:
    result = LpToQuboConverter().preprocess(LP.from_str("Minimize\nEnd"))

    assert isinstance(result, Failed)
    assert "Failed to convert LP to QUBO" in result.reason
    assert "objective section is empty" in result.reason


def test_preprocess_rejects_continuous_lp() -> None:
    lp = """
    Minimize
     obj: 2 x
    Bounds
     0 <= x <= 3
    End
    """

    result = LpToQuboConverter().preprocess(LP.from_str(lp))

    assert isinstance(result, Failed)
    assert "Only binary variables are supported" in result.reason


def test_preprocess_reads_lp_from_file(tmp_path) -> None:
    lp_path = tmp_path / "problem.lp"
    lp_path.write_text(BINARY_LP)

    result = LpToQuboConverter(path_to_lp=lp_path, penalty_factor=10).preprocess(None)

    assert isinstance(result, Data)
    np.testing.assert_allclose(result.data.as_matrix(), [[-9.0, 10.0], [10.0, -8.0]])


def test_preprocess_reports_unreadable_lp_file(tmp_path) -> None:
    lp_path = tmp_path / "missing.lp"

    result = LpToQuboConverter(path_to_lp=lp_path).preprocess(None)

    assert isinstance(result, Failed)
    assert f"Failed to read LP file '{lp_path}'" in result.reason
