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


def test_parse_lp_string_preserves_objective_variables_and_constraints() -> None:
    converter = LpToQuboConverter()

    program = converter._parse_lp_string_to_quadratic_program(BINARY_LP)

    assert [variable.name for variable in program.variables] == ["x", "y"]
    assert program.objective.linear.to_dict(use_name=True) == {"x": 1.0, "y": 2.0}
    assert program.linear_constraints[0].sense.name == "GE"
    assert program.linear_constraints[0].rhs == 1.0


def test_parse_lp_string_reads_bounds_and_binary_declarations() -> None:
    converter = LpToQuboConverter()
    lp = """
    Maximize
     profit: 3 x - y
    Bounds
     -2 <= x <= 4
    Binaries
     y
    End
    """

    program = converter._parse_lp_string_to_quadratic_program(lp)

    x, y = program.variables
    assert (x.name, x.lowerbound, x.upperbound) == ("x", -2.0, 4.0)
    assert y.vartype.name == "BINARY"
    assert program.objective.sense.name == "MAXIMIZE"


def test_parse_lp_string_rejects_missing_objective_sense() -> None:
    with pytest.raises(ValueError, match="missing Minimize/Maximize"):
        LpToQuboConverter()._parse_lp_string_to_quadratic_program("obj: x\nEnd")


def test_preprocess_uses_fallback_parser_when_qiskit_parser_fails(monkeypatch) -> None:
    def fail_to_parse(self, lp_str):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(QuadraticProgram, "read_from_lp_file", fail_to_parse)

    result = LpToQuboConverter(penalty_factor=10).preprocess(LP.from_str(BINARY_LP))

    assert isinstance(result, Data)
    np.testing.assert_allclose(result.data.as_matrix(), [[-9.0, 10.0], [10.0, -8.0]])


def test_parse_lp_string_supports_constraint_senses_and_linear_terms() -> None:
    lp = """
    Minimize
     obj: x + 2*y - 0.5*z
    Subject To
     upper: x + y <= 2
     exact: x - y = 0
    Bounds
     -1 <= z <= 3
    Binaries
     x y
    End
    """

    program = LpToQuboConverter()._parse_lp_string_to_quadratic_program(lp)

    assert program.objective.linear.to_dict(use_name=True) == {
        "x": 1.0,
        "y": 2.0,
        "z": -0.5,
    }
    assert [constraint.sense.name for constraint in program.linear_constraints] == [
        "LE",
        "EQ",
    ]
    assert program.variables[2].name == "z"
    assert (program.variables[2].lowerbound, program.variables[2].upperbound) == (-1.0, 3.0)


def test_parse_lp_string_rejects_unsupported_bounds() -> None:
    lp = "Minimize\n obj: x\nBounds\n x >= 0\nBinaries\n x\nEnd"

    with pytest.raises(ValueError, match="Unsupported bounds line"):
        LpToQuboConverter()._parse_lp_string_to_quadratic_program(lp)


def test_extract_qubo_matrix_is_symmetric() -> None:
    program = QuadraticProgram()
    program.binary_var("x")
    program.binary_var("y")
    program.minimize(linear={"x": 1.0}, quadratic={("x", "y"): 3.0, ("y", "y"): 2.0})

    matrix = LpToQuboConverter()._extract_qubo_matrix(program)

    np.testing.assert_allclose(matrix, [[1.0, 3.0], [3.0, 2.0]])


def test_preprocess_binary_lp_returns_penalized_qubo_and_metrics() -> None:
    converter = LpToQuboConverter(penalty_factor=10)

    result = converter.preprocess(LP.from_str(BINARY_LP))

    assert isinstance(result, Data)
    np.testing.assert_allclose(result.data.as_matrix(), [[-9.0, 10.0], [10.0, -8.0]])
    assert converter.get_metrics()["original_num_variables"] == 2
    assert converter.get_metrics()["discretised_num_variables"] == 2
    assert converter.runtime_s is not None


def test_preprocess_rejects_non_lp_input() -> None:
    result = LpToQuboConverter().preprocess("not an LP")

    assert isinstance(result, Failed)
    assert result.reason == "Expected LP, got str"


def test_preprocess_reports_invalid_lp() -> None:
    result = LpToQuboConverter().preprocess(LP.from_str("Minimize\nEnd"))

    assert isinstance(result, Failed)
    assert "Failed to parse LP string" in result.reason
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
