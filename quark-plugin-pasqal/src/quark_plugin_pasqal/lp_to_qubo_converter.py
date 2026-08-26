"""LP to QUBO converter using qiskit-optimization."""

from __future__ import annotations

import time
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems.quadratic_program import VarType

if TYPE_CHECKING:
    from quark.interface_types import Qubo

from quark.core import Core, Data, Failed, Result
from quark.interface_types.lp import LP
from quark.interface_types.qubo import Qubo as QuboType


class LpToQuboConverter(Core):
    """Convert binary linear programs to QUBO problems using qiskit-optimization."""

    def __init__(
        self,
        penalty_factor: float = 1e6,
        continuous_var_precision: int = 8,
        discretisation_scale: float = 1.0,
        path_to_lp: str | Path | None = None,
    ):
        """Initialize the LP to QUBO converter.
        :param path_to_lp: Optional path to the LP file to convert. When omitted, the LP is read from the input data.
        :param penalty_factor: Multiplier for constraint penalty terms in the objective.
            Larger values enforce constraint satisfaction more strictly. Default 1e6.
        :param continuous_var_precision: Retained for configuration compatibility; non-binary
            variables are rejected.
        :param discretisation_scale: Retained for configuration compatibility; non-binary
            variables are rejected.
        """
        self.path_to_lp = path_to_lp
        self.penalty_factor = penalty_factor
        self.continuous_var_precision = continuous_var_precision
        self.discretisation_scale = discretisation_scale
        self.runtime_s: float | None = None
        self.original_num_variables: int | None = None
        self.discretised_num_variables: int | None = None
        self.embedding_error: float | None = None

    def preprocess(self, data: Any) -> Result:
        """Convert an LP to a QUBO.
        
        :param data: LP instance containing the LP string
        :return: Data(Qubo) on success, Failed(...) on error
        """
        start_time = time.time()

        if self.path_to_lp is not None:
            try:
                data = LP.from_file(self.path_to_lp)
            except OSError as e:
                return Failed(reason=f"Failed to read LP file '{self.path_to_lp}': {e}")

        if not isinstance(data, LP):
            return Failed(reason=f"Expected LP, got {type(data).__name__}")

        lp_str = data.as_str()
        try:
            qp = QuadraticProgram()

            if hasattr(qp, "read_from_lp_string"):
                qp.read_from_lp_string(lp_str)
            else:
                # qiskit-optimization>=0.7 removed read_from_lp_string; try temporary .lp file.
                with TemporaryDirectory() as tmp_dir:
                    lp_file = Path(tmp_dir) / "problem.lp"
                    lp_file.write_text(lp_str)
                    qp.read_from_lp_file(str(lp_file))
        except Exception:
            # Some qiskit/docplex parser paths require IBM CPLEX. Fall back to an internal parser.
            try:
                qp = self._parse_lp_string_to_quadratic_program(lp_str)
            except Exception as e:
                return Failed(reason=f"Failed to parse LP string: {str(e)}")

        if qp.get_num_vars() == 0:
            return Failed(reason="Failed to parse LP string: objective section is empty")

        self.original_num_variables = qp.get_num_vars()

        non_binary_variables = [var.name for var in qp.variables if var.vartype != VarType.BINARY]
        if non_binary_variables:
            return Failed(
                reason=(
                    "Only binary variables are supported by the QUBO converter; "
                    f"unsupported variables: {', '.join(non_binary_variables)}"
                )
            )

        self.discretised_num_variables = qp.get_num_vars()

        # Use qiskit's converter to turn the QP into a QUBO (encodes constraints as penalties)
        try:
            converter = QuadraticProgramToQubo(penalty=self.penalty_factor)
            qubo_qp = converter.convert(qp)
        except Exception as e:
            return Failed(reason=f"Failed to convert to QUBO: {str(e)}")

        # Extract the QUBO matrix from the quadratic program
        try:
            qubo_matrix = self._extract_qubo_matrix(qubo_qp)
            qubo = QuboType.from_matrix(qubo_matrix)
        except Exception as e:
            return Failed(reason=f"Failed to extract QUBO matrix: {str(e)}")

        self.runtime_s = time.time() - start_time

        return Data(qubo)

    def postprocess(self, data: Any) -> Result:
        """Pass through post-processing (no transformation needed)."""
        return Data(data)

    def get_metrics(self) -> dict:
        """Return metrics about the conversion process."""
        return {
            "runtime_s": self.runtime_s,
            "original_num_variables": self.original_num_variables,
            "discretised_num_variables": self.discretised_num_variables,
            "penalty_factor": self.penalty_factor,
            "continuous_var_precision": self.continuous_var_precision,
        }

    def get_unique_name(self) -> str | None:
        """Return a unique identifier for this converter instance."""
        return f"lp_to_qubo_pen{self.penalty_factor:.0e}_prec{self.continuous_var_precision}"

    def _extract_qubo_matrix(self, qp: QuadraticProgram) -> np.ndarray:
        """Extract the QUBO matrix from a QuadraticProgram.
        
        :param qp: QuadraticProgram (ideally with only binary variables)
        :return: Symmetric QUBO matrix as numpy array
        """
        n = qp.get_num_vars()
        matrix = np.zeros((n, n))

        # Extract linear terms (diagonal)
        linear = qp.objective.linear.to_dict(use_name=False)
        for i, coeff in linear.items():
            matrix[i, i] += coeff

        # Extract quadratic terms (off-diagonal)
        quadratic = qp.objective.quadratic.to_dict(use_name=False)
        for (i, j), coeff in quadratic.items():
            if i == j:
                matrix[i, i] += coeff
            else:
                matrix[i, j] += coeff
                matrix[j, i] += coeff

        return matrix

    def _parse_lp_string_to_quadratic_program(self, lp_str: str) -> QuadraticProgram:
        """Parse a CPLEX-style LP string into a QuadraticProgram without requiring CPLEX."""
        lines = [line.strip() for line in lp_str.splitlines() if line.strip()]
        sections: dict[str, list[str]] = {
            "objective": [],
            "constraints": [],
            "bounds": [],
            "binaries": [],
        }

        sense: str | None = None
        current_section: str | None = None

        for raw_line in lines:
            line = raw_line.strip()
            lower = line.lower()

            if lower in {"minimize", "minimum", "min"}:
                sense = "min"
                current_section = "objective"
                continue
            if lower in {"maximize", "maximum", "max"}:
                sense = "max"
                current_section = "objective"
                continue
            if lower in {"subject to", "such that", "st", "s.t."}:
                current_section = "constraints"
                continue
            if lower == "bounds":
                current_section = "bounds"
                continue
            if lower == "binaries":
                current_section = "binaries"
                continue
            if lower == "end":
                break

            if current_section is not None:
                sections[current_section].append(line)

        if sense is None:
            raise ValueError("LP string is missing Minimize/Maximize section")

        objective_expr = " ".join(sections["objective"]).strip()
        if not objective_expr:
            raise ValueError("LP objective section is empty")
        if ":" in objective_expr:
            objective_expr = objective_expr.split(":", 1)[1].strip()

        objective_linear = self._parse_linear_expression(objective_expr)

        bounds: dict[str, tuple[float, float]] = {}
        binaries: set[str] = set()

        for line in sections["bounds"]:
            match = re.match(
                r"^([+-]?\d+(?:\.\d+)?)\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([+-]?\d+(?:\.\d+)?)$",
                line,
            )
            if not match:
                raise ValueError(f"Unsupported bounds line: {line}")
            lb, name, ub = match.groups()
            bounds[name] = (float(lb), float(ub))

        for line in sections["binaries"]:
            binaries.update(token for token in line.split() if token)

        vars_in_objective = set(objective_linear)
        vars_in_constraints: set[str] = set()
        parsed_constraints: list[tuple[str, dict[str, float], str, float]] = []

        for idx, line in enumerate(sections["constraints"]):
            name = f"c{idx + 1}"
            expression = line
            if ":" in line:
                name_part, expression = line.split(":", 1)
                if name_part.strip():
                    name = name_part.strip()
            expression = expression.strip()

            sense_match = re.search(r"(<=|>=|=)", expression)
            if not sense_match:
                raise ValueError(f"Constraint missing sense (<=, >=, =): {line}")

            op = sense_match.group(1)
            lhs = expression[: sense_match.start()].strip()
            rhs_str = expression[sense_match.end() :].strip()

            lhs_linear = self._parse_linear_expression(lhs)
            vars_in_constraints.update(lhs_linear)

            try:
                rhs = float(rhs_str)
            except ValueError as exc:
                raise ValueError(f"Constraint RHS must be numeric: {line}") from exc

            parsed_constraints.append((name, lhs_linear, op, rhs))

        all_vars = vars_in_objective | vars_in_constraints | set(bounds) | binaries
        qp = QuadraticProgram()

        for name in sorted(all_vars):
            if name in binaries:
                qp.binary_var(name=name)
            else:
                lb, ub = bounds.get(name, (0.0, 1.0))
                qp.continuous_var(name=name, lowerbound=lb, upperbound=ub)

        if sense == "min":
            qp.minimize(linear=objective_linear)
        else:
            qp.maximize(linear=objective_linear)

        for name, lhs_linear, op, rhs in parsed_constraints:
            qp.linear_constraint(linear=lhs_linear, sense=op, rhs=rhs, name=name)

        return qp

    def _parse_linear_expression(self, expr: str) -> dict[str, float]:
        """Parse a linear expression like `x + 2*y - 3*z` into a coefficient map."""
        text = expr.strip().replace(" ", "")
        if not text:
            return {}
        if text[0] not in "+-":
            text = "+" + text

        coeffs: dict[str, float] = {}
        for match in re.finditer(r"([+-])([^+-]+)", text):
            sign, term = match.groups()
            factor = -1.0 if sign == "-" else 1.0

            if "*" in term:
                coeff_str, var = term.split("*", 1)
            else:
                coeff_match = re.match(r"(\d+(?:\.\d+)?)([A-Za-z_][A-Za-z0-9_]*)$", term)
                if coeff_match:
                    coeff_str, var = coeff_match.groups()
                else:
                    coeff_str, var = "1", term

            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", var):
                raise ValueError(f"Unsupported linear term: {term}")

            coeff = float(coeff_str) * factor
            coeffs[var] = coeffs.get(var, 0.0) + coeff

        return coeffs
