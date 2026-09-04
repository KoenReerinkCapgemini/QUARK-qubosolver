"""Convert LP problems to QUBOs for the QUARK pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems.quadratic_program import VarType

from quark.core import Core, Data, Failed, Result
from quark.interface_types.lp import LP
from quark.interface_types.qubo import Qubo


class LpToQuboConverter(Core):
    """Convert binary LP problems to QUARK ``Qubo`` objects with Qiskit."""

    def __init__(
        self,
        penalty_factor: float = 1e6,
        path_to_lp: str | Path | None = None,
    ) -> None:
        """Initialize a converter.

        Args:
            penalty_factor: Weight applied to constraints when Qiskit converts
                the quadratic program to a QUBO.
            path_to_lp: Optional LP file to read instead of the input passed to
                :meth:`preprocess`.
        """
        self.penalty_factor = penalty_factor
        self.path_to_lp = path_to_lp
        self.runtime_s: float | None = None
        self.original_num_variables: int | None = None
        self.qubo_num_variables: int | None = None

    def preprocess(self, data: Any) -> Result:
        """Parse an LP with Qiskit and convert it to a QUBO.

        Args:
            data: A QUARK :class:`LP`, unless ``path_to_lp`` was configured.

        Returns:
            ``Data(Qubo)`` on success, or ``Failed`` with a conversion error.
        """
        start_time = time.perf_counter()

        if self.path_to_lp is not None:
            try:
                data = LP.from_file(self.path_to_lp)
            except OSError as error:
                return Failed(reason=f"Failed to read LP file '{self.path_to_lp}': {error}")

        if not isinstance(data, LP):
            return Failed(reason=f"Expected LP, got {type(data).__name__}")

        try:
            qp = self._read_lp(data.as_str())
            if qp.get_num_vars() == 0:
                raise ValueError("objective section is empty")

            self.original_num_variables = qp.get_num_vars()
            non_binary = [
                variable.name
                for variable in qp.variables
                if variable.vartype != VarType.BINARY
            ]
            if non_binary:
                raise ValueError(
                    "Only binary variables are supported; "
                    f"unsupported variables: {', '.join(non_binary)}"
                )

            qubo_qp = QuadraticProgramToQubo(penalty=self.penalty_factor).convert(qp)
            matrix = self._qubo_matrix(qubo_qp)
            self.qubo_num_variables = matrix.shape[0]
            self.runtime_s = time.perf_counter() - start_time
            return Data(Qubo.from_matrix(matrix))
        except Exception as error:
            return Failed(reason=f"Failed to convert LP to QUBO: {error}")

    def postprocess(self, data: Any) -> Result:
        """Pass the conversion result through unchanged in a QUARK ``Data`` wrapper."""
        return Data(data)

    def get_metrics(self) -> dict[str, Any]:
        """Return conversion timing, variable counts, and the penalty factor."""
        return {
            "runtime_s": self.runtime_s,
            "original_num_variables": self.original_num_variables,
            "qubo_num_variables": self.qubo_num_variables,
            "penalty_factor": self.penalty_factor,
        }

    def get_unique_name(self) -> str:
        """Return a stable name that includes the configured penalty factor."""
        return f"lp_to_qubo_pen{self.penalty_factor:.0e}"

    @staticmethod
    def _read_lp(lp_string: str) -> QuadraticProgram:
        """Parse an LP string through Qiskit's file-based parser.

        Writing a temporary file keeps this compatible with Qiskit versions
        that do not expose a string-parsing method.
        """
        with TemporaryDirectory() as temporary_directory:
            lp_path = Path(temporary_directory) / "problem.lp"
            lp_path.write_text(lp_string)
            program = QuadraticProgram()
            program.read_from_lp_file(str(lp_path))
            return program

    @staticmethod
    def _qubo_matrix(program: QuadraticProgram) -> np.ndarray:
        """Build a symmetric matrix from a Qiskit quadratic-program objective."""
        matrix = np.zeros((program.get_num_vars(), program.get_num_vars()))

        for index, coefficient in program.objective.linear.to_dict(use_name=False).items():
            matrix[index, index] += coefficient

        for (row, column), coefficient in program.objective.quadratic.to_dict(
            use_name=False
        ).items():
            matrix[row, column] += coefficient
            if row != column:
                matrix[column, row] += coefficient

        return matrix
