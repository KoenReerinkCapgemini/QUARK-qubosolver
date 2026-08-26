"""Run QUARK QUBO instances with the Pasqal QUBO solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from quark.core import Core, Data, Failed, Result
from quark.interface_types import Other, Qubo, SampleDistribution
from qubosolver.config import SolverConfig
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.solver import QuboSolver


@dataclass
class QuboToPasqal(Core):
    """Solve a QUARK ``Qubo`` and return a ``SampleDistribution``.

    ``solver_config`` is passed directly to ``qubo-solver``. If it is omitted,
    the plugin creates a local quantum configuration.
    """

    solver_config: SolverConfig | None = None
    use_quantum: bool | None = None
    validate_input: bool = True

    def __post_init__(self) -> None:
        self.runtime_s: float | None = None
        self.qubo_size: int | None = None
        self.best_cost: float | None = None
        self.num_samples: int | None = None
        self._solution: Any = None

    def preprocess(self, data: Any) -> Result:
        """Validate, solve, and carry the raw solver result downstream."""
        if not isinstance(data, Qubo):
            return Failed(reason=f"Expected Qubo, got {type(data).__name__}")

        try:
            matrix = np.asarray(data.as_matrix(), dtype=np.float32)
            self._validate_matrix(matrix)
            self.qubo_size = int(matrix.shape[0])
            instance = QUBOInstance(coefficients=torch.from_numpy(matrix))

            start_time = time.perf_counter()
            config = self._effective_config()
            solver = QuboSolver(instance, config)
            self._solution = solver.solve()
            self.runtime_s = time.perf_counter() - start_time
            self.num_samples = int(self._solution.bitstrings.shape[0])
            if self._solution.costs.numel() > 0:
                self.best_cost = float(self._solution.costs.min().item())
        except (ValueError, TypeError, RuntimeError) as error:
            return Failed(reason=f"Pasqal QUBO solve failed: {error}")

        return Data(Other(self._solution))

    def postprocess(self, data: Any) -> Result:
        """Convert the solver result into QUARK's sample datatype."""
        if isinstance(data, Data):
            solution = data.data
        elif isinstance(data, Other):
            solution = data.data
        elif data is not None:
            solution = data
        else:
            solution = self._solution
        if isinstance(solution, Other):
            solution = solution.data
        if solution is None:
            return Failed(reason="Pasqal solver returned no solution")

        try:
            samples = self._to_samples(solution)
            shots = self._shot_count(solution)
            return Data(SampleDistribution.from_list(samples, shots))
        except (ValueError, TypeError, AttributeError) as error:
            return Failed(reason=f"Could not convert Pasqal result: {error}")

    def get_metrics(self) -> dict[str, Any]:
        """Return benchmark metrics collected during execution."""
        return {
            "runtime_s": self.runtime_s,
            "qubo_size": self.qubo_size,
            "num_samples": self.num_samples,
            "best_cost": self.best_cost,
            "solver_mode": self._solver_mode(),
        }

    def get_unique_name(self) -> str:
        """Return a stable module name for benchmark output files."""
        return f"pasqal_{self._solver_mode()}"

    def _validate_matrix(self, matrix: np.ndarray) -> None:
        if not self.validate_input:
            return
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QUBO matrix must be square")
        if not np.isfinite(matrix).all():
            raise ValueError("QUBO matrix must contain only finite values")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("QUBO matrix must be symmetric")
        use_quantum = self._uses_quantum()
        if use_quantum and matrix.shape[0] > 80:
            raise ValueError("Pasqal quantum solver supports at most 80 variables")
        if use_quantum and np.any(matrix[~np.eye(matrix.shape[0], dtype=bool)] < 0):
            raise ValueError("Pasqal quantum solver does not support negative off-diagonal coefficients")

    @staticmethod
    def _to_samples(solution: Any) -> list[tuple[str, float]]:
        bitstrings = solution.bitstrings.tolist()
        counts = solution.counts
        probabilities = solution.probabilities
        if counts is not None and counts.numel() == len(bitstrings):
            return [
                ("".join(str(int(bit)) for bit in bits), float(count))
                for bits, count in zip(bitstrings, counts.tolist())
            ]
        if probabilities is not None and probabilities.numel() == len(bitstrings):
            return [
                ("".join(str(int(bit)) for bit in bits), float(probability))
                for bits, probability in zip(bitstrings, probabilities.tolist())
            ]
        if counts is None and probabilities is None and len(bitstrings) == 1:
            return [("".join(str(int(bit)) for bit in bitstrings[0]), 1.0)]
        raise ValueError("solver result contains neither matching counts nor probabilities")

    @staticmethod
    def _shot_count(solution: Any) -> int:
        if solution.counts is None:
            return 1 if solution.probabilities is None else 0
        return int(solution.counts.sum().item())

    def _solver_mode(self) -> str:
        return "quantum" if self._uses_quantum() else "classical"

    def _effective_config(self) -> SolverConfig:
        if self.solver_config is not None:
            return self.solver_config
        return SolverConfig(use_quantum=self.use_quantum if self.use_quantum is not None else True)

    def _uses_quantum(self) -> bool:
        if self.use_quantum is not None:
            return self.use_quantum
        return self.solver_config is None or self.solver_config.use_quantum is not False
