from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, override

from quark.core import Core, Result
from quark.interface_types import InterfaceType

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from qubosolver import QUBOInstance
    from qubosolver.config import ClassicalConfig, SolverConfig
    from qubosolver.solver import QuboSolver, QuboSolverClassical, QuboSolverQuantum
except ImportError:  # pragma: no cover - optional dependency
    QUBOInstance = None
    ClassicalConfig = None
    SolverConfig = None
    QuboSolver = None
    QuboSolverClassical = None
    QuboSolverQuantum = None


@dataclass
class QuboSolverModule(Core):
    """
    QUARK module that wraps the qubosolver package for solving QUBO instances.

    The module accepts either:
    - a dictionary with a ``coefficients`` / ``qubo`` entry, or
    - an existing ``QUBOInstance`` object.

    The preprocessing step prepares the instance and configuration, while the
    postprocessing step solves it and returns the result.
    """

    use_quantum: bool = False
    classical_solver_type: str = "cplex"
    cplex_maxtime: float = 10.0
    cplex_log_path: str | None = None

    @override
    def preprocess(self, data: InterfaceType) -> Result:
        payload = self._prepare_payload(data)
        payload["status"] = "prepared"
        return payload

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        payload = self._prepare_payload(data)
        solver = self._build_solver(
            payload["instance"],
            payload.get("config"),
            use_quantum=payload.get("use_quantum", self.use_quantum),
        )
        payload["solution"] = solver.solve()
        payload["status"] = "solved"
        return payload

    def _prepare_payload(self, data: InterfaceType) -> dict[str, Any]:
        if isinstance(data, Mapping):
            payload = dict(data)
            instance = payload.get("instance")
            if instance is None:
                coefficients = payload.get("coefficients", payload.get("qubo", payload.get("matrix")))
                instance = self._build_instance(coefficients)
            config = payload.get("config")
            use_quantum = self._resolve_use_quantum(payload.get("use_quantum"), config)
            return {"instance": instance, "config": config, "use_quantum": use_quantum}

        if isinstance(data, dict):
            return {"instance": self._build_instance(data), "config": None, "use_quantum": self.use_quantum}

        if data is None:
            raise ValueError("No QUBO input was provided.")

        return {"instance": self._build_instance(data), "config": None, "use_quantum": self.use_quantum}

    def _build_instance(self, coefficients: Any) -> Any:
        if QUBOInstance is None:
            raise ImportError("The qubosolver package is required to build QUBO instances.")
        if torch is None:
            raise ImportError("PyTorch is required to convert QUBO coefficients into tensors.")

        if hasattr(coefficients, "coefficients") and hasattr(coefficients, "__class__"):
            return coefficients

        if isinstance(coefficients, Mapping):
            coefficients = coefficients.get("coefficients", coefficients.get("qubo", coefficients.get("matrix")))

        matrix = torch.tensor(coefficients, dtype=torch.float32)
        if matrix.ndim != 2:
            raise ValueError("QUBO coefficients must be provided as a 2D matrix.")
        return QUBOInstance(coefficients=matrix)

    def _build_solver(self, instance: Any, config: Any = None, use_quantum: bool | None = None) -> Any:
        resolved_use_quantum = self._resolve_use_quantum(use_quantum, config)
        if config is None:
            config = self._build_solver_config(resolved_use_quantum)
        elif isinstance(config, Mapping):
            config = self._coerce_solver_config(config, resolved_use_quantum)

        if QuboSolver is not None:
            return QuboSolver(instance, config)

        if resolved_use_quantum:
            if QuboSolverQuantum is None:
                raise ImportError("The quantum QUBO solver backend is not available.")
            return QuboSolverQuantum(instance, config)

        if QuboSolverClassical is None:
            raise ImportError("The classical QUBO solver backend is not available.")
        return QuboSolverClassical(instance, config)

    def _resolve_use_quantum(self, use_quantum: Any, config: Any = None) -> bool:
        if use_quantum is not None:
            return bool(use_quantum)
        if isinstance(config, Mapping):
            if "use_quantum" in config:
                return bool(config["use_quantum"])
        if hasattr(config, "use_quantum"):
            return bool(getattr(config, "use_quantum"))
        return bool(self.use_quantum)

    def _coerce_solver_config(self, config: Mapping[str, Any], use_quantum: bool) -> Any:
        if SolverConfig is None or ClassicalConfig is None:
            raise ImportError("The qubosolver configuration classes are required.")

        config_payload = dict(config)
        classical_config = config_payload.get("classical")
        if classical_config is None and not use_quantum:
            classical_config = self._build_classical_config()
        elif isinstance(classical_config, Mapping):
            classical_config = self._build_classical_config(classical_config)
        elif classical_config is None and use_quantum:
            classical_config = self._build_classical_config()

        return SolverConfig(use_quantum=use_quantum, classical=classical_config)

    def _build_classical_config(self, config: Any = None) -> Any:
        if ClassicalConfig is None:
            raise ImportError("The qubosolver configuration classes are required.")

        if config is None:
            config = {}

        if isinstance(config, Mapping):
            options = dict(config)
            return ClassicalConfig(
                classical_solver_type=options.get("classical_solver_type", self.classical_solver_type),
                cplex_maxtime=options.get("cplex_maxtime", self.cplex_maxtime),
                cplex_log_path=options.get("cplex_log_path", self.cplex_log_path or "quark_solver.log"),
            )

        return config

    def _build_solver_config(self, use_quantum: bool | None = None) -> Any:
        if SolverConfig is None or ClassicalConfig is None:
            raise ImportError("The qubosolver configuration classes are required.")

        resolved_use_quantum = self._resolve_use_quantum(use_quantum)
        classical_config = self._build_classical_config()
        return SolverConfig(use_quantum=resolved_use_quantum, classical=classical_config)
