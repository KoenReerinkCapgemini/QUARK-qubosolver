from __future__ import annotations

import torch

from src.PLUGIN_QUBOSolver_FORMATTED.QUBOSolver_module import QuboSolverModule


if __name__ == "__main__":
    module = QuboSolverModule(
        use_quantum=False,
        classical_solver_type="cplex",
        cplex_maxtime=10.0,
        cplex_log_path="test_solver.log",
    )

    data = {
        "coefficients": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "use_quantum": False,
        "config": {
            "classical": {
                "classical_solver_type": "cplex",
                "cplex_maxtime": 10.0,
                "cplex_log_path": "test_solver.log",
            }
        },
    }

    prepared = module.preprocess(data)
    print("Prepared payload:", prepared)

    solved = module.postprocess(prepared)
    print("Solved result:", solved)
