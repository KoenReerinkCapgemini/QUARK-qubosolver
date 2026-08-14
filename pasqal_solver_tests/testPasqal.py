import torch
from qubosolver import QUBOInstance
from qubosolver.config import SolverConfig
from qubosolver.solver import QuboSolver


# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = QUBOInstance(coefficients=Q)

# Create a SolverConfig object to use a quantum backend
config = SolverConfig(use_quantum=True)

# Instantiate the quantum solver.
solver = QuboSolver(instance, config)

# Solve the QUBO problem.
solution = solver.solve()
print(solution)

# Returns the following
# QUBOSolution(bitstrings=tensor([[0, 0]]), costs=tensor([0.]), counts=None, probabilities=None, solution_status=)