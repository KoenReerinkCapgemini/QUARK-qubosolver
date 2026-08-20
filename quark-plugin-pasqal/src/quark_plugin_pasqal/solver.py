"""Pasqal QAA solver: embeds a QUBO onto a neutral-atom register and solves it via the
Quantum Adiabatic Algorithm (QAA) using Pulser.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pulser
from pulser.backend import BitStrings
from pulser.devices import AnalogDevice, DigitalAnalogDevice
from pulser_simulation import QutipBackendV2, QutipConfig
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from quark.core import Core, Data, Failed, Result
from quark.interface_types.qubo import Qubo
from quark.interface_types.quantum_result import SampleDistribution

logger = logging.getLogger(__name__)

_DEVICES = {
    "DigitalAnalogDevice": DigitalAnalogDevice,
    "AnalogDevice": AnalogDevice,
}


class PasqalQAASolver(Core):
    """Solve a QUBO via the Quantum Adiabatic Algorithm (QAA) on a neutral-atom register.

    The off-diagonal terms of the QUBO are encoded via the pairwise Rydberg interaction
    (C6 / r^6) between atoms, which requires them to be non-negative. The register that
    best reproduces these interactions is found via a numerical embedding, and an
    adiabatic (InterpolatedWaveform) pulse is run to prepare the ground state of the
    problem Hamiltonian.
    """

    def __init__(
        self,
        evolution_time: float = 4000,
        device: str = "DigitalAnalogDevice",
        backend: str = "simulation",
        shots: int = 1000,
    ):
        """Initialize the Pasqal QAA solver.

        :param evolution_time: Duration of the adiabatic pulse in ns. Default 4000.
        :param device: Pulser device to embed the register on, either "DigitalAnalogDevice"
            or "AnalogDevice". Default "DigitalAnalogDevice".
        :param backend: Execution backend, either "simulation" (pulser-simulation) or "cloud".
            Only "simulation" is supported in v1. Default "simulation".
        :param shots: Number of measurement shots to sample from the final state. Default 1000.
        """
        if device not in _DEVICES:
            msg = f"Unknown device {device!r}, expected one of {list(_DEVICES)}"
            raise ValueError(msg)

        self.evolution_time = evolution_time
        self.device_name = device
        self.backend = backend
        self.shots = shots

        self.embedding_error: float | None = None
        self.num_atoms: int | None = None
        self.runtime_s: float | None = None

    def preprocess(self, data: Qubo) -> Result:
        """Embed the QUBO onto a neutral-atom register and run the QAA.

        :param data: The QUBO to solve
        :return: Data(SampleDistribution) on success, Failed(...) on error
        """
        start_time = time.time()

        if self.backend != "simulation":
            return Failed(reason=f"Backend {self.backend!r} is not supported in v1; only 'simulation' is implemented")

        q_matrix = data.as_matrix()
        n = q_matrix.shape[0]

        off_diag_mask = ~np.eye(n, dtype=bool)
        if np.any(q_matrix[off_diag_mask] < 0):
            return Failed(
                reason=(
                    "Pasqal QAA embedding requires all off-diagonal QUBO terms to be non-negative "
                    "(the Rydberg interaction is purely repulsive). This QUBO contains negative cross-terms "
                    "and cannot be embedded directly."
                )
            )

        diagonal = np.diag(q_matrix)
        if n > 1 and not np.allclose(diagonal, diagonal[0]):
            logger.warning(
                "Non-uniform QUBO diagonal detected; PasqalQAASolver has no local addressability in v1 "
                "and will approximate the diagonal by its global average (%.4g).",
                diagonal.mean(),
            )
        avg_diag = float(diagonal.mean()) if n > 0 else 0.0

        device = _DEVICES[self.device_name]

        q_off_diag = q_matrix.copy()
        np.fill_diagonal(q_off_diag, 0.0)

        coords, embedding_error = self._embed_register(q_off_diag, device, n)
        self.embedding_error = embedding_error
        self.num_atoms = n

        qubits = {f"q{i}": coord for i, coord in enumerate(coords)}
        try:
            register = pulser.Register(qubits)
            sequence = pulser.Sequence(register, device)
        except Exception as e:
            return Failed(reason=f"Failed to build a valid register/sequence for this device: {e}")

        sequence.declare_channel("rydberg_global", "rydberg_global")

        positive_off_diag = q_off_diag[q_off_diag > 0]
        omega_max = float(np.median(positive_off_diag)) if positive_off_diag.size else 1.0
        delta_magnitude = abs(avg_diag) if avg_diag != 0 else 5.0
        delta_0 = -delta_magnitude
        delta_f = delta_magnitude

        pulse = pulser.Pulse(
            pulser.InterpolatedWaveform(self.evolution_time, [1e-9, omega_max, 1e-9]),
            pulser.InterpolatedWaveform(self.evolution_time, [delta_0, 0, delta_f]),
            0,
        )
        sequence.add(pulse, "rydberg_global")

        config = QutipConfig(observables=[BitStrings(evaluation_times=[1.0], num_shots=self.shots)])
        try:
            results = QutipBackendV2(sequence, config=config).run()
        except Exception as e:
            return Failed(reason=f"Simulation failed: {e}")

        counts = results.final_bitstrings
        sample_distribution = SampleDistribution.from_list(list(counts.items()), nbshots=self.shots)

        self.runtime_s = time.time() - start_time

        return Data(sample_distribution)

    def postprocess(self, data: Any) -> Result:
        """Pass through post-processing (no transformation needed)."""
        return Data(data)

    def get_metrics(self) -> dict:
        """Return metrics about the embedding and simulation."""
        return {
            "evolution_time_ns": self.evolution_time,
            "embedding_error": self.embedding_error,
            "num_atoms": self.num_atoms,
            "runtime_s": self.runtime_s,
        }

    def get_unique_name(self) -> str | None:
        """Return a unique identifier for this solver instance."""
        return f"pasqal_qaa_T{self.evolution_time:.0f}_shots{self.shots}"

    @staticmethod
    def _embed_register(q_off_diag: np.ndarray, device: Any, n: int) -> tuple[np.ndarray, float]:
        """Find 2D atom coordinates whose Rydberg interactions best approximate q_off_diag.

        :return: (coords of shape (n, 2), residual embedding error)
        """
        if n <= 1:
            return np.zeros((n, 2)), 0.0

        def evaluate_mapping(flat_coords: np.ndarray, q_off_diag: np.ndarray, device: Any) -> float:
            coords = flat_coords.reshape((n, 2))
            distances = pdist(coords)
            if np.any(distances == 0):
                return 1e10
            new_q = squareform(device.interaction_coeff / distances**6)
            return float(np.linalg.norm(new_q - q_off_diag))

        rng = np.random.default_rng(0)
        x0 = rng.random(n * 2)
        res = minimize(
            evaluate_mapping,
            x0,
            args=(q_off_diag, device),
            method="Nelder-Mead",
            tol=1e-6,
            options={"maxiter": 200000, "maxfev": None},
        )
        return res.x.reshape((n, 2)), float(res.fun)
