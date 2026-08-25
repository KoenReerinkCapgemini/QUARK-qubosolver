"""Basic tests for PasqalQAASolver."""

import numpy as np
import pytest

from quark.core import Data, Failed
from quark.interface_types.qubo import Qubo
from quark_plugin_pasqal import PasqalQAASolver, register

# The classic Pulser QAA tutorial QUBO: optimal solutions are "01011" and "00111".
KNOWN_QUBO = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)


class TestPasqalQAASolver:
    """Tests for the Pasqal QAA QUBO solver."""

    def test_registration(self):
        """Test that the plugin registers itself with the factory."""
        from quark.plugin_manager import factory

        register()
        assert "pasqal_qaa_solver" in factory.plugin_creation_funcs

    def test_known_qubo_solution(self):
        """Test that the solver recovers the known optimal bitstrings."""
        solver = PasqalQAASolver(evolution_time=4000, shots=500)
        result = solver.preprocess(Qubo.from_matrix(KNOWN_QUBO))

        assert isinstance(result, Data), f"Expected Data, got {type(result)}"
        samples = result.data.as_list()
        top = sorted(samples, key=lambda item: -item[1])[:2]
        top_bitstrings = {bitstr for bitstr, _ in top}
        assert top_bitstrings == {"01011", "00111"}

    def test_negative_off_diagonal_fails(self):
        """Test that a QUBO with negative cross-terms is rejected."""
        q_bad = KNOWN_QUBO.copy()
        q_bad[0, 1] = q_bad[1, 0] = -5.0

        solver = PasqalQAASolver()
        result = solver.preprocess(Qubo.from_matrix(q_bad))

        assert isinstance(result, Failed), f"Expected Failed, got {type(result)}"
        assert "non-negative" in result.reason

    def test_unsupported_backend_fails(self):
        """Test that the unsupported 'cloud' backend is rejected in v1."""
        solver = PasqalQAASolver(backend="cloud")
        result = solver.preprocess(Qubo.from_matrix(KNOWN_QUBO))

        assert isinstance(result, Failed), f"Expected Failed, got {type(result)}"
        assert "cloud" in result.reason

    def test_unknown_device_raises(self):
        """Test that an unknown device name raises at construction time."""
        with pytest.raises(ValueError, match="Unknown device"):
            PasqalQAASolver(device="NotARealDevice")

    def test_postprocess_passthrough(self):
        """Test that postprocess passes data through unchanged."""
        solver = PasqalQAASolver()
        sentinel = object()
        result = solver.postprocess(sentinel)

        assert isinstance(result, Data)
        assert result.data is sentinel

    def test_metrics(self):
        """Test that metrics are properly recorded after preprocessing."""
        solver = PasqalQAASolver(evolution_time=4000, shots=500)
        solver.preprocess(Qubo.from_matrix(KNOWN_QUBO))
        metrics = solver.get_metrics()

        assert metrics["evolution_time_ns"] == 4000
        assert metrics["num_atoms"] == 5
        assert metrics["embedding_error"] is not None
        assert metrics["runtime_s"] is not None

    def test_unique_name(self):
        """Test unique name generation."""
        solver = PasqalQAASolver(evolution_time=4000, shots=1000)
        name = solver.get_unique_name()

        assert name == "pasqal_qaa_T4000_shots1000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
