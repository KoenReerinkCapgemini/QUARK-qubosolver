"""Basic tests for LpToQuboConverter."""

import numpy as np
import pytest

from quark.core import Data, Failed
from quark.interface_types.lp import LP
from quark_plugin_lp_to_qubo import LpToQuboConverter


class TestLpToQuboConverter:
    """Tests for LP to QUBO conversion."""

    def test_simple_bip_conversion(self):
        """Test conversion of a simple Binary Integer Program."""
        lp_str = """
Minimize
  obj: x + 2*y
Subject To
  c1: x + y <= 1
Bounds
  0 <= x <= 1
  0 <= y <= 1
Binaries
  x y
End
"""
        lp = LP.from_str(lp_str)
        converter = LpToQuboConverter(penalty_factor=1e6)
        result = converter.preprocess(lp)

        assert isinstance(result, Data), f"Expected Data, got {type(result)}"
        qubo = result.data
        matrix = qubo.as_matrix()
        assert matrix.shape[0] > 0, "QUBO matrix should not be empty"
        assert np.allclose(matrix, matrix.T), "QUBO matrix should be symmetric"

    def test_invalid_input(self):
        """Test handling of invalid input."""
        converter = LpToQuboConverter()
        result = converter.preprocess("not an LP")

        assert isinstance(result, Failed), f"Expected Failed, got {type(result)}"

    def test_metrics(self):
        """Test that metrics are properly recorded."""
        lp_str = """
Minimize
  x
Subject To
Bounds
  0 <= x <= 1
Binaries
  x
End
"""
        lp = LP.from_str(lp_str)
        converter = LpToQuboConverter()
        converter.preprocess(lp)
        metrics = converter.get_metrics()

        assert "runtime_s" in metrics
        assert "original_num_variables" in metrics
        assert metrics["original_num_variables"] == 1

    def test_unique_name(self):
        """Test unique name generation."""
        converter = LpToQuboConverter(penalty_factor=1e5, continuous_var_precision=10)
        name = converter.get_unique_name()

        assert name is not None
        assert "lp_to_qubo" in name
        assert "pen" in name
        assert "prec" in name
        assert "1e+05" in name or "1e+5" in name
        assert "10" in name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
