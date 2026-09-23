"""Small deterministic QUBO provider for local QUARK examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from quark.core import Core, Data, Result
from quark.interface_types import Qubo


@dataclass
class SimpleQuboProvider(Core):
    """Provide a small deterministic QUBO for local pipeline examples.

    The default matrix has two optimal bitstrings, ``01`` and ``10``. A
    caller can replace ``matrix`` to exercise the same provider interface
    with another small instance.
    """

    matrix: list[list[float]] = field(
        default_factory=lambda: [[-1.0, 1.0], [1.0, -1.0]]
    )

    def preprocess(self, data: Any) -> Result:
        """Return the configured matrix as a QUARK ``Qubo`` data object."""
        return Data(Qubo.from_matrix(np.asarray(self.matrix, dtype=np.float32)))

    def postprocess(self, data: Any) -> Result:
        """Wrap downstream data in QUARK's standard ``Data`` container."""
        return Data(data)