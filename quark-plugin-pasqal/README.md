# quark-plugin-pasqal

A QUARK plugin that wraps Pasqal's [Pulser](https://github.com/pasqal-io/Pulser) library to solve QUBO problems via the **Quantum Adiabatic Algorithm (QAA)** on neutral-atom hardware (or simulation).

## How it fits in the pipeline

```
tsp_graph_provider  →  tsp_qubo_mapping  →  pasqal_qaa_solver
         (Graph)              (Qubo)           (SampleDistribution)
```

The Pasqal solver is a leaf module: it receives a `Qubo`, physically embeds it onto a neutral-atom register, runs QAA, and returns a `SampleDistribution`.

## Constraints

- **All off-diagonal QUBO terms must be non-negative.** The Rydberg interaction used to embed the QUBO is purely repulsive, so QUBO instances with negative cross-terms cannot be directly embedded. The solver returns `Failed` with a descriptive message in this case.
- **Non-uniform diagonals** are approximated by their global average since per-atom local addressability is out of scope for v1; a warning is logged when this happens.
- Only the `pulser-simulation` backend is supported in v1; the Pasqal cloud backend is a stretch goal.

## Installation

```bash
pip install quark-plugin-pasqal
```

## Usage

Add to your `config.yml`:

```yaml
plugins: ["quark_plugin_pasqal"]

pipeline:
  - "tsp_graph_provider": { nodes: 5, seed: 42 }
  - "tsp_qubo_mapping_dnx"
  - "pasqal_qaa_solver":
      evolution_time: 4000
      device: "DigitalAnalogDevice"
      backend: "simulation"
      shots: 1000
```

### Programmatic Usage

```python
from quark_plugin_pasqal import PasqalQAASolver
from quark.interface_types.qubo import Qubo
import numpy as np

Q = np.array([
    [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
    [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
    [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
    [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
    [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
])

solver = PasqalQAASolver(evolution_time=4000, shots=1000)
result = solver.preprocess(Qubo.from_matrix(Q))
```
