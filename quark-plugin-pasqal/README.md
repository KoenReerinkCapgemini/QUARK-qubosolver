# quark-plugin-pasqal

QUARK plugin for solving QUBO problems with Pasqal's [`qubo-solver`](https://github.com/pasqal-io/qubo-solver) package.

The plugin provides three pipeline modules:

- `lp_to_qubo_converter` converts binary LP files or QUARK `LP` objects into `Qubo` objects with Qiskit's `QuadraticProgramToQubo` converter.
- `qubo_to_pasqal` solves a QUARK `Qubo` and returns a `SampleDistribution`.
- `simple_qubo_provider` supplies a small deterministic QUBO for local examples and tests.

## Installation

From the repository root, install the framework and this plugin into the active `uv` environment:

```powershell
uv sync
uv pip install -e .\quark-plugin-pasqal
```

For a published package, install it with:

```bash
pip install quark-plugin-pasqal
```

## QUBO pipeline

Pass a QUBO provider into `qubo_to_pasqal` in a QUARK configuration:

```yaml
plugins:
  - "quark_plugin_pasqal"

pipeline:
  - "simple_qubo_provider"
  - "qubo_to_pasqal":
      use_quantum: true
      validate_input: true
```

`use_quantum: true` uses Pasqal's local quantum emulator by default. Set it to `false` to use the classical solver path. A custom `qubo-solver` `SolverConfig` can also be passed when constructing `QuboToPasqal` programmatically.

The solver records `runtime_s`, `qubo_size`, `num_samples`, `best_cost`, `solver_mode`, and `best_bitstrings` as QUARK metrics.

### Programmatic use

```python
import numpy as np
from quark.core import Data
from quark.interface_types import Qubo
from quark_plugin_pasqal import QuboToPasqal

qubo = Qubo.from_matrix(np.array([
    [-1.0, 1.0],
    [1.0, -1.0],
]))
solver = QuboToPasqal(use_quantum=True)
result = solver.preprocess(qubo)

if isinstance(result, Data):
    samples = solver.postprocess(result)
    print(samples)
```

## LP to QUBO conversion

The converter accepts LP problems only when every variable is binary. Continuous and integer variables are rejected; this plugin does not discretize non-binary variables. Constraint penalties are controlled by `penalty_factor` and default to `1e6`.

An LP-to-QUBO pipeline can be configured as follows:

```yaml
plugins:
  - "quark_plugin_pasqal"

pipeline:
  - "lp_to_qubo_converter":
      path_to_lp: "quark-plugin-pasqal/examples/example_1.lp"
      penalty_factor: 1000000
  - "qubo_to_pasqal":
      use_quantum: true
```

The same conversion is available in Python:

```python
from quark.interface_types.lp import LP
from quark_plugin_pasqal import LpToQuboConverter

lp = LP.from_file("quark-plugin-pasqal/examples/example_1.lp")
result = LpToQuboConverter(penalty_factor=1).preprocess(lp)
```

## Supported input and limitations

- Input QUBOs must be square, symmetric, and contain finite values when validation is enabled.
- Pasqal's quantum solver currently supports at most 80 variables.
- Quantum mode rejects negative off-diagonal coefficients.
- Invalid input and solver failures are returned as QUARK `Failed` results.
- Cloud execution requires Pasqal credentials and device access; local examples use the emulator and do not require cloud credentials.

## Examples and tests

Run a small quantum example from the repository root:

```powershell
uv run python -m quark -c quark-plugin-pasqal/examples/small_quantum.yml
```

Compare the quantum-emulator and classical solver paths:

```powershell
uv run python -m quark -c quark-plugin-pasqal/examples/small_compare.yml
```

Run the LP conversion example directly:

```powershell
uv run python quark-plugin-pasqal/examples/run_lp_to_qubo.py
```

Run the plugin test suite:

```powershell
uv run pytest quark-plugin-pasqal/tests -q
```
