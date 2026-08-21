# quark-plugin-pasqal

A QUARK plugin that solves QUARK `Qubo` instances with the Pasqal `qubo-solver` package.

## Pipeline

```text
Qubo provider -> qubo_to_pasqal -> SampleDistribution
```

## Configuration

```yaml
plugins:
  - "quark_plugin_pasqal"

pipeline:
  - "qubo_provider"
  - "qubo_to_pasqal":
      validate_input: true
```

The optional `solver_config` argument can be supplied programmatically as a `qubo-solver` `SolverConfig`. Cloud credentials should be configured through the Pasqal client environment, never committed to source control.

## Supported input

The plugin accepts a symmetric QUARK `Qubo`. In quantum mode, the underlying solver currently supports at most 80 variables and rejects negative off-diagonal coefficients. Invalid inputs return QUARK `Failed` results.

## Programmatic use

```python
import numpy as np
from quark.core import Data
from quark.interface_types import Qubo
from quark_plugin_pasqal import QuboToPasqal

qubo = Qubo.from_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
solver = QuboToPasqal()
result = solver.preprocess(qubo)

if isinstance(result, Data):
    samples = solver.postprocess(result)
    print(samples)
```

The first local test should use the default local emulator. Test cloud execution separately because it requires credentials, device access, and network connectivity.

## Run an end-to-end QUARK example

From the repository root, first install the local plugin into the active environment:

```powershell
uv pip install -e .\\quark-plugin-pasqal
```

Then run the deterministic two-variable example:

```powershell
uv run python -m quark -c quark-plugin-pasqal/examples/small_quantum.yml
```

The provider creates the QUBO

```text
[[-1, 1],
 [ 1,-1]]
```

Its best bitstrings are `01` and `10`, both with cost `-1`. QUARK writes the run results to a results directory and records the Pasqal runtime, QUBO size, sample count, and best cost.

To compare the local quantum path with the solver's classical path, use:

```powershell
uv run python -m quark -c quark-plugin-pasqal/examples/small_compare.yml
```

The comparison config creates two pipelines from the same provider: one with `use_quantum: true` and one with `use_quantum: false`. This is a local emulator versus classical-solver comparison, not a claim about hardware performance. It does not require Pasqal cloud credentials.
