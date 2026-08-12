# quark-plugin-lp-to-qubo

A QUARK plugin for converting Linear Programs (LPs) to Quadratic Unconstrained Binary Optimization (QUBO) problems using [qiskit-optimization](https://qiskit.org/ecosystem/optimization/).

## Features

- **Binary Integer Program (BIP) Support**: Directly converts BIPs to QUBO
- **Continuous Variable Discretisation**: Automatically discretises continuous variables via binary expansion
- **Constraint Encoding**: Converts all constraints to equalities and encodes them as penalty terms in the QUBO objective
- **Configurable Precision**: Control binary discretisation precision and penalty strength via constructor parameters
- **QUARK Integration**: Seamlessly integrates into QUARK benchmarking pipelines

## Installation

```bash
pip install quark-plugin-lp-to-qubo
```

## Usage

### As a QUARK Plugin

Add to your `config.yml`:

```yaml
plugins: ["quark_plugin_lp_to_qubo"]

pipeline:
  - "lp_provider"  # Provides LP instances
  - "lp_to_qubo_converter":
      penalty_factor: 1e6
      continuous_var_precision: 8
  - "qubo_solver"  # Your QUBO solver (e.g., pasqal_qaa_solver)
```

### Programmatic Usage

```python
from quark_plugin_lp_to_qubo import LpToQuboConverter
from quark.interface_types.lp import LP

# Create a converter
converter = LpToQuboConverter(
    penalty_factor=1e6,
    continuous_var_precision=8
)

# Create an LP from a string
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

# Convert to QUBO
result = converter.preprocess(lp)
if isinstance(result, Data):
    qubo = result.data
    print(qubo.as_matrix())
```

## Parameters

- **`penalty_factor`** (float, default 1e6): Multiplier for constraint penalty terms. Larger values enforce constraint satisfaction more strictly.
- **`continuous_var_precision`** (int, default 8): Number of bits for binary discretisation of continuous variables. Each continuous variable becomes ~2^precision binary variables.
- **`discretisation_scale`** (float, default 1.0): Scale factor for discretised variable bounds.

## How It Works

1. **LP Parsing**: Reads the LP string using qiskit's `QuadraticProgram.read_from_lp_string()`
2. **Variable Detection**: Identifies continuous and binary/integer variables
3. **Discretisation**: Continuous variables are encoded as binary expansions:
   $$x \approx lb + \Delta \cdot \sum_{k=0}^{K} 2^k b_k$$
4. **Constraint Encoding**: All constraints are converted to equalities via slack variables and encoded as penalty terms:
   $$\text{QUBO} = f(x) + \lambda \cdot (Ax - b)^2$$
5. **QUBO Extraction**: Converts the constrained quadratic program to a pure QUBO via qiskit's `QuadraticProgramToQubo`

## Limitations

- **Non-negative variables only**: Variables must have non-negative bounds; unbounded variables cannot be discretised.
- **Discretisation overhead**: Continuous variables significantly increase QUBO size. Precision of 8 bits per variable typically adds ~256× more terms.
- **Penalty tuning**: The `penalty_factor` must be chosen carefully. Too small and constraints are violated; too large and the solver ignores the objective.

## Metrics

The converter returns the following metrics in `get_metrics()`:

- `runtime_s`: Conversion runtime in seconds
- `original_num_variables`: Number of variables in the original LP
- `discretised_num_variables`: Number of variables after discretisation
- `penalty_factor`: Penalty factor used for constraint encoding
- `continuous_var_precision`: Precision bits used for continuous variable discretisation

## References

- [Qiskit Optimization Documentation](https://qiskit.org/ecosystem/optimization/)
- [QUARK Framework](https://github.com/capgemini-engineering/QUARK)

## License

Apache 2.0
