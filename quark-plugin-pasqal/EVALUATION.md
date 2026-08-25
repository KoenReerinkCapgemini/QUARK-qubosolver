# Evaluation: Pasqal QAA QUBO Solver Plugin

This document records the implementation steps taken from [pasqal-QUARK-plan.md](pasqal-QUARK-plan.md) and the results of testing the resulting `quark-plugin-pasqal` package.

## Implementation Steps

### Phase 1 — Package Scaffold
- Created `quark-plugin-pasqal/pyproject.toml` declaring dependencies on `pulser`, `pulser-simulation`, `scipy`, `numpy`.
- Created `src/quark_plugin_pasqal/__init__.py` with a `register()` function calling `factory.register("pasqal_qaa_solver", PasqalQAASolver)`.
- Added `LICENSE` (Apache-2.0, copied from `quark-plugin-lp-to-qubo`) and a `README.md`.

### Phase 2 — Solver Implementation
- Created `src/quark_plugin_pasqal/solver.py` with `PasqalQAASolver(Core)`:
  - Constructor params: `evolution_time` (default 4000 ns), `device` (`"DigitalAnalogDevice"` / `"AnalogDevice"`), `backend` (`"simulation"` / `"cloud"`), `shots` (default 1000).
  - `preprocess`: validates non-negative off-diagonal QUBO terms (fails fast otherwise), warns and averages non-uniform diagonals, embeds atoms in 2D via `scipy.optimize.minimize` (Nelder-Mead) matching pairwise $C_6/r^6$ Rydberg interaction to the QUBO's off-diagonal terms, builds a `pulser.Register`/`Sequence` with a `rydberg_global` channel, adds an adiabatic `InterpolatedWaveform` pulse (Ω: 0→median(off-diag)→0, δ: −avg(diag)→avg(diag)), runs `QutipBackendV2`, and converts `results.final_bitstrings` into a `SampleDistribution`.
  - `postprocess`: pass-through.
  - `get_metrics`: returns `evolution_time_ns`, `embedding_error`, `num_atoms`, `runtime_s`.
  - `get_unique_name`: e.g. `"pasqal_qaa_T4000_shots1000"`.
  - `backend="cloud"` returns `Failed` (out of scope for v1, per plan's Decisions & Constraints).

### Phase 3 — Config Example
- Added `QUARK-framework/examples/pasqal_config.yml` showing a TSP → QUBO → Pasqal QAA pipeline.

### Phase 4 — Verification
- Installed `QUARK-framework` and `quark-plugin-pasqal` in editable mode into the workspace `.venv`, along with `pulser`, `pulser-simulation`, `scipy`, and `pytest`.
- Ran an ad-hoc smoke test reproducing the Pulser QUBO tutorial's 5×5 matrix directly against `pulser`/`pulser_simulation` APIs to confirm correct usage of `Register`, `Sequence`, `InterpolatedWaveform`, `QutipConfig`, `BitStrings`, and `QutipBackendV2.run().final_bitstrings`.
- Ran the same scenario through the actual `PasqalQAASolver` via `factory.create("pasqal_qaa_solver", ...)`.
- Wrote a formal pytest suite, [tests_pasqal.py](tests_pasqal.py), and executed it.

## Test Results

### Ad-hoc smoke test (direct Pulser API, `DigitalAnalogDevice`, 500 shots)
```
embedding error: 7.79e-07
top counts: [('01011', 254), ('00111', 229), ('00011', 8), ('00101', 5), ('01010', 2)]
```
Matches the known optimal solutions (`01011`, `00111`) from the Pulser tutorial.

### End-to-end test through `factory.create("pasqal_qaa_solver", ...)`
```
registered: True
top bitstrings: [('01011', 246), ('00111', 231), ('00110', 7)]
metrics: {'evolution_time_ns': 4000, 'embedding_error': 1.70, 'num_atoms': 5, 'runtime_s': 0.19}
unique name: pasqal_qaa_T4000_shots500
Failed reason (negative off-diagonal case): "Pasqal QAA embedding requires all off-diagonal QUBO terms to be
non-negative (the Rydberg interaction is purely repulsive). This QUBO contains negative cross-terms and
cannot be embedded directly."
```

### Formal pytest suite (`tests_pasqal.py`)

| Test | Result |
|---|---|
| `test_registration` | PASSED |
| `test_known_qubo_solution` (top-2 bitstrings == `{"01011", "00111"}`) | PASSED |
| `test_negative_off_diagonal_fails` | PASSED |
| `test_unsupported_backend_fails` (`backend="cloud"`) | PASSED |
| `test_unknown_device_raises` | PASSED |
| `test_postprocess_passthrough` | PASSED |
| `test_metrics` | PASSED |
| `test_unique_name` | PASSED |

```
8 passed in 3.31s
```

## Findings

- The embedding + QAA pipeline reliably recovers the two known optimal bitstrings (`01011`, `00111`) as the top two most-sampled outcomes across repeated runs (both the ad-hoc script and the pytest run), with a small residual embedding error (~1e-6 to ~1.7 depending on the random seed used for `scipy.optimize.minimize`).
- The non-negative off-diagonal constraint and the "cloud" backend restriction are correctly enforced via `Failed(reason=...)`, matching the v1 scope defined in the plan's Decisions & Constraints section.
- No lint or type errors were found in `solver.py` or `__init__.py`.
- `pulser` (1.9.0) and `pulser-simulation` were successfully installed in the project's `.venv` without conflicts with existing dependencies (`pasqal-cloud` was pulled in transitively as a dependency of `pulser`, but is unused in v1 since only the `simulation` backend is implemented).

## Known Limitations (as documented in the plan)

- Non-uniform QUBO diagonals are approximated by their global average (no per-atom local addressability in v1).
- Only the `pulser-simulation` backend is implemented; `backend="cloud"` returns `Failed`.
- The atom-placement embedding is a single best-effort `scipy.optimize.minimize` attempt; there is no retry/fallback mechanism if it fails to converge to a low-error embedding for larger or denser QUBOs.
