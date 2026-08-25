# Plan: Pasqal QAA QUBO Solver Plugin for QUARK

**TL;DR**: Create a new QUARK plugin package `quark-plugin-pasqal` that wraps Pasqal's [Pulser](https://github.com/pasqal-io/Pulser) library. It solves QUBO problems via the **Quantum Adiabatic Algorithm (QAA)** on neutral-atom hardware (or simulation), slotting into the QUARK pipeline between an existing QUBO mapper and any downstream postprocessing.

---

## How it fits in the pipeline

```
tsp_graph_provider  →  tsp_qubo_mapping  →  pasqal_qaa_solver
         (Graph)              (Qubo)           (SampleDistribution)
```

The Pasqal solver is a leaf module: it receives a `Qubo`, physically embeds it onto a neutral-atom register, runs QAA, and returns a `SampleDistribution`.

---

## Steps

### Phase 1 — Package Scaffold *(independent, can do first)*
1. Create `quark-plugin-pasqal/` package directory with `pyproject.toml` declaring dependencies on `pulser`, `pulser-simulation`, `scipy`, and `quark`
2. Create `src/quark_plugin_pasqal/__init__.py` with a `register()` function that calls `factory.register("pasqal_qaa_solver", PasqalQAASolver)`

### Phase 2 — Solver Implementation *(depends on Phase 1)*

3. Create `src/quark_plugin_pasqal/solver.py` with `PasqalQAASolver(Core)`:
   - **Constructor params**: `evolution_time` (ns, default `4000`), `device` (`"DigitalAnalogDevice"` or `"AnalogDevice"`), `backend` (`"simulation"` or `"cloud"`), `shots` (default `1000`)
   - **`preprocess(data: Qubo) -> Result`**:
     - Extract `Q = data.as_matrix()`
     - Validate all off-diagonal entries are ≥ 0 (Rydberg interaction is repulsive only); return `Failed(reason=...)` if not
     - Run `scipy.optimize.minimize` to find 2D atom coordinates that best approximate the off-diagonal terms via the $C_6 / r^6$ Rydberg interaction law
     - Build a `pulser.Register` and `pulser.Sequence` with a `rydberg_global` channel
     - Add an **InterpolatedWaveform** adiabatic pulse: Ω ramps from 0→peak→0, δ sweeps from negative→positive (ground-state preparation → problem Hamiltonian)
     - Run `QutipBackendV2(sequence).run()`, collect `final_bitstrings` count dict
     - Convert to `SampleDistribution.from_list([(bitstr, count), ...], nbshots=shots)`
     - Return `Data(sample_distribution)`
   - **`postprocess(data: SampleDistribution) -> Result`**: pass-through, return `Data(data)`
   - **`get_metrics() -> dict`**: return `{"evolution_time_ns": ..., "embedding_error": ..., "num_atoms": ..., "runtime_s": ...}`
   - **`get_unique_name() -> str`**: e.g. `"pasqal_qaa_T4000_shots1000"`

### Phase 3 — Config Example *(parallel with Phase 2)*

4. Add `examples/pasqal_config.yml` to the QUARK-framework `examples/` dir showing a full TSP → QUBO → Pasqal pipeline

### Phase 4 — Verification

5. Smoke-test with the known 5×5 Pulser tutorial QUBO matrix (optimal solutions are `01011` and `00111`)
6. Run an end-to-end QUARK benchmark with a small (≤6-node) TSP to confirm JSON output contains a `SampleDistribution` with plausible top bitstrings

---

## Relevant files

- `src/quark/core.py` — `Core` ABC: `preprocess`, `postprocess`, `get_metrics`, `get_unique_name`
- `src/quark/interface_types/qubo.py` — `Qubo.as_matrix()` is the key input
- `src/quark/interface_types/quantum_result.py` — `SampleDistribution.from_list(samples, nbshots)` is the output
- `src/quark/plugin_manager/factory.py` — `factory.register("name", Class)`
- `src/quark/plugin_manager/loader.py` — calls `plugin.register()` on import
- `examples/config.yml` — reference YAML pipeline structure

---

## Decisions & Constraints

- **Fundamental constraint**: the Pasqal QAA embedding requires all off-diagonal QUBO terms to be **non-negative** (Rydberg interaction is purely repulsive). QUBO instances with negative cross-terms cannot be directly embedded — the solver returns `Failed` with a descriptive message in v1.
- **Non-uniform diagonal** (requiring per-atom local addressability) is out of scope for v1; if diagonals differ, a warning is logged and a global average is used as an approximation.
- **Real Pasqal cloud** (`pasqal-cloud` package) is a stretch goal; v1 only runs `pulser-simulation`.
- `qadence` (Pasqal's higher-level SDK) is *not* used — it's archived/unmaintained; `pulser` directly is the right choice.

---

## Further Considerations

1. **QUBO sign convention**: Many TSP QUBO mappers produce negative off-diagonal terms (penalties for invalid states). Should the plugin attempt an automatic sign-flip/normalization, or strictly fail-fast? Recommendation: fail-fast in v1, document the constraint clearly.
2. **Embedding quality**: The `scipy.minimize` atom-placement is best-effort; for large or dense QUBOs it may fail to find a valid register within device constraints. Should there be a retry mechanism or a fallback? Recommendation: single attempt, surface the `embedding_error` metric so users can assess quality.
