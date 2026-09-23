# QUARK-qubosolver
QUARK plugin for the Pasqal QUBO solver.

## Run

From the repository root:

```powershell
uv sync
uv pip install -e .\quark-plugin-pasqal
uv run python -m quark -c quark-plugin-pasqal/examples/small_quantum.yml
```

This runs the local Pasqal emulator. To compare quantum and classical modes:

```powershell
uv run python -m quark -c quark-plugin-pasqal/examples/small_compare.yml
```

Run tests with:

```powershell
uv run pytest quark-plugin-pasqal/tests -q
```
