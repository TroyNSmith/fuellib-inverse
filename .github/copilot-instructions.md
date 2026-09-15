# fuellib-inverse

Sandbox for the inverse-design module of the `FuelLib` package: given target mixture
properties (density, viscosity, aromatics %, ...), solve for component mass fractions.

## Build and Test

Managed with `uv`. Prefer the `taskipy` tasks over raw tool invocations:

- `uv run task format` — ruff format
- `uv run task lint` — ruff check --fix
- `uv run task types` — `ty check .`
- `uv run task imports` — `lint-imports` (import-linter; enforces module layering, see below)
- `uv run task test` — pytest (no `test_*.py` files exist yet — only manual scripts
  `test/scratch.py` and `test/fuel_to_csv.py`)
- Run scripts as `uv run python -B <script>` (the `-B` avoids stale `__pycache__`
  bytecode silently masking edits in this `uv`-managed venv).

## Architecture

- `src/fuellib_inverse/fuel.py` — `Fuel` class: loads GCxGC composition + group-contribution
  decomposition for a named fuel (`data/fuelData/*.gcxgc.csv`, `*.gani.csv`), exposes
  critical properties and mixture correlations (density, kinematic viscosity, ...) as
  `unxt.Quantity`.
- `src/fuellib_inverse/gcm/` — Gani group-contribution method (`gani.py`) for estimating
  critical properties from a mixture's group decomposition.
- `src/fuellib_inverse/rd/` — RDKit molecule helpers (SMILES parsing, formula/element counts).
- `src/fuellib_inverse/inverse/` — inverse-design solver: `types.py` defines
  `CriticalProperties` (per-compound properties parsed from a CSV) and `SolverConfig`
  (constraint tree + optimizer params, built by `SolverConfig.from_input` from a
  human-readable `input.txt`-format file - see `test/INPUT_FORMAT.md` and
  `parse_input.py`); `components.py`/`mixture.py` compute per-component and
  mixture-level correlations; `core.py::solve` optimizes mass fractions (via `optax` +
  `jax.grad`) to match the configured constraints.
- `src/fuellib_inverse/utils/` — shared helpers: `units.py` (unxt/astropy unit registration
  and conversions), `element/` (periodic table lookups).
- Import-linter (`[tool.importlinter]` in pyproject.toml) enforces that `fuel` may not
  import from `gcm`, except `gcm.gani` calling back into `fuel` for its GCM inputs.

## Conventions

- **Unit tracking**: physical quantities (MW, Tc, Vm_stp, omega, density, viscosity, etc.)
  are real `unxt.Quantity`/`AbstractQuantity` objects, manipulated with
  `quaxed.numpy as qnp` — not bare `jax.numpy`. Fractions (mass/mole/volume fractions,
  optimizer logits `Z`) stay bare `jax.Array` (dimensionless), since they must flow
  through `jax.nn.softmax`/`optax`.
- Temperature conversions must go through `utils/units.py::convert_temperature` (needs
  astropy equivalencies for K↔Celsius↔Fahrenheit offset conversions); all other unit
  conversions use `Quantity.to(unit)` directly.
- To `jax.jit`-compile code that builds/operates on `Quantity` objects internally, compose
  `jax.jit(quax.quaxify(func))` — a plain `@jax.jit` won't correctly trace through
  Quantity arithmetic (see `inverse/core.py::solve`'s `step`). Loss/objective values must
  still be stripped to bare scalars (`utils/units.py::strip`) before being summed across
  constraint categories that have different physical units.
- `ty check` requires `.value` results passed into `qnp` functions to be coerced via
  `qnp.array(...)`/`jnp.asarray(...)` first — raw `.value` has a `StaticValue | Array`
  union type that fails plain-array overloads (e.g. `qnp.log`).
- Docstrings consistently use Sphinx/RST-style field lists (`:param:`, `:type:`,
  `:return:`, `:rtype:`), not the numpy `Parameters`/`Returns` header style — match this
  even though `[tool.ruff.lint.pydocstyle]` sets `convention = "numpy"`.
