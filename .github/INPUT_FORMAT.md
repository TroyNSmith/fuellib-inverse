# FuelLib-Inverse Input File Format (`input.txt`)

[input.txt](input.txt) is a human-readable, draft configuration format for the
inverse-design solver. It is meant as a more approachable stand-in for the
[input.json](input.json) file that `SolverConfig.from_json`
(`src/fuellib_inverse/inverse/types.py`) actually parses today — nothing in the
codebase currently reads `input.txt` directly, so treat it as a spec for a
future parser rather than a live config file. Each `%section ... end` block
below maps onto one branch of the JSON constraint tree consumed by
`SolverConfig` and, ultimately, `inverse/core.py::solve`.

## Syntax

- Lines starting with `#` are comments; inline `# ...` comments are also allowed.
- Top-level `key = value` pairs configure the optimizer itself.
- `%section_name ... end` blocks each define one physical/composition
  constraint. Every constraint section shares the same general keywords:
  a `lambda` penalty weight, a target/reference value, and a tolerance
  (`rtol`/`atol`) defining how far the optimized mixture may deviate before
  being penalized.
- Some sections (`%density`, `%viscosity`) nest a `%target ... end` sub-block
  to specify multiple targets at different conditions (e.g. temperature).

## Optimization keywords

These top-level keys configure the optimizer and are not part of any
`%section`.

| Keyword | Description |
| --- | --- |
| `max_iter` | Maximum number of optimization steps. |
| `tolerance` | Convergence tolerance for stopping the optimization early. |
| `learning_rate` | Scaling factor for the size of each gradient-descent step. |
| `regularization_strength` | Multiplier controlling how harshly the optimizer penalizes large parameter values. |

## Input sections

| Section | Purpose | Has `%target` sub-block? |
| --- | --- | --- |
| `%gcxgc` | Match a reference GCxGC composition CSV, penalizing each matched compound's optimized weight % outside tolerance of the reference weight %. | No |
| `%aromatics_content` | Constrain the mixture's aromatic content to a target value or range. | No |
| `%density` | Constrain mixture density to target value(s)/range(s) at one or more temperatures. | Yes |
| `%viscosity` | Constrain mixture kinematic viscosity to target value(s)/range(s) at one or more temperatures. | Yes |

## `%gcxgc` keywords

Unlike the other sections, `%gcxgc` doesn't have a single scalar target —
instead every compound's weight % in the reference CSV becomes its own
target, each checked against the same `rtol`/`atol`.

| Keyword | Description |
| --- | --- |
| `lambda` | Weighting coefficient for adjusting the penalty. |
| `ref` | Path to the reference GCxGC data (must contain `Family`, `Weight %` columns). |
| `rtol` | Relative tolerance for each reference compound's weight % (`+/- rtol * weight %`). |
| `atol` | Absolute tolerance for each reference compound's weight % (`+/- atol`). |

## `%aromatics_content` keywords

| Keyword | Description | Default |
| --- | --- | --- |
| `units` | Units for the target value(s): `vol%` (volume %) or `frac` (fraction). | `vol%` |
| `lambda` | Weighting coefficient for adjusting the penalty. | `1.0` |
| `target` | Target value (`val`) or range (`[min, max]`) for the property. | — |
| `rtol` | Relative tolerance for a scalar target (`target = val +/- rtol * val`); ignored for ranges. | `0.05` |
| `atol` | Absolute tolerance for a scalar target (`target = val +/- atol`); ignored for ranges. | `0.00` |

## `%density` / `%viscosity` keywords

These two sections share the same keyword schema: a single-value property
constrained at one or more temperatures via a nested `%target` block.

| Keyword | Description | Default |
| --- | --- | --- |
| `units` | Units for the target values (e.g. `kg/m^3` for density, `mm^2/s` for viscosity). | `kg/m^3` (density), `mm^2/s` (viscosity) |
| `temp_units` | Units for the temperatures used as `%target` keys (`K`, `Kelvin`, `Celsius`, `Fahrenheit`). | `Celsius` |
| `lambda` | Weighting coefficient for adjusting the penalty. | `1.0` |
| `rtol` | Relative tolerance for a scalar target at a given temperature (`target = val +/- rtol * val`); ignored for ranges. | `0.05` |
| `atol` | Absolute tolerance for a scalar target at a given temperature (`target = val +/- atol`); ignored for ranges. | `0.00` |
| `%target ... end` | Sub-block mapping each temperature (in `temp_units`) to a target value or `[min, max]` range. | — |

### `%target` sub-block entries

| Key | Value | Description |
| --- | --- | --- |
| `<temperature>` | `val` | Scalar target value at that temperature; checked against `rtol`/`atol`. |
| `<temperature>` | `[min, max]` | Target range at that temperature; `rtol`/`atol` are ignored. |
