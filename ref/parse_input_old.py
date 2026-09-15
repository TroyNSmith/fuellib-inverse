"""
Parser for the human-readable ``input.txt`` solver configuration format.

See `test/INPUT_FORMAT.md` for the format spec. `load` is the sole public entry
point: it reads and tokenizes an input.txt-format file and fully resolves it into
`SolverConfig`'s `(optimization_overrides, constraints)` shapes - constraint bounds
already `unxt.Quantity` objects, `%density`/`%viscosity` condition keys already
Kelvin-magnitude floats, and `%gcxgc`'s `ref` already resolved to a `Path` - ready
to hand directly to `SolverConfig` with no further resolution step.
"""

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from unxt import AbstractQuantity, Quantity

from ..utils.units import convert_temperature, strip

Block = dict[str, Any]

# Maps an input.txt top-level optimization key to its `_OptimizationParams` name
# and the type it should be cast to.
_OPTIMIZATION_KEYS: dict[str, tuple[str, type]] = {
    "max_iter": ("max_iterations", int),
    "tolerance": ("tolerance", float),
    "learning_rate": ("learning_rate", float),
    "regularization_strength": ("regularization_strength", float),
}


# --- Grammar parsing: input.txt text -> raw nested dict -------------------------
def _strip_comment(line: str) -> str:
    """Strip a trailing `# ...` comment and surrounding whitespace from `line`."""
    return line.split("#", 1)[0].strip()


def _parse_value(raw: str) -> float | list[float] | str:
    """
    Parse a `key = value` right-hand side into a float, a `[min, max]` list of
    floats, or (if neither parses) the raw string (e.g. a path or unit name).

    :param raw: Right-hand side text, already stripped of comments/whitespace.
    :type raw: str
    :return: The parsed value.
    :rtype: float | list[float] | str
    """
    if raw.startswith("["):
        return [float(v) for v in ast.literal_eval(raw)]
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return raw


def _parse_block(lines: Iterator[tuple[int, str]]) -> Block:
    """
    Recursively consume `lines` into a nested dict, stopping at a bare `end` line
    (or exhaustion of `lines`, for the top-level block). A `%name` line opens a
    nested sub-block (consumed recursively); all other lines must be `key = value`.

    :param lines: Shared iterator of `(line_number, line_text)` pairs (already
        comment-stripped and blank-filtered), advanced in place so sibling/nested
        calls consume disjoint spans of the file.
    :type lines: Iterator[tuple[int, str]]
    :return: The parsed block.
    :rtype: Block
    """
    block: Block = {}
    for lineno, line in lines:
        if line == "end":
            return block
        if line.startswith("%"):
            block[line[1:].strip()] = _parse_block(lines)
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Malformed input.txt line {lineno}: {line!r}")
        block[key.strip()] = _parse_value(value.strip())
    return block


def _parse(text: str) -> Block:
    """Parse `input.txt`-format text into a raw nested dict mirroring its structure."""
    lines = (
        (lineno, stripped)
        for lineno, raw_line in enumerate(text.splitlines(), start=1)
        if (stripped := _strip_comment(raw_line))
    )
    return _parse_block(lines)


# --- Semantic resolution: raw dict -> resolved (optimization, constraints) ------


def _quantity(value: float, units: str) -> AbstractQuantity:
    """
    Convert a scalar value with the given units to a `Quantity`, converting
    temperatures to Kelvin.
    """
    quantity = Quantity(jnp.asarray(float(value)), units)
    if str(quantity.unit.physical_type).lower() == "temperature":
        return convert_temperature(quantity, "K")
    return quantity


def _bounds(
    value: float | list[float], rtol: float, atol: float, units: str
) -> tuple[AbstractQuantity, AbstractQuantity]:
    """
    Resolve a `target` (scalar or `[min, max]` range) into a `(min, max)` pair of
    `Quantity`s in `units`. A scalar target is widened by `atol + rtol*|target|`
    on each side; a range is used as-is (`rtol`/`atol` are ignored).
    """
    if isinstance(value, list):
        lo, hi = value
    else:
        half_width = atol + rtol * abs(value)
        lo, hi = value - half_width, value + half_width
    return _quantity(lo, units), _quantity(hi, units)


def _build_gcxgc(node: Block, base_dir: Path) -> Block:
    """Build a `_GCxGCLeaf`-shaped dict from a parsed `%gcxgc` block."""
    return {
        "lambda_": float(node["lambda"]),
        "rtol": float(node["rtol"]),
        "atol": float(node["atol"]),
        "csv_path": base_dir / node["ref"],
    }


def _build_aromatics(node: Block) -> Block:
    """Build a `_MinMax`-shaped dict from a parsed `%aromatics_content` block."""
    units = node.get("units", "vol%")
    rtol = float(node.get("rtol", 0.05))
    atol = float(node.get("atol", 0.0))
    target = node["target"]
    if units == "frac":
        # Normalize fraction targets to percent, since mixture aromatics content
        # is always reported/compared in percent (see `mixture.aromatics_content`).
        target = (
            [v * 100.0 for v in target] if isinstance(target, list) else target * 100.0
        )
        atol *= 100.0
    lo, hi = _bounds(target, rtol, atol, "dimensionless")
    return {"lambda_": float(node["lambda"]), "min": lo, "max": hi}


def _build_temperature_property(node: Block) -> dict[float, Any]:
    """
    Build a `{<Kelvin magnitude>: _MinMax, ...}` dict from a parsed
    `%density`/`%viscosity` block, one `_MinMax` per entry in its `%target`
    sub-block.
    """
    units = node["units"]
    temp_units = node.get("temp_units", "Celsius")
    rtol = float(node.get("rtol", 0.05))
    atol = float(node.get("atol", 0.0))
    lam = float(node["lambda"])

    result: dict[float, Any] = {}
    for temp, value in node["target"].items():
        kelvin = float(strip(_quantity(float(temp), temp_units)))
        lo, hi = _bounds(value, rtol, atol, units)
        result[kelvin] = {"lambda_": lam, "min": lo, "max": hi}
    return result


def load(input_path: str | Path) -> tuple[dict[str, Any], dict[str | float, Any]]:
    """
    Read, parse, and fully resolve an input.txt-format file.

    Top-level `max_iter`/`tolerance`/`learning_rate`/`regularization_strength` keys
    become optimization overrides; each `%section ... end` block (`%gcxgc`,
    `%aromatics_content`, `%density`, `%viscosity`) is mapped onto its
    corresponding branch of the constraint tree - `composition.gcxgc`,
    `composition.aromatics.volume_percent`, `volatility.density`,
    `fluidity.viscosity` respectively. Relative paths (e.g. `%gcxgc`'s `ref`) are
    resolved against `input_path`'s parent directory.

    :param input_path: Path to the input.txt-format file.
    :type input_path: str | Path
    :return: Optimization overrides (to merge over `_default_optimization()`) and
        a fully resolved constraint tree, ready to hand to `SolverConfig`.
    :rtype: tuple[dict[str, Any], dict[str | float, Any]]
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = _parse(input_path.read_text())
    base_dir = input_path.parent

    optimization: dict[str, Any] = {}
    constraints: dict[str | float, Any] = {}

    for key, value in raw.items():
        if key in _OPTIMIZATION_KEYS:
            name, cast = _OPTIMIZATION_KEYS[key]
            optimization[name] = cast(value)
        elif key == "gcxgc":
            constraints.setdefault("composition", {})["gcxgc"] = _build_gcxgc(
                value, base_dir
            )
        elif key == "aromatics_content":
            constraints.setdefault("composition", {})["aromatics"] = {
                "volume_percent": _build_aromatics(value)
            }
        elif key == "density":
            constraints.setdefault("volatility", {})["density"] = (
                _build_temperature_property(value)
            )
        elif key == "viscosity":
            constraints.setdefault("fluidity", {})["viscosity"] = (
                _build_temperature_property(value)
            )
        else:
            raise ValueError(f"Unrecognized top-level input.txt key: {key!r}")

    return optimization, constraints
