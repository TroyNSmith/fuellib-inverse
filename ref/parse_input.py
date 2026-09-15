"""Input file parsing for the inverse module."""

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from ..utils.units import convert_temperature, strip
from . import mixture
from .types import CriticalProperties

OPTIMIZATION_KEYS: dict[str, tuple[str, type]] = {
    "max_iter": ("max_iterations", int),
    "tolerance": ("tolerance", float),
    "learning_rate": ("learning_rate", float),
    "regularization_strength": ("regularization_strength", float),
}

Block = dict[str, Any]
Bounds = tuple[AbstractQuantity, AbstractQuantity]


def square_hinge_penalty(
    value: AbstractQuantity, low_bound: AbstractQuantity, high_bound: AbstractQuantity
) -> Array:
    """
    Squared-hinge penalty for a value constrained to ``[low_bound, high_bound]``.
    Zero when the value is within bounds.
    """
    value = value.to(low_bound.unit)

    zero = Quantity(0.0, low_bound.unit)
    below = qnp.maximum(low_bound - value, zero)
    above = qnp.maximum(value - high_bound, zero)

    return strip(below**2 + above**2)


class Constraint(ABC):
    """A single constraint in the inverse problem."""

    name: str
    lambda_: float
    bounds: tuple[AbstractQuantity, AbstractQuantity]

    @abstractmethod
    def loss(self, Y: Array, props: CriticalProperties) -> Array:
        """Compute the loss for a given weight vector."""


class AromaticsConstraint(Constraint):
    """Constraint on the aromatic content of a fuel mixture."""

    def __init__(
        self,
        name: str,
        lambda_: float,
        bounds: tuple[AbstractQuantity, AbstractQuantity],
    ):
        self.name = name
        self.lambda_ = lambda_
        self.bounds = bounds

    def loss(self, Y: Array, props: CriticalProperties) -> Array:
        """Compute the loss for a given weight vector."""
        prediction = Quantity(mixture.aromatics_content(Y, props), "dimensionless")
        return self.lambda_ * square_hinge_penalty(prediction, *self.bounds)


def _strip_comment(line: str) -> str:
    """Strip a trailing `# ...` comment and surrounding whitespace from `line`."""
    return line.split("#", 1)[0].strip()


def _parse_value(raw: str) -> float | list[float] | str:
    """
    Parse a `key = value` right-hand side into a float, a `[min, max]` list of
    floats, or the raw string (e.g. a path or unit name).
    """
    if raw.startswith("["):
        return [float(v) for v in ast.literal_eval(raw)]

    try:
        return float(raw.replace(",", ""))

    except ValueError:
        return raw


def _parse_block(lines: Iterator[tuple[int, str]]) -> Block:
    """
    Consume `lines` into a nested dict, stopping at an `end` line (or exhaustion
    of `lines` for the top-level block). A `%name` line opens a nested sub-block;
    all other lines must be `key = value`.
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
    """Parse `input.txt` into a nested dict mirroring its structure."""
    lines = (
        (lineno, stripped)
        for lineno, raw_line in enumerate(text.splitlines(), start=1)
        if (stripped := _strip_comment(raw_line))
    )
    return _parse_block(lines)


def _quantity(value: float, units: str) -> AbstractQuantity:
    """Convert a value with the given units to a `Quantity`."""
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


def _build_aromatics(block: Block) -> Constraint:
    """Build a constraint from a parsed `%aromatics_content` block."""
    units = block.get("units", "vol%")
    rtol = float(block.get("rtol", 0.05))
    atol = float(block.get("atol", 0.0))
    target = block["target"]
    if units == "frac":
        # Normalize fraction targets to percent
        target = (
            [v * 100.0 for v in target] if isinstance(target, list) else target * 100.0
        )
        atol *= 100.0

    lo, hi = _bounds(target, rtol, atol, "dimensionless")

    return AromaticsConstraint(
        name="aromatics_content", lambda_=float(block["lambda"]), bounds=(lo, hi)
    )


def load(input_path: str | Path) -> tuple[dict[str, Any], list[Constraint]]:
    """Read and parsean input.txt-format file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = _parse(input_path.read_text())
    base_dir = input_path.parent

    optimization: dict[str, Any] = {}
    constraints: list[Constraint] = []

    for key, value in raw.items():
        if key in OPTIMIZATION_KEYS:
            name, cast = OPTIMIZATION_KEYS[key]
            optimization[name] = cast(value)

        elif key == "aromatics_content":
            constraints.append(_build_aromatics(value))

    return optimization, constraints
