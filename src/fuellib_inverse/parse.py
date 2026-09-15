"""Parsing utilities for FuelLib Inverse."""

from pathlib import Path
from typing import Any, Literal

import optax
import pyparsing as pp
from fuellib import fuel
from pydantic import BaseModel, ConfigDict, Field

from .constraints import (
    AromaticsConstraint,
    Constraint,
    DensityConstraint,
    ViscosityConstraint,
)

SIMPLE_NUMERICAL_KEYS = [
    "max_iter",
    "tolerance",
    "learning_rate",
    "regularization_strength",
]
SIMPLE_STRING_KEYS = [
    "name",
    "optimizer",
]

BLOCKS = [
    "gcxgc",
    "aromatics_content",
    "density",
    "viscosity",
]
BLOCK_NUMERICAL_KEYS = [
    "lambda",
    "atol",
    "rtol",
    "target",
]
BLOCK_STRING_KEYS = [
    "units",
    "temp_units",
    "ref",
]


def _to_number(tokens: pp.ParseResults) -> int | float:
    """Convert a matched numeric token to a number."""
    raw = tokens[0].replace(",", "")
    return int(raw) if raw.lstrip("+-").isdigit() else float(raw)


def _unwrap(value: Any) -> Any:
    """Unwrap a value that pyparsing nested in a ``ParseResults``."""
    return value[0] if isinstance(value, pp.ParseResults) else value


# Token definitions for parsing
num_re = r"[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?"

simple_numerical_key = pp.one_of(SIMPLE_NUMERICAL_KEYS, as_keyword=True)("key")
simple_numerical_value = pp.Regex(num_re).set_parse_action(_to_number)("value")
simple_string_key = pp.one_of(SIMPLE_STRING_KEYS, as_keyword=True)("key")
simple_string_value = pp.Word(pp.alphanums + "_-.")("value")

block_start = pp.Literal("%").suppress() + pp.one_of(BLOCKS)("name")
block_end = pp.Literal("end").suppress()

block_numerical_key = pp.one_of(BLOCK_NUMERICAL_KEYS, as_keyword=True)("key")
block_numerical_scalar = pp.Regex(num_re).set_parse_action(_to_number)
block_numerical_range = (
    pp.Suppress("[")
    + block_numerical_scalar
    + pp.Suppress(",")
    + block_numerical_scalar
    + pp.Suppress("]")
).set_parse_action(lambda tokens: [tuple(tokens)])
block_numerical_value = (block_numerical_range | block_numerical_scalar)("value")

block_string_key = pp.one_of(BLOCK_STRING_KEYS, as_keyword=True)("key")
block_string_value = pp.Word(pp.alphanums + "%^/.")("value")

block_target_start = pp.Literal("%target").suppress()
block_target_end = pp.Literal("end").suppress()
block_target_content = pp.SkipTo(block_target_end)("content")

block_target_key = pp.Regex(num_re).set_parse_action(_to_number)("temperature")
block_target_entry = (
    block_target_key + pp.Suppress("=") + block_numerical_value
).ignore(pp.python_style_comment)

# A block's content must skip over any nested `%target ... end` sub-block
nested_target_block = block_target_start + block_target_content + block_target_end
block_content = pp.SkipTo(block_end, ignore=nested_target_block)("content")


def simple_numerical_keys(text: str) -> dict[str, int | float | None]:
    """Parse the simple ``key = value`` optimization parameters out of an input file."""
    # Token definitions for parsing simple key-value pairs.
    assignment = (
        simple_numerical_key + pp.Suppress("=") + simple_numerical_value
    ).ignore(pp.python_style_comment)

    # Build the parsed dictionary from the scanned matches.
    parsed = {key: None for key in SIMPLE_NUMERICAL_KEYS}
    for match, _start, _end in assignment.scan_string(text):
        parsed[match["key"]] = match["value"]
    return parsed


def simple_string_keys(text: str) -> dict[str, str | None]:
    """Parse the simple string key-value pairs out of an input file."""
    assignment = (simple_string_key + pp.Suppress("=") + simple_string_value).ignore(
        pp.python_style_comment
    )

    parsed = {key: None for key in SIMPLE_STRING_KEYS}
    for match, _start, _end in assignment.scan_string(text):
        parsed[match["key"]] = match["value"]
    return parsed


def block_numerical_keys(
    text: str,
) -> dict[str, int | float | tuple[int | float, int | float] | None]:
    """Parse the numerical keys within `%name ... end` blocks out of an input file."""
    assignment = (
        block_numerical_key + pp.Suppress("=") + block_numerical_value
    ).ignore(pp.python_style_comment)

    parsed = {key: None for key in BLOCK_NUMERICAL_KEYS}
    for match, _start, _end in assignment.scan_string(text):
        parsed[match["key"]] = _unwrap(match["value"])
    return parsed


def block_string_keys(text: str) -> dict[str, str | None]:
    """Parse the string keys within `%name ... end` blocks out of an input file."""
    assignment = (block_string_key + pp.Suppress("=") + block_string_value).ignore(
        pp.python_style_comment
    )

    parsed = {key: None for key in BLOCK_STRING_KEYS}
    for match, _start, _end in assignment.scan_string(text):
        parsed[match["key"]] = match["value"]
    return parsed


def block_target(
    text: str,
) -> dict[str, dict[int | float, int | float | tuple[int | float, int | float]]]:
    """Parse the `%target ... end` sub-block within `%name ... end` blocks out of an input file."""
    block = block_target_start + block_target_content + block_target_end

    for match, _start, _end in block.scan_string(text):
        targets = {}
        for entry, _s, _e in block_target_entry.scan_string(match["content"]):
            targets[_unwrap(entry["temperature"])] = _unwrap(entry["value"])
        return {"target": targets}

    return {}


def block_keys(
    text: str,
) -> dict[str, Any]:
    """Parse the `%name ... end` blocks out of an input file."""
    block = block_start + block_content + block_end

    blocks = {}
    for match, _start, _end in block.scan_string(text):
        blocks[match["name"]] = (
            block_numerical_keys(match["content"])
            | block_string_keys(match["content"])
            | block_target(match["content"])
        )

    return blocks


def input_file(file_path: str | Path) -> dict:
    """Load and parse the input file for FuelLib Inverse.

    :param file_path: Path to the input file.
    :return: Parsed input data.
    """
    text = Path(file_path).read_text()
    return {
        **simple_numerical_keys(text),
        **simple_string_keys(text),
        **block_keys(text),
    }


class OptimizationParameters(BaseModel):
    """Constraints for inverse optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_iter: int = 100_000
    tolerance: float = 1e-06
    learning_rate: float = 0.01
    regularization_strength: float = 0.1
    optimizer_method: Literal["adam"] = "adam"

    num_compounds: int
    constraints: list[Constraint] = Field(default_factory=list)

    @classmethod
    def from_txt(cls, file_path: str | Path) -> "OptimizationParameters":
        """Load constraints from a text file."""
        inp_dict = input_file(file_path)

        # Simple keywords
        name = inp_dict.get("name")
        if not name:
            raise ValueError("The 'name' field is required.")
        max_iter = inp_dict.get("max_iter", 100_000)
        tolerance = inp_dict.get("tolerance", 1e-06)
        learning_rate = inp_dict.get("learning_rate", 0.01)
        regularization_strength = inp_dict.get("regularization_strength", 0.1)
        optimizer_method = inp_dict.get("optimizer", "adam")

        fuel_ = fuel(name=name)

        # Blocks
        constraints = []
        ## %aromatics_content
        aromatics_dict = inp_dict.get("aromatics_content")
        aromatics_constraint = (
            AromaticsConstraint.from_input("aromatics_content", aromatics_dict, fuel_)
            if aromatics_dict
            else None
        )
        if aromatics_constraint:
            constraints.append(aromatics_constraint)

        ## %density
        density_dict = inp_dict.get("density", None)
        density_constraints = (
            DensityConstraint.from_input("density", density_dict, fuel_)
            if density_dict
            else None
        )
        if density_constraints:
            constraints.extend(density_constraints)

        ## %viscosity
        viscosity_dict = inp_dict.get("viscosity", None)
        viscosity_constraints = (
            ViscosityConstraint.from_input("viscosity", viscosity_dict, fuel_)
            if viscosity_dict
            else None
        )
        if viscosity_constraints:
            constraints.extend(viscosity_constraints)

        pars = {
            "name": name,
            "max_iter": max_iter,
            "tolerance": tolerance,
            "learning_rate": learning_rate,
            "regularization_strength": regularization_strength,
            "optimizer_method": optimizer_method,
            "num_compounds": fuel_.num_compounds,
            "constraints": constraints,
        }

        return cls.model_validate(pars)

    @property
    def optimizer(self) -> optax.GradientTransformation:
        if self.optimizer_method == "adam":
            return optax.adam(learning_rate=self.learning_rate)

        raise ValueError(f"Unsupported optimizer method: {self.optimizer_method}")
