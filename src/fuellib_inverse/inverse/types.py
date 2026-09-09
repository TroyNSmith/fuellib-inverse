"""Types for the inverse module."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import jax.numpy as jnp
import pandas as pd
from beartype import beartype as typechecker
from jax import Array
from jaxtyping import Float, jaxtyped
from unxt import Quantity

from ..utils.units import convert_temperature, strip


def _parse_csv_column(df: pd.DataFrame, column_name: str, target_units: str) -> Array:
    """
    Parse a column from a DataFrame and convert it to a JAX array.

    :param df: DataFrame containing the data.
    :type df: pd.DataFrame
    :param column_name: Name of the column to parse.
    :type column_name: str
    :param target_units: Target units for the column data.
    :type target_units: str
    :return: JAX array containing the parsed data.
    :rtype: jnp.ndarray
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    column = df[column_name]
    units = column.iloc[0]
    if units == "":
        raise ValueError(
            f"Units for column '{column_name}' are missing in the first row of the DataFrame."
        )

    values = jnp.asarray(column.iloc[1:].to_numpy(dtype=float))
    quants = Quantity(values, units)

    if str(quants.unit.physical_type).lower() == "temperature":
        quants = convert_temperature(quants, target_units)

    return strip(quants)


# Numeric columns to read from the critical-properties CSV, mapped to their
# target units. To add a new critical property: add a field of the same name
# below, then add one entry here.
_CRITICAL_PROPERTY_UNITS: dict[str, str] = {
    "MW": "g/mol",
    "Tc": "K",
    "Pc": "Pa",
    "Vc": "m^3/mol",
    "Tb": "K",
    "Tm": "K",
    "Hf": "J/mol",
    "Gf": "J/mol",
    "Hv_stp": "J/mol",
    "Vm_stp": "m^3/mol",
    "Cp_stp": "J/(mol*K)",
    "Cp_B": "J/(mol*K)",
    "Cp_C": "J/(mol*K)",
    "omega": "dimensionless",
}

# Text columns to read verbatim (no unit conversion).
_CRITICAL_PROPERTY_TEXT_COLUMNS: dict[str, str] = {
    "smiles": "SMILES",
    "families": "Family",
    "hydrocarbon_types": "Hydrocarbon Type",
}


@jaxtyped(typechecker=typechecker)
@dataclass(frozen=True)
class CriticalProperties:
    """Critical properties of a fuel."""

    smiles: list[str]
    families: list[str]
    hydrocarbon_types: list[str]
    MW: Float[Array, "num_compounds"]  # g/mol
    Tc: Float[Array, "num_compounds"]  # K
    Pc: Float[Array, "num_compounds"]  # Pa
    Vc: Float[Array, "num_compounds"]  # m^3/mol
    Tb: Float[Array, "num_compounds"]  # K
    Tm: Float[Array, "num_compounds"]  # K
    Hf: Float[Array, "num_compounds"]  # J/mol
    Gf: Float[Array, "num_compounds"]  # J/mol
    Hv_stp: Float[Array, "num_compounds"]  # J/mol
    Vm_stp: Float[Array, "num_compounds"]  # m^3/mol
    Cp_stp: Float[Array, "num_compounds"]  # J/(mol*K)
    Cp_B: Float[Array, "num_compounds"]  # J/(mol*K)
    Cp_C: Float[Array, "num_compounds"]  # J/(mol*K)
    omega: Float[Array, "num_compounds"]  # dimensionless

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "CriticalProperties":
        """
        Build critical properties from a CSV file.

        :param csv_path: Path to the CSV file containing critical properties.
        :type csv_path: str | Path
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        text_fields = {
            field_name: df[column].iloc[1:].tolist()
            for field_name, column in _CRITICAL_PROPERTY_TEXT_COLUMNS.items()
        }
        numeric_fields = {
            field_name: _parse_csv_column(df, field_name, units)
            for field_name, units in _CRITICAL_PROPERTY_UNITS.items()
        }
        return cls(**text_fields, **numeric_fields)

    @property
    def num_compounds(self) -> int:
        """Return the number of compounds."""
        return len(self.smiles)


class _OptimizationParams(TypedDict):
    """Parameters for the optimization process."""

    max_iterations: int
    tolerance: float
    learning_rate: float
    regularization_strength: float


def _default_optimization() -> _OptimizationParams:
    return {
        "max_iterations": 1000,
        "tolerance": 1e-6,
        "learning_rate": 0.01,
        "regularization_strength": 0.1,
    }


class _MinMax(TypedDict):
    """Bounds for a single scalar constraint, penalized outside [min, max]."""

    lambda_: float
    max: float
    min: float


_MIN_MAX_KEYS = frozenset(_MinMax.__required_keys__)

# A constraint tree is an arbitrarily nested mapping whose leaves are `_MinMax`
# bounds, e.g. {"composition": {"aromatics": {"volume_percent": {...}}}}.
# New constraint categories can be added freely, just nest a `_MinMax` leaf
# anywhere in the tree - no code changes are required here.
#
# A node may also carry a "units" key naming the units its sibling `_MinMax`
# bounds are expressed in, e.g. {"density": {"units": "kg/m^3", "15 Celsius":
# {...}}}. Such bounds - and any sibling keys that are themselves parsable
# quantities, e.g. "15 Celsius" - are converted to SI by `SolverConfig`, which
# also replaces such keys with their SI magnitude (e.g. a Kelvin float).
# Leaves with no "units" (e.g. an already-dimensionless percentage) are used
# as-is.
ConstraintTree = dict[str | float, Any]

DEFAULT_CONSTRAINTS: ConstraintTree = {
    "composition": {
        "aromatics": {"volume_percent": {"lambda_": 1.0, "max": 25.0, "min": 8.0}}
    },
}


def _validate_constraint_tree(tree: ConstraintTree, path: str = "") -> None:
    """
    Recursively validate that every leaf of a constraint tree is a valid `_MinMax`.

    :param tree: Constraint tree (or subtree) to validate.
    :type tree: ConstraintTree
    :param path: Dotted key path to `tree`, used for error messages.
    :type path: str
    """
    for key, value in tree.items():
        node_path = f"{path}.{key}" if path else str(key)
        if not isinstance(value, dict):
            raise TypeError(
                f"Constraint '{node_path}' must be a mapping, got {value!r}."
            )
        if set(value.keys()) == _MIN_MAX_KEYS:
            continue
        _validate_constraint_tree(value, node_path)


def _to_si(value: float, units: str) -> float:
    """
    Convert a scalar value with the given units to its SI-equivalent magnitude.

    :param value: Numeric value expressed in `units`.
    :type value: float
    :param units: Units string parsable by `unxt.Quantity` (e.g. "kg/m^3", "Celsius").
    :type units: str
    :return: The value's magnitude when expressed in SI units.
    :rtype: float
    """
    quantity = Quantity(jnp.asarray(float(value)), units)
    if str(quantity.unit.physical_type).lower() == "temperature":
        quantity = convert_temperature(quantity, "K")
    else:
        quantity = quantity.decompose([])
    return float(strip(quantity))


def _try_parse_condition_key(key: str) -> float | None:
    """
    Try to interpret a constraint-tree key as a "<value> <units>" quantity (e.g.
    "15 Celsius"), returning its SI-converted magnitude.

    :param key: Key to attempt to parse.
    :type key: str
    :return: The SI magnitude if `key` is a quantity, else None for a plain
        label (e.g. "density", "volume_percent").
    :rtype: float | None
    """
    parts = key.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    value_str, units = parts
    try:
        return _to_si(float(value_str), units)
    except ValueError:
        return None


def _resolve_constraint_units(tree: ConstraintTree) -> ConstraintTree:
    """
    Recursively resolve "units" keys in a constraint tree (see `ConstraintTree`),
    converting every sibling `_MinMax` leaf's bounds - and any condition keys
    that parse as physical quantities, e.g. "15 Celsius" - to SI. Does not
    mutate `tree`.

    :param tree: Constraint tree (or subtree) to resolve.
    :type tree: ConstraintTree
    :return: An equivalent tree with all values expressed in SI units.
    :rtype: ConstraintTree
    """
    units = tree.get("units")
    resolved: ConstraintTree = {}
    for key, value in tree.items():
        if key == "units":
            continue
        if isinstance(value, dict) and set(value.keys()) == _MIN_MAX_KEYS:
            if units is not None:
                value = {
                    "lambda_": value["lambda_"],
                    "min": _to_si(value["min"], units),
                    "max": _to_si(value["max"], units),
                }
            parsed_key = _try_parse_condition_key(key) if isinstance(key, str) else None
            resolved[parsed_key if parsed_key is not None else key] = value
        elif isinstance(value, dict):
            resolved[key] = _resolve_constraint_units(value)
        else:
            resolved[key] = value
    return resolved


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the inverse problem solver."""

    optimization: _OptimizationParams = field(default_factory=_default_optimization)
    constraints: ConstraintTree = field(default_factory=lambda: DEFAULT_CONSTRAINTS)

    def __post_init__(self) -> None:
        constraints = _resolve_constraint_units(self.constraints)
        _validate_constraint_tree(constraints)
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "SolverConfig":
        """
        Build solver configuration from a JSON file.

        The top-level "optimization" key configures the optimizer; every other
        top-level key is treated as part of the constraint tree (see
        `ConstraintTree`), so new constraint categories can be added to the
        JSON file without changing this code. Constraint bounds are converted
        to SI units (via `unxt`) wherever a "units" key is declared.

        :param json_path: Path to the JSON file containing solver configuration.
        :type json_path: str | Path
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path) as f:
            input_data = json.load(f)

        optimization = _OptimizationParams(
            **{**_default_optimization(), **input_data.get("optimization", {})}
        )
        constraints = {k: v for k, v in input_data.items() if k != "optimization"}

        return cls(
            optimization=optimization, constraints=constraints or DEFAULT_CONSTRAINTS
        )
