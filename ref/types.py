"""Types for the inverse module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import jax.numpy as jnp
import pandas as pd
import quaxed.numpy as qnp
from beartype import beartype as typechecker
from jax import Array
from jaxtyping import jaxtyped
from unxt import AbstractQuantity, Quantity

from ..utils.units import convert_temperature
from . import parse_input


def _parse_csv_column(
    df: pd.DataFrame, column_name: str, target_units: str
) -> AbstractQuantity:
    """
    Parse a column from a DataFrame and convert it to a `Quantity` in its canonical unit.

    :param df: DataFrame containing the data.
    :type df: pd.DataFrame
    :param column_name: Name of the column to parse.
    :type column_name: str
    :param target_units: Target units for the column data.
    :type target_units: str
    :return: Quantity containing the parsed data, converted to `target_units`.
    :rtype: AbstractQuantity
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
        return convert_temperature(quants, target_units)

    return quants.to(target_units)


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
    MW: AbstractQuantity  # g/mol
    Tc: AbstractQuantity  # K
    Pc: AbstractQuantity  # Pa
    Vc: AbstractQuantity  # m^3/mol
    Tb: AbstractQuantity  # K
    Tm: AbstractQuantity  # K
    Hf: AbstractQuantity  # J/mol
    Gf: AbstractQuantity  # J/mol
    Hv_stp: AbstractQuantity  # J/mol
    Vm_stp: AbstractQuantity  # m^3/mol
    Cp_stp: AbstractQuantity  # J/(mol*K)
    Cp_B: AbstractQuantity  # J/(mol*K)
    Cp_C: AbstractQuantity  # J/(mol*K)
    omega: AbstractQuantity  # dimensionless

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

    def mole_fractions(self, Y: Array) -> Array:
        """
        Calculate the mole fraction of a fuel mixture.

        :param Y: Weight fractions of the fuel components.
        :type Y: Array
        :return: Mole fractions of the fuel components.
        :rtype: Array
        """
        X = (Y / self.MW) / qnp.sum(Y / self.MW)
        return qnp.array(X.value)

    def volume_fractions(self, X: Array) -> Array:
        """
        Calculate the volume fraction of a fuel mixture.

        :param X: Mole fractions of the fuel components.
        :type X: Array
        :return: Volume fraction of the fuel components.
        :rtype: Array
        """
        V = (X * self.Vm_stp) / qnp.sum(X * self.Vm_stp)
        return qnp.array(V.value)

    def molar_liquid_vols(self, T: AbstractQuantity) -> AbstractQuantity:
        """
        Calculate the molar liquid volumes of the fuel components at a given temperature.

        :param T: Temperature at which to calculate the molar liquid volumes.
        :type T: AbstractQuantity
        :return: Molar liquid volumes of the fuel components.
        :rtype: AbstractQuantity
        """
        Tstp = Quantity(298.15, "K")

        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")

        condition = qnp.array(T.value) > qnp.array(Tc.value)
        x = -qnp.power(1 - (Tstp / Tc), 2.0 / 7.0)
        y = qnp.power(1 - (T / Tc), 2.0 / 7.0) + x

        phi: Array = qnp.where(condition, x, y).value  # ty: ignore[unresolved-attribute]

        z1 = 0.29056
        z2 = 0.08775
        z: Array = (z1 - z2 * self.omega).value  # ty: ignore[invalid-assignment]

        return self.Vm_stp * qnp.power(z, phi)

    def densities(self, T: AbstractQuantity) -> AbstractQuantity:
        """
        Calculate the densities of the fuel components at a given temperature.

        :param T: Temperatures at which to calculate the densities.
        :type T: AbstractQuantity
        (Note: `props` parameter is removed since the method now uses instance attributes.)
        :return: Densities of the fuel components.
        :rtype: AbstractQuantity
        """
        Vm = self.molar_liquid_vols(T)
        return (self.MW / Vm).to("kg/m^3")


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


class _MinMax(TypedDict):  # noqa: PYI049 (used cross-module by core.py)
    """Bounds for a single scalar constraint, penalized outside [min, max]."""

    lambda_: float
    max: AbstractQuantity
    min: AbstractQuantity


class _GCxGCLeaf(TypedDict):  # noqa: PYI049 (used cross-module by core.py)
    """
    Config for matching a GCxGC reference composition CSV, penalizing each matched
    compound's optimized weight % outside \\[Weight % * (1 - rtol) - atol, Weight %
    * (1 + rtol) + atol\\].
    """

    lambda_: float
    rtol: float
    atol: float
    csv_path: Path


# A constraint tree is a nested mapping whose leaves are either `_MinMax` bounds,
# e.g. {"composition": {"aromatics": {"volume_percent": {...}}}}, or a `_GCxGCLeaf`
# (under a "gcxgc" key), e.g. {"composition": {"gcxgc": {...}}}. `parse_input.load`
# is responsible for building a fully resolved tree (bounds already `Quantity`s)
# straight from `input.txt` - no further resolution happens once it's built.
ConstraintTree = dict[str | float, Any]

DEFAULT_CONSTRAINTS: ConstraintTree = {
    "composition": {
        "aromatics": {
            "volume_percent": {
                "lambda_": 1.0,
                "min": Quantity(jnp.asarray(8.0), "dimensionless"),
                "max": Quantity(jnp.asarray(25.0), "dimensionless"),
            }
        }
    },
}


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the inverse problem solver."""

    optimization: _OptimizationParams = field(default_factory=_default_optimization)
    constraints: ConstraintTree = field(default_factory=lambda: DEFAULT_CONSTRAINTS)

    @classmethod
    def from_input(cls, input_path: str | Path) -> "SolverConfig":
        """
        Build solver configuration from a human-readable input.txt-format file
        (see `test/INPUT_FORMAT.md` for the format spec, and `parse_input` for the
        parser that reads and fully resolves it).

        :param input_path: Path to the input.txt-format file containing solver
            configuration.
        :type input_path: str | Path
        """
        optimization_overrides, constraints = parse_input.load(input_path)
        optimization = _OptimizationParams(
            **{**_default_optimization(), **optimization_overrides}
        )
        return cls(
            optimization=optimization, constraints=constraints or DEFAULT_CONSTRAINTS
        )
