"""Constraints object for FuelLib Inverse."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Self

import jax.numpy as jnp
import numpy as np
import pint
from fuellib import PintUnits, fuel
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

DEFAULT_UNITS = {
    "temp": "K",
    "density": "kg/m^3",
    "viscosity": "mm^2/s",
    "MW": "g/mol",
}


def _magnitude(value: pint.Quantity | float) -> float:
    """Return a raw float from a bound that may be a `pint.Quantity` or a plain number."""
    if isinstance(value, pint.Quantity):
        return float(value.magnitude)
    return float(value)


def _bounds(target_range: pint.Quantity | tuple[float, float]) -> tuple[float, float]:
    """Return the ``(lower, upper)`` bounds of a target range as plain floats."""
    lower, upper = target_range
    return _magnitude(lower), _magnitude(upper)


def half_width_scale(lower: float, upper: float) -> float:
    """Return a characteristic scale for a target range, used to normalize penalties.

    Uses the half-width of the allowed range (which already reflects the
    property's magnitude via ``atol``/``rtol``), so that a given absolute
    deviation is penalized proportionally to how large the property's own
    tolerance band is (e.g. a 1 kg/m^3 density miss is penalized less than a
    1 mm^2/s viscosity miss). Falls back to the midpoint magnitude, then to
    1.0, for degenerate (zero-width, zero-midpoint) ranges.
    """
    width = (upper - lower) / 2.0
    if width > 0:
        return width
    midpoint = abs((upper + lower) / 2.0)
    return midpoint if midpoint > 0 else 1.0


def _resolve_target_range(
    target: float | tuple[float, float] | None,
    atol: float,
    rtol: float,
    property: str,
) -> tuple[float, float]:
    """Resolve a scalar target (with symmetric tolerances) or an explicit range."""
    if isinstance(target, tuple) and len(target) == 2:
        return target
    if isinstance(target, int | float):
        margin = atol + rtol * target
        return (target - margin, target + margin)
    raise ValueError(f"Target for {property} must have exactly one or two values.")


def square_hinge_penalty(
    value: Array, lower: float, upper: float, scale: float = 1.0
) -> Array:
    """Squared-hinge penalty for a value constrained to ``[lower, upper]``.

    The deviation is normalized by `scale` before squaring, so that penalties
    across constraints on properties with different units/magnitudes (e.g.
    density in kg/m^3 vs. viscosity in mm^2/s) are comparable.
    """
    below = jnp.maximum(lower - value, 0) / scale
    above = jnp.maximum(value - upper, 0) / scale
    return below**2 + above**2


def Y2X(Y: Array, MW: ArrayLike) -> Array:
    """Convert weight fraction vector to mole fraction vector given molecular weights."""
    return Y / MW / jnp.sum(Y / MW)


class Constraint(BaseModel, ABC):
    """Base class for all constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    property: str
    lambda_: float
    target_range: pint.Quantity | tuple[float, float]
    fuel: fuel

    @abstractmethod
    def value(self, Y: Array) -> Array:
        """Return the current value of the constraint given a weight fraction vector."""

    @abstractmethod
    def loss(self, Y: Array) -> Array:
        """Return the loss for the constraint given a weight fraction vector."""


class AromaticsConstraint(Constraint):
    """Aromatics content constraint for inverse optimization."""

    @classmethod
    def from_input(
        cls, property: str, inp_dict: dict, fuel_: fuel
    ) -> "AromaticsConstraint":
        """Parse an aromatics constraint from an input dictionary."""
        lambda_: float = inp_dict.get("lambda", 1.0)
        atol: float = inp_dict.get("atol", 0.0)
        rtol: float = inp_dict.get("rtol", 0.05)
        target: float | tuple[float, float] | None = inp_dict.get("target", None)

        constraint = {
            "property": property,
            "lambda_": lambda_,
            "target_range": _resolve_target_range(target, atol, rtol, property),
            "fuel": fuel_,
        }
        return cls.model_validate(constraint)

    @cached_property
    def aromatics(self) -> np.ndarray:
        """Return the list of aromatics for the assigned fuel."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        fam = np.array(self.fuel.fam)
        return np.where(fam == 1, 1, 0)

    @cached_property
    def MW(self) -> np.ndarray:
        """Return the molecular weights for the assigned fuel."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        return np.array(self.fuel.MW.to(DEFAULT_UNITS["MW"]))

    def value(self, Y: Array) -> Array:
        """Return the current value of the aromatics fraction given a weight fraction vector."""
        X = Y2X(Y, self.MW)
        return jnp.sum(X * self.aromatics)

    def loss(self, Y: Array) -> Array:
        """Return the loss for the aromatics constraint given a weight fraction vector."""
        lower, upper = _bounds(self.target_range)
        scale = half_width_scale(lower, upper)
        return self.lambda_ * square_hinge_penalty(self.value(Y), lower, upper, scale)


class TemperatureMixin(BaseModel):
    """Temperature-dependent physical constraint for inverse optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature: pint.Quantity

    @classmethod
    def from_input(cls, property: str, inp_dict: dict, fuel_: fuel) -> list[Self]:
        """Parse a temperature-dependent physical constraint from an input dictionary."""
        lambda_: float = inp_dict.get("lambda", 1.0)
        temp_units: str = inp_dict.get("temp_units", DEFAULT_UNITS["temp"])
        prop_units: str = inp_dict.get("units", DEFAULT_UNITS[property])

        atol: float = inp_dict.get("atol", 0.0)
        rtol: float = inp_dict.get("rtol", 0.05)
        targets: dict | None = inp_dict.get("target", None)

        if not targets:
            raise ValueError(
                f"No target values provided for {property} temperature constraints."
            )

        constraints: list[Self] = []
        for raw_temp, target in targets.items():
            target_range = _resolve_target_range(target, atol, rtol, property)
            constraint = {
                "property": property,
                "lambda_": lambda_,
                "temperature": PintUnits.Quantity(raw_temp, temp_units),
                "fuel": fuel_,
                "target_range": PintUnits.Quantity(target_range, prop_units),
            }
            constraints.append(cls.model_validate(constraint))

        return constraints


class DensityConstraint(Constraint, TemperatureMixin):
    """Temperature-dependent density constraint for inverse optimization."""

    @cached_property
    def density(self) -> np.ndarray:
        """Return the array of densities for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        rho = self.fuel.density(self.temperature).to(DEFAULT_UNITS["density"])
        return np.array(rho.magnitude)

    def value(self, Y: Array) -> Array:
        """Return the current value of the density given a weight fraction vector."""
        return Y @ self.density

    def loss(self, Y: Array) -> Array:
        """Return the loss for the density constraint given a weight fraction vector."""
        lower, upper = _bounds(self.target_range)
        scale = half_width_scale(lower, upper)
        return self.lambda_ * square_hinge_penalty(self.value(Y), lower, upper, scale)


class ViscosityConstraint(Constraint, TemperatureMixin):
    """Temperature-dependent viscosity constraint for inverse optimization."""

    @cached_property
    def viscosity(self) -> np.ndarray:
        """Return the array of viscosities for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        mu = self.fuel.viscosity_kinematic(self.temperature).to(
            DEFAULT_UNITS["viscosity"]
        )
        return np.array(mu.magnitude)

    @cached_property
    def MW(self) -> np.ndarray:
        """Return the array of molecular weights for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        mw = self.fuel.MW.to(DEFAULT_UNITS["MW"])
        return np.array(mw.magnitude)

    def value(self, Y: Array) -> Array:
        """Return the current value of the viscosity given a weight fraction vector."""
        X = Y2X(Y, self.MW)
        return jnp.power(jnp.sum(X * jnp.power(self.viscosity, 1.0 / 3.0)), 3.0)

    def loss(self, Y: Array) -> Array:
        """Return the loss for the viscosity constraint given a weight fraction vector."""
        lower, upper = _bounds(self.target_range)
        scale = half_width_scale(lower, upper)
        return self.lambda_ * square_hinge_penalty(self.value(Y), lower, upper, scale)
