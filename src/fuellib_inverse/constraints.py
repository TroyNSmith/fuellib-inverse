"""Constraints object for FuelLib Inverse."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Self

import jax.numpy as jnp
import pint
from fuellib import PintUnits, fuel
from jax import Array
from pydantic import BaseModel, ConfigDict

DEFAULT_UNITS = {
    "temp": "K",
    "density": "kg/m^3",
    "viscosity": "mm^2/s",
    "MW": "g/mol",
}


def square_hinge_penalty(value: Array, lower: float, upper: float) -> Array:
    """Squared-hinge penalty for a value constrained to ``[min, max]``."""
    below = jnp.maximum(lower - value, 0)
    above = jnp.maximum(value - upper, 0)
    return below**2 + above**2


def Y2X(Y: Array, MW: Array) -> Array:
    """Convert weight fraction vector to mole fraction vector given molecular weights."""
    return Y / MW / jnp.sum(Y / MW)


def _magnitude(value: pint.Quantity | float) -> float:
    """Return a raw float from a bound that may be a `pint.Quantity` or a plain number."""
    if isinstance(value, pint.Quantity):
        return float(value.magnitude)
    return float(value)


class Constraint(BaseModel, ABC):
    """Base class for all constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    property: str
    lambda_: float
    target_range: pint.Quantity | tuple[float, float]
    fuel: fuel

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

        target: pint.Quantity | float | None = inp_dict.get("target", None)

        if isinstance(target, tuple) and len(target) == 2:
            target_range = target
        elif isinstance(target, int | float):
            low = target - atol - rtol * target
            high = target + atol + rtol * target
            target_range = (low, high)
        else:
            raise ValueError(
                f"Target for {property} must have exactly one or two values."
            )

        constraint = {
            "property": property,
            "lambda_": lambda_,
            "target_range": target_range,
            "fuel": fuel_,
        }
        return cls.model_validate(constraint)

    @cached_property
    def aromatics(self) -> Array:
        """Return the list of aromatics for the assigned fuel."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        fam = jnp.array(self.fuel.fam)
        return jnp.where(fam == 1, 1, 0)

    @cached_property
    def MW(self) -> Array:
        """Return the molecular weights for the assigned fuel."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        return jnp.array(self.fuel.MW.to(DEFAULT_UNITS["MW"]))

    def loss(self, Y: Array) -> Array:
        """Return the loss for the aromatics constraint given a weight fraction vector."""
        X = Y2X(Y, self.MW)
        val = jnp.sum(X * self.aromatics)
        lower, upper = self.target_range
        return self.lambda_ * square_hinge_penalty(val, float(lower), float(upper))


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
        for temp, target in targets.items():
            temp = PintUnits.Quantity(temp, temp_units)
            if isinstance(target, tuple) and len(target) == 2:
                target_range = target
            elif isinstance(target, float):
                low = target - atol - rtol * target
                high = target + atol + rtol * target
                target_range = (low, high)
            else:
                raise ValueError(
                    f"Target for {property} must have exactly one or two values."
                )

            constraint = {
                "property": property,
                "lambda_": lambda_,
                "temperature": temp,
                "fuel": fuel_,
                "target_range": PintUnits.Quantity(target_range, prop_units),
            }
            constraints.append(cls.model_validate(constraint))

        return constraints


class DensityConstraint(Constraint, TemperatureMixin):
    """Temperature-dependent density constraint for inverse optimization."""

    @cached_property
    def density(self) -> Array:
        """Return the array of densities for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        rho = self.fuel.density(self.temperature).to(DEFAULT_UNITS["density"])
        return jnp.array(rho.magnitude)

    def loss(self, Y: Array) -> Array:
        """Return the loss for the density constraint given a weight fraction vector."""
        val = Y @ self.density
        lower, upper = self.target_range
        return self.lambda_ * square_hinge_penalty(
            val, _magnitude(lower), _magnitude(upper)
        )


class ViscosityConstraint(Constraint, TemperatureMixin):
    """Temperature-dependent viscosity constraint for inverse optimization."""

    @cached_property
    def viscosity(self) -> Array:
        """Return the array of viscosities for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        mu = self.fuel.viscosity_kinematic(self.temperature).to(
            DEFAULT_UNITS["viscosity"]
        )
        return jnp.array(mu.magnitude)

    @cached_property
    def MW(self) -> Array:
        """Return the array of molecular weights for the assigned fuel components."""
        if not self.fuel:
            raise ValueError("No fuel assigned to the constraint.")
        mw = self.fuel.MW.to(DEFAULT_UNITS["MW"])
        return jnp.array(mw.magnitude)

    def loss(self, Y: Array) -> Array:
        """Return the loss for the viscosity constraint given a weight fraction vector."""
        X = Y2X(Y, self.MW)
        val = jnp.power(jnp.sum(X * jnp.power(self.viscosity, 1.0 / 3.0)), 3.0)
        lower, upper = self.target_range
        return self.lambda_ * square_hinge_penalty(
            val, _magnitude(lower), _magnitude(upper)
        )
