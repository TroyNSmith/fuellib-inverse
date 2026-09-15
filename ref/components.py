"""Functions for calculating individual properties of fuel components."""

from typing import TypeVar

import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from ..utils.units import convert_temperature
from .types import CriticalProperties

NumCompounds = TypeVar("NumCompounds", bound=int)


# Percentage / fraction conversions
def mole_fractions(Y: Array, props: CriticalProperties) -> Array:
    """Convert weight fractions to mole fractions of a fuel mixture."""
    MW = props.MW
    X = (Y / MW) / qnp.sum(Y / MW)
    return qnp.array(X.value)


def volume_fractions(X: Array, props: CriticalProperties) -> Array:
    """Convert mole fractions to volume fractions of a fuel mixture."""
    Vm_stp = props.Vm_stp
    V = (X * Vm_stp) / qnp.sum(X * Vm_stp)
    return qnp.array(V.value)


# Component property correlations
def molar_liquid_vols(
    T: AbstractQuantity, props: CriticalProperties
) -> AbstractQuantity:
    """Calculate the molar liquid volume of fuel compounds over a range of temperatures."""
    Tstp = Quantity(298.15, "K")

    T = convert_temperature(T, "K")
    Tc = convert_temperature(props.Tc, "K")

    condition = qnp.array(T.value) > qnp.array(Tc.value)
    x = -qnp.power(1 - (Tstp / Tc), 2.0 / 7.0)
    y = qnp.power(1 - (T / Tc), 2.0 / 7.0) + x

    phi: Array = qnp.where(condition, x, y).value  # ty: ignore[unresolved-attribute]

    z1 = 0.29056
    z2 = 0.08775
    z: Array = (z1 - z2 * props.omega).value  # ty: ignore[invalid-assignment]

    return props.Vm_stp * qnp.power(z, phi)


def densities(T: AbstractQuantity, props: CriticalProperties) -> AbstractQuantity:
    """Calculate the densities of fuel components."""
    Vm = molar_liquid_vols(T, props)
    return (props.MW / Vm).to("kg/m^3")


def kinematic_viscosities(
    T: AbstractQuantity, props: CriticalProperties
) -> AbstractQuantity:
    """Calculate the kinematic viscosities of fuel components."""
    T = convert_temperature(T, "Celsius")
    Tb = convert_temperature(props.Tb, "Celsius")

    num = Quantity(442.78, "Celsius") + 1.6452 * Tb
    denom = T + Quantity(239.0, "Celsius") - 0.19 * Tb
    return Quantity(qnp.exp(-3.0171 + (num / denom)).value, "mm^2/s").to("m^2/s")
