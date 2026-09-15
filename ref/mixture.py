"""JIT-optimized mixture correlation functions for FuelLib."""

import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from .types import CriticalProperties


def aromatics_content(X: Array, props: CriticalProperties) -> Array:
    """
    Calculate the aromatics content of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Array
    :param props: Critical properties of the fuel components (uses
        `props.hydrocarbon_types`).
    :type props: CriticalProperties
    :return: Aromatics content as a mole fraction.
    :rtype: Array
    """
    aromatics_mask = qnp.array(
        [1 if hct.lower() == "aromatic" else 0 for hct in props.hydrocarbon_types]
    )
    return qnp.sum(X * aromatics_mask)


def density(X: Array, densities: AbstractQuantity) -> AbstractQuantity:
    """
    Calculate the density of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Array
    :param densities: Densities of the fuel components.
    :type densities: AbstractQuantity
    :return: Density of the fuel mixture.
    :rtype: AbstractQuantity
    """
    return qnp.sum(X * densities)


def kinematic_viscosity(
    X: Array, kinematic_viscosities: AbstractQuantity
) -> AbstractQuantity:
    """
    Calculate the kinematic viscosity of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Array
    :param kinematic_viscosities: Kinematic viscosities of the fuel components.
    :type kinematic_viscosities: AbstractQuantity
    :return: Kinematic viscosity of the fuel mixture.
    :rtype: AbstractQuantity
    """
    nu_i = qnp.array(kinematic_viscosities.to("m^2/s").value)
    return Quantity(qnp.exp(qnp.sum(X * qnp.log(nu_i), axis=-1)), "m^2/s")
