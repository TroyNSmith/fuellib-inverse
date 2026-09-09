"""JIT-optimized mixture correlation functions for FuelLib."""

from typing import Literal

import quaxed.numpy as qnp
from beartype import beartype as typechecker
from jax import Array
from jaxtyping import ArrayLike, Float, Shaped, jaxtyped
from unxt import AbstractQuantity, Quantity


def aromatics_content(X: Array, hydrocarbon_types: list[str]) -> Array:
    """
    Calculate the aromatics content of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Shaped[Array, "num_compounds"]
    :param hydrocarbon_types: List of hydrocarbon families for each component.
    :type hydrocarbon_types: list[str]
    :return: Aromatics content as a quantity with units of mole fraction.
    :rtype: AbstractQuantity
    """
    aromatics_mask = qnp.array(
        [1 if hct.lower() == "aromatic" else 0 for hct in hydrocarbon_types]
    )
    return qnp.sum(X * aromatics_mask)


def density(X: Array, densities: Array) -> Array:
    """
    Calculate the density of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Shaped[Array, "num_compounds"]
    :param densities: Densities of the fuel components as a quantity with units of kg/m^3.
    :type densities: Shaped[AbstractQuantity, "num_compounds"]
    :return: Density of the fuel mixture as a quantity with units of kg/m^3.
    :rtype: AbstractQuantity
    """
    return qnp.sum(X * densities)


def kinematic_viscosity(T: Array, X: Array, kinematic_viscosities: Array) -> Array:
    """
    Calculate the kinematic viscosity of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Shaped[Array, "num_compounds"]
    :param kinematic_viscosities: Kinematic viscosities of the fuel components as a quantity with units of m^2/s.
    :type kinematic_viscosities: Shaped[AbstractQuantity, "num_compounds"]
    :return: Kinematic viscosity of the fuel mixture as a quantity with units of m^2/s.
    :rtype: AbstractQuantity
    """
    return qnp.exp(qnp.sum(X * qnp.log(kinematic_viscosities), axis=-1))  # m^2/s
