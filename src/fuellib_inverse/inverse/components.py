"""Functions for calculating individual properties of fuel components."""

import jax.numpy as jnp
from jax import Array


# Percentage / fraction conversions
def mole_fractions(Y: Array, MW: Array) -> Array:
    """
    Calculate the mole fraction of a fuel mixture.

    :param Y: Weight percentages of the fuel components.
    :type Y: Array
    :param MW: Molecular weights of the fuel components.
    :type MW: Array
    :return: Mole fractions of the fuel components.
    :rtype: Array
    """
    Y = jnp.array(Y)
    MW = jnp.array(MW)
    return (Y / MW) / jnp.sum(Y / MW)


def volume_fractions(X: Array, Vm_stp: Array) -> Array:
    """
    Calculate the volume fraction of a fuel mixture.

    :param X: Mole fractions of the fuel components.
    :type X: Array
    :param Vm_stp: Molar volumes of the fuel components at standard temperature and pressure (STP).
    :type Vm_stp: Array
    :return: Volume fraction of the fuel components.
    :rtype: Array
    """
    X = jnp.array(X)
    Vm_stp = jnp.array(Vm_stp)
    return (X * Vm_stp) / jnp.sum(X * Vm_stp)


# Component property correlations
def molar_liquid_vols(T: Array, Tc: Array, Vm_stp: Array, omega: Array) -> Array:
    """
    Calculate the molar liquid volume of fuel compounds over a range of temperatures.

    :param T: Temperatures (K) at which to calculate the molar liquid volume.
    :type T: Array
    :param Tc: Critical temperatures (K) of the fuel components.
    :type Tc: Array
    :param Vm_stp: Molar volumes (m^3/mol) of the fuel components at standard temperature and pressure (STP).
    :type Vm_stp: Array
    :param omega: Acentric factors of the fuel components.
    :type omega: Array
    :return: Molar liquid volume (m^3/mol) of fuel compounds at the specified temperature.
    :rtype: Array
    """
    Tstp = 298.15

    x = -jnp.power(1 - (Tstp / Tc), 2.0 / 7.0)
    y = jnp.power(1 - (T / Tc), 2.0 / 7.0) + x

    phi = jnp.where(T > Tc, x, y)

    z = 0.29056 - 0.08775 * omega

    return Vm_stp * jnp.power(z, phi)


def densities(T: Array, Tc: Array, Vm_stp: Array, omega: Array, MW: Array) -> Array:
    """
    Calculate the densities of fuel components.

    :param T: Temperatures (K) at which to calculate the densities.
    :type T: Array
    :param Tc: Critical temperatures (K) of the fuel components.
    :type Tc: Array
    :param Vm_stp: Molar volumes (m^3/mol) of the fuel components at standard temperature and pressure (STP).
    :type Vm_stp: Array
    :param omega: Acentric factors of the fuel components.
    :type omega: Array
    :param MW: Molecular weights (g/mol) of the fuel components.
    :type MW: Array
    :return: Densities (kg/m^3) of the fuel components.
    :rtype: Array
    """
    MW = MW / 1000.0  # Convert g/mol to kg/mol for density in kg/m^3
    Vm = molar_liquid_vols(T, Tc, Vm_stp, omega)  # m^3/mol
    return MW / Vm


def kinematic_viscosities(T: Array, Tb: Array) -> Array:
    """
    Calculate the kinematic viscosities of fuel components.

    :param T: Temperatures (K) at which to calculate the kinematic viscosities.
    :type T: Array
    :param Tb: Boiling points (K) of the fuel components.
    :type Tb: Array
    :return: Kinematic viscosities (m^2/s) of the fuel components.
    :rtype: Array
    """
    T = T - 273.15  # Convert K to Celsius
    Tb = Tb - 273.15  # Convert K to Celsius

    num = 442.78 + 1.6452 * Tb
    denom = T + 239.0 - 0.19 * Tb
    nu_mm2_s = jnp.exp(-3.0171 + (num / denom))  # mm^2/s
    return nu_mm2_s * 1e-6  # Convert mm^2/s to m^2/s
