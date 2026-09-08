"""Units conversion utilities for FuelLib."""

"""unxt and astropy units handling with helpers."""

import astropy.units as apyu
import jax.numpy as jnp
from jax import Array
from unxt import AbstractQuantity

# astropy does not have some common unit strings; we need to define and register them
## Pressure (defined in terms of Pa)
atm = apyu.def_unit("atm", 101325 * apyu.Pa, doc="Standard atmosphere")
dyne_cm2 = apyu.def_unit("dyne/cm^2", 0.1 * apyu.Pa, doc="Dyne per square centimeter")
cgs = apyu.def_unit("cgs", 0.1 * apyu.Pa, doc="CGS unit of pressure")
mks = apyu.def_unit("mks", 1 * apyu.Pa, doc="MKS unit of pressure")
## Temperature (defined in terms of K)
fahrenheit = apyu.def_unit(
    "Fahrenheit",
    1 * apyu.imperial.deg_F,
    doc="Fahrenheit temperature unit",
)

## Register the new units with astropy so they can be used in unxt.Quantity objects.
apyu.add_enabled_units([atm, mks, dyne_cm2, cgs, fahrenheit])


def convert_temperature(temp: AbstractQuantity, target_unit: str) -> AbstractQuantity:
    """
    Convert a temperature quantity to a different unit.

    :param temp: Temperature quantity to convert.
    :type temp: AbstractQuantity
    :param target_unit: Unit to convert to.
    :type target_unit: str
    :return: Converted temperature quantity.
    :rtype: unxt.AbstractQuantity
    """
    with apyu.add_enabled_equivalencies(apyu.temperature()):
        return temp.uconvert(target_unit)


def strip(quant: AbstractQuantity) -> Array:
    """
    Strip the units from an AbstractQuantity and return the raw value in an array.

    :param quant: Quantity with units.
    :type quant: AbstractQuantity
    :return: Raw array without units.
    :rtype: jax.Array
    """
    return jnp.array(quant.value)


__all__ = ["convert_temperature", "strip"]
