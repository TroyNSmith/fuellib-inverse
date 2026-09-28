"""Mixture correlations for the fuellib package."""

from typing import TYPE_CHECKING, Literal

from ..utils import Units, types
from ..utils import resolved_np as np
from . import components
from .helpers import arithmetic_mixing_rule, weight_to_mol_fraction

if TYPE_CHECKING:
    from ..fuel import Fuel


def density(
    fuel: "Fuel", Yi: types.Array1D, T: types.Quantity0D | types.Quantity1D
) -> types.Quantity0D | types.Quantity1D:
    """Density for the mixture at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: types.Array1D
    :param T: Temperature at which to evaluate the density.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Density of the mixture.
    :rtype: types.Quantity0D | types.Quantity1D
    """
    rho_i = components.density(fuel, T).to("kg/m^3").magnitude
    return Units.Q(Yi @ rho_i, "kg/m^3")


def kinematic_viscosity(
    fuel: "Fuel",
    Yi: types.Array1D,
    T: types.Quantity0D | types.Quantity1D,
    *,
    correlation: Literal["kendall-monroe", "arrhenius"] = "kendall-monroe",
) -> types.Quantity0D | types.Quantity1D:
    """Kinematic viscosity for the mixture at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: types.Array1D
    :param T: Temperature at which to evaluate the kinematic viscosity.
    :type T: types.Quantity0D | types.Quantity1D
    :param correlation: Correlation method to use for the calculation.
    :type correlation: Literal["kendall-monroe", "arrhenius"]
    :return: Kinematic viscosity of the mixture.
    :rtype: types.Quantity0D | types.Quantity1D
    """
    MW = fuel.MW.to("kg/mol").magnitude
    Xi = weight_to_mol_fraction(Yi, np.array(MW))
    nu_i = components.kinematic_viscosity(fuel, T).to("m^2/s").magnitude

    if correlation.lower() == "arrhenius":
        nu = np.exp(np.dot(Xi, np.log(np.array(nu_i))))

    elif correlation.lower() == "kendall-monroe":
        nu = np.power(np.dot(Xi, np.power(np.array(nu_i), 1.0 / 3.0)), 3.0)

    else:
        msg = f"Unsupported correlation method: {correlation}"
        raise ValueError(msg)

    return Units.Q(nu, "m^2/s")


def saturated_vapor_pressure(
    fuel: "Fuel",
    Yi: types.Array1D,
    T: types.Quantity0D | types.Quantity1D,
    *,
    correlation: Literal["ambrose-walton", "lee-kesler"] = "lee-kesler",
) -> types.Quantity0D | types.Quantity1D:
    """Saturated vapor pressure for the mixture at a given temperature or temperatures.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: types.Array1D
    :param T: Temperature at which to evaluate the saturated vapor pressure.
    :type T: types.Quantity0D | types.Quantity1D
    :param correlation: Correlation method to use for the calculation.
    :type correlation: Literal["ambrose-walton", "lee-kesler"]
    :return: Saturated vapor pressure of the mixture.
    :rtype: types.Quantity0D | types.Quantity1D
    """
    MW = fuel.MW.to("kg/mol").magnitude
    Xi = weight_to_mol_fraction(Yi, np.array(MW))
    P_i = (
        components.saturated_vapor_pressure(fuel, T, correlation=correlation)
        .to("Pa")
        .magnitude
    )
    return Units.Q(Xi @ P_i, "Pa")


def surface_tension(
    fuel: "Fuel",
    Yi: types.Array1D,
    T: types.Quantity0D | types.Quantity1D,
    *,
    correlation: Literal["pitzer", "brock-bird"] = "brock-bird",
) -> types.Quantity0D | types.Quantity1D:
    """Surface tension for the mixture at a given temperature or temperatures.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: types.Array1D
    :param T: Temperature at which to evaluate the surface tension.
    :type T: types.Quantity0D | types.Quantity1D
    :param correlation: Correlation method to use for the calculation.
    :type correlation: Literal["pitzer", "brock-bird"]
    :return: Surface tension of the mixture.
    :rtype: types.Quantity0D | types.Quantity1D
    """
    MW = fuel.MW.to("kg/mol").magnitude
    Xi = weight_to_mol_fraction(Yi, np.array(MW))
    sigma_i = (
        components.surface_tension(fuel, T, correlation=correlation).to("N/m").magnitude
    )
    st = arithmetic_mixing_rule(Xi, sigma_i)
    return Units.Q(st, "N/m")


def thermal_conductivity(
    fuel: "Fuel", Yi: types.Array1D, T: types.Quantity0D | types.Quantity1D
) -> types.Quantity0D | types.Quantity1D:
    """Thermal conductivity for the mixture at a given temperature or temperatures.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param Yi: Weight fractions of the components.
    :type Yi: types.Array1D
    :param T: Temperature at which to evaluate the thermal conductivity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Thermal conductivity of the mixture.
    :rtype: types.Quantity0D | types.Quantity1D
    """
    tc_i = components.thermal_conductivity(fuel, T).to("W/(m*K)").magnitude
    tc = np.power(np.dot(Yi, np.power(np.array(tc_i), -2)), -0.5)
    return Units.Q(tc, "W/(m*K)")
