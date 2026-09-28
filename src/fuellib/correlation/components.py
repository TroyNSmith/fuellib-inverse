"""Component correlations for the fuellib package.

Jax.numpy is used for numerical computations if available; otherwise, NumPy is used.
Note that JAX arrays are immutable, unlike NumPy arrays, so codes that modify arrays
in-place may need to be adapted. Additionally, JAX does not understand Pint Quantities
and cannot trace operations involving them, so units must be stripped at the interface
when using JAX.
"""

from typing import TYPE_CHECKING, Literal

from ..utils import Units, types
from ..utils import resolved_np as np
from .helpers import atleast_1d

if TYPE_CHECKING:
    from ..fuel import Fuel


def latent_heat_vaporization_stp(fuel: "Fuel") -> types.Quantity1D:
    """Standard latent heat of vaporization for each compound.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :return: Standard latent heat of vaporization for each compound in J/kg.
    :rtype: types.Quantity1D
    """
    Hv_stp = fuel.get_property("gani", "Hv_stp").to("J/mol").magnitude
    MW = fuel.MW.to("kg/mol").magnitude
    return Units.Quantity(Hv_stp / MW, "J/kg")


def epsilon_by_kb(fuel: "Fuel") -> types.Quantity1D:
    """Compute epsilon divided by Boltzmann constant for each compound.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :return: Epsilon divided by Boltzmann constant for each compound in Kelvin.
    :rtype: types.Quantity1D
    """
    omega = fuel.get_property("gani", "omega")
    Tc = fuel.get_property("gani", "Tc").to("K")
    return (Units.Q(1.0, "") + 0.1693 * omega) * Tc


def sigma(fuel: "Fuel") -> types.Quantity1D:
    """Compute the collision diameter (sigma) for each compound.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :return: Collision diameter (sigma) for each compound in meters.
    :rtype: types.Quantity1D
    """
    w = fuel.get_property("gani", "omega").magnitude
    tc = fuel.get_property("gani", "Tc").to("K").magnitude
    pc = fuel.get_property("gani", "Pc").to("atm").magnitude
    sigma = (2.3551 - 0.0874 * w) * np.power((tc / pc), 1.0 / 3)
    return Units.Quantity(sigma, "angstrom").to("m")


def molar_liquid_vol(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the molar liquid volume for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the molar liquid volume.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Molar liquid volume for each compound in m^3/mol.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Tvec = atleast_1d(T).to("K").magnitude
    Tc = fuel.get_property("gani", "Tc").to("K").magnitude[:, np.newaxis]
    W = fuel.get_property("gani", "omega").magnitude[:, np.newaxis]
    Vm_stp = fuel.get_property("gani", "Vm_stp").to("m^3/mol").magnitude[:, np.newaxis]

    x = -np.power((1 - (298.0) / Tc), 2.0 / 7.0)
    y = np.power((1 - (Tvec) / Tc), 2.0 / 7.0) + x
    phi = np.where(Tvec > Tc, x, y)

    z = 0.29056 - 0.08775 * W
    Vmi = Vm_stp * np.power(z, phi)
    return Units.Quantity(Vmi, "m^3/mol")


def density(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the density for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the density.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Density for each compound in kg/m^3.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Mlv = molar_liquid_vol(fuel, T).to("m^3/mol")
    MW = fuel.MW.to("kg/mol").magnitude[:, np.newaxis]
    rho = MW / Mlv.magnitude
    return Units.Quantity(rho, "kg/m^3")


def kinematic_viscosity(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the kinematic viscosity for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the kinematic viscosity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Kinematic viscosity for each compound in m^2/s.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Tvec = atleast_1d(T).to("celsius").magnitude
    Tb = fuel.get_property("gani", "Tb").to("celsius").magnitude[:, np.newaxis]

    rhs = -3.0171 + (442.78 + 1.6452 * Tb) / (Tvec + 239 - 0.19 * Tb)
    return Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")


def dynamic_viscosity(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the dynamic viscosity for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the dynamic viscosity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Dynamic viscosity for each compound in Pa*s.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    nu = kinematic_viscosity(fuel, T).to("m^2/s")
    rho = density(fuel, T).to("kg/m^3")
    mu = nu.magnitude * rho.magnitude
    return Units.Quantity(mu, "Pa*s")


def molar_specific_heat_capacity(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the molar specific heat capacity for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the molar specific heat capacity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Molar specific heat capacity for each compound in J/(mol*K).
    :rtype: types.Quantity1D | types.Quantity2D
    """
    # Placeholder implementation, replace with actual correlation
    Tvec = atleast_1d(T).to("K").magnitude
    Cp_stp = (
        fuel.get_property("gani", "Cp_stp").to("J/(mol*K)").magnitude[:, np.newaxis]
    )
    Cp_B = fuel.get_property("gani", "Cp_B").to("J/(mol*K)").magnitude[:, np.newaxis]
    Cp_C = fuel.get_property("gani", "Cp_C").to("J/(mol*K)").magnitude[:, np.newaxis]
    theta = (Tvec - 298.15) / 700

    cp = Cp_stp + Cp_B * theta + Cp_C * theta**2
    return Units.Quantity(cp, "J/(mol*K)")


def liquid_mass_specific_heat_capacity(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the liquid mass specific heat capacity for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the liquid mass specific heat capacity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Liquid mass specific heat capacity for each compound in J/(kg*K).
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Cp_molar = molar_specific_heat_capacity(fuel, T).to("J/(mol*K)").magnitude
    MW = fuel.MW.to("kg/mol").magnitude[:, np.newaxis]
    return Units.Quantity(Cp_molar / MW, "J/(kg*K)")


def saturated_vapor_pressure(
    fuel: "Fuel",
    T: types.Quantity0D | types.Quantity1D,
    *,
    correlation: Literal["ambrose-walton", "lee-kesler"] = "lee-kesler",
) -> types.Quantity1D | types.Quantity2D:
    """Compute the saturated vapor pressure for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the saturated vapor pressure.
    :type T: types.Quantity0D | types.Quantity1D
    :param correlation: Correlation method to use for computing the saturated vapor pressure.
    :type correlation: Literal["ambrose-walton", "lee-kesler"]
    :return: Saturated vapor pressure for each compound in Pa.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Tvec = atleast_1d(T).to("K").magnitude
    Tc = fuel.get_property("gani", "Tc").to("K").magnitude[:, np.newaxis]
    Pc = fuel.get_property("gani", "Pc").to("Pa").magnitude[:, np.newaxis]
    omega = fuel.get_property("gani", "omega").magnitude[:, np.newaxis]
    Tr = Tvec / Tc

    if correlation.lower() == "ambrose-walton":
        # NOTE: May cause trouble at high temperatures
        tau = 1 - Tr
        f0 = (
            -5.97616 * tau
            + 1.29874 * tau**1.5
            - 0.60394 * tau**2.5
            - 1.06841 * tau**5.0
        )
        f0 /= Tr
        f1 = (
            -5.03365 * tau
            + 1.11505 * tau**1.5
            - 5.41217 * tau**2.5
            - 7.46628 * tau**5.0
        )
        f1 /= Tr
        f2 = (
            -0.64771 * tau
            + 2.41539 * tau**1.5
            - 4.26979 * tau**2.5
            - 3.25259 * tau**5.0
        )
        f2 /= Tr
        rhs = np.exp(f0 + omega * f1 + omega**2 * f2)

    elif correlation.lower() == "lee-kesler":
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        rhs = np.exp(f0 + omega * f1)

    else:
        msg = f"Unsupported correlation method: {correlation}"
        raise ValueError(msg)

    return Units.Quantity(Pc * rhs, "Pa")


def surface_tension(
    fuel: "Fuel",
    T: types.Quantity0D | types.Quantity1D,
    *,
    correlation: Literal["pitzer", "brock-bird"] = "brock-bird",
) -> types.Quantity1D | types.Quantity2D:
    """Compute the surface tension for each compound at a given temperature.

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature at which to compute the surface tension.
    :type T: types.Quantity0D | types.Quantity1D
    :param correlation: Correlation method to use for the calculation.
    :type correlation: Literal["pitzer", "brock-bird"]
    :return: Surface tension for each compound in N/m.
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Tvec = atleast_1d(T).to("K").magnitude
    Tc = fuel.get_property("gani", "Tc").to("K").magnitude[:, np.newaxis]
    Pc = fuel.get_property("gani", "Pc").to("bar").magnitude[:, np.newaxis]
    Tb = fuel.get_property("gani", "Tb").to("K").magnitude[:, np.newaxis]
    omega = fuel.get_property("gani", "omega").magnitude[:, np.newaxis]
    Tr = Tvec / Tc

    if correlation.lower() == "brock-bird":
        Tbr = Tb / Tc
        Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
    elif correlation.lower() == "pitzer":
        w = omega
        Q = (
            (1.86 + 1.18 * w)
            / 19.05
            * (np.power((3.75 + 0.91 * w) / (0.291 - 0.08 * w), 2.0 / 3.0))
        )

    st = (
        np.power(Pc, 2.0 / 3.0)
        * np.power(Tc, 1.0 / 3.0)
        * Q
        * np.power(1 - Tr, 11.0 / 9.0)
    )
    return Units.Quantity(st, "dyne/cm").to("N/m")


def thermal_conductivity(
    fuel: "Fuel", T: types.Quantity0D | types.Quantity1D
) -> types.Quantity1D | types.Quantity2D:
    """Compute the thermal conductivity for each compound at given temperature(s).

    :param fuel: Fuel object.
    :type fuel: Fuel
    :param T: Temperature(s) at which to compute the thermal conductivity.
    :type T: types.Quantity0D | types.Quantity1D
    :return: Thermal conductivity for each compound in W/(m*K).
    :rtype: types.Quantity1D | types.Quantity2D
    """
    Tvec = atleast_1d(T).to("K").magnitude
    Tc = fuel.get_property("gani", "Tc").to("K").magnitude[:, np.newaxis]
    Tb = fuel.get_property("gani", "Tb").to("K").magnitude[:, np.newaxis]
    MW = fuel.MW.to("kg/mol").magnitude[:, np.newaxis] * 1e3
    Tr = Tvec / Tc
    fam = np.array(fuel.fam)[:, np.newaxis]

    Astar = np.where(
        fam == 1,
        0.0346,
        np.where(
            fam == 2,
            0.0310,
            np.where(
                fam == 3,
                0.0361,
                0.00350,
            ),
        ),
    )
    beta = np.where(
        fam == 1,
        1.0,
        np.where(
            fam == 2,
            1.0,
            np.where(
                fam == 3,
                1.0,
                0.5,
            ),
        ),
    )
    A = Astar * np.power(Tb, 1.2) / (np.power(MW, beta) * np.power(Tc, 0.167))
    conductivity = A * np.power(1 - Tr, 0.38) / np.power(Tr, 1.0 / 6.0)
    return Units.Quantity(conductivity, "W/(m*K)")
