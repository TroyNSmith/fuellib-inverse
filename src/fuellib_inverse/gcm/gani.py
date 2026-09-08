"""Constantinou-Gani Group Contribution Method (GCM) for fuel components."""

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
import quaxed.numpy as qnp
import unxt as u
from unxt import AbstractQuantity

from ..utils.units import convert_temperature

if TYPE_CHECKING:
    from ..fuel import Fuel


class GaniGCM:
    """Constantinou-Gani Group Contribution Method (GCM) for fuel components."""

    _csv_path: Path = Path(__file__).parent / "gani.csv"

    groups: npt.NDArray[np.str_]  # Group names
    smarts: npt.NDArray[np.str_]  # Group SMARTS patterns
    orders: npt.NDArray[np.int_]  # Group orders
    types: npt.NDArray[np.str_]  # Group types (e.g., "alkane", "aromatic", etc.)
    # Coefficients for the Gani method (from gani.csv)
    _tck: npt.NDArray[np.float64]
    _pck: npt.NDArray[np.float64]
    _vck: npt.NDArray[np.float64]
    _tbk: npt.NDArray[np.float64]
    _tmk: npt.NDArray[np.float64]
    _hfk: npt.NDArray[np.float64]
    _gfk: npt.NDArray[np.float64]
    _hvk: npt.NDArray[np.float64]
    _wk: npt.NDArray[np.float64]
    _vmk: npt.NDArray[np.float64]
    _CpAk: npt.NDArray[np.float64]
    _CpBk: npt.NDArray[np.float64]
    _CpCk: npt.NDArray[np.float64]

    def __init__(self):
        df = pd.read_csv(self._csv_path, skipinitialspace=True, index_col=0)
        self.groups = df.columns.to_numpy(dtype=str)
        self.smarts = df.loc["smarts"].to_numpy(dtype=str)
        self.orders = df.loc["order"].to_numpy(dtype=int)
        self.types = df.loc["type"].to_numpy(dtype=str)
        self._tck = df.loc["tck"].to_numpy(dtype=float)
        self._pck = df.loc["pck"].to_numpy(dtype=float)
        self._vck = df.loc["vck"].to_numpy(dtype=float)
        self._tbk = df.loc["tbk"].to_numpy(dtype=float)
        self._tmk = df.loc["tmk"].to_numpy(dtype=float)
        self._hfk = df.loc["hfk"].to_numpy(dtype=float)
        self._gfk = df.loc["gfk"].to_numpy(dtype=float)
        self._hvk = df.loc["hvk"].to_numpy(dtype=float)
        self._wk = df.loc["wk"].to_numpy(dtype=float)
        self._vmk = df.loc["vmk"].to_numpy(dtype=float)
        self._CpAk = df.loc["CpAk"].to_numpy(dtype=float)
        self._CpBk = df.loc["CpBk"].to_numpy(dtype=float)
        self._CpCk = df.loc["CpCk"].to_numpy(dtype=float)

    def load_fuel_decomposition(self, csv_path: str | Path) -> npt.NDArray[np.int_]:
        """Load the group decomposition of a fuel from a CSV file.

        The CSV columns may equal ``self.groups`` or be a subset of them, in
        any order, but must not contain groups outside ``self.groups``. The
        returned decomposition is reordered to match ``self.groups``, with
        zeros filled in for any groups missing from the CSV.

        :param csv_path: Path to the CSV file containing the group decomposition.
        :type csv_path: str or Path
        :return: Compound decompositions.
        :rtype: npt.NDArray[np.int_]
        :raises ValueError: If the CSV file does not contain any compounds, or
            contains groups not in ``self.groups``.
        """
        df = pd.read_csv(csv_path, skipinitialspace=True, index_col=0)
        compounds = df.index.tolist()
        if len(compounds) < 1:
            raise ValueError(f"{csv_path} must contain at least one compound.")

        csv_groups = df.columns.to_numpy(dtype=str)
        unknown_groups = set(csv_groups) - set(self.groups)
        if unknown_groups:
            raise ValueError(
                f"{csv_path} contains groups not in self.groups: "
                f"{sorted(unknown_groups)}"
            )

        decomp = np.zeros((len(compounds), len(self.groups)), dtype=int)
        csv_decomp = df.to_numpy(dtype=int)
        group_indices = {group: i for i, group in enumerate(self.groups)}
        for csv_col, group in enumerate(csv_groups):
            decomp[:, group_indices[group]] = csv_decomp[:, csv_col]

        return decomp

    def Tc(self, fuel: "Fuel", *, unit: str = "K") -> AbstractQuantity:
        """Calculate the critical temperature of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the critical temperature.
        :type fuel: Fuel
        :param unit: The unit for the returned critical temperature. Default is "K".
        :type unit: str, optional
        :return: The critical temperatures in the specified unit.
        :rtype: AbstractQuantity
        """
        tc0 = u.Quantity(181.128, "K")
        tc = qnp.log(qnp.matmul(fuel.gani_decomp, self._tck))
        return convert_temperature((tc0 * tc), unit)

    def Pc(self, fuel: "Fuel", *, unit: str = "bar") -> AbstractQuantity:
        """Calculate the critical pressure of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the critical pressure.
        :type fuel: Fuel
        :param unit: The unit for the returned critical pressure. Default is "bar".
        :type unit: str, optional
        :return: The critical pressures in the specified unit.
        :rtype: AbstractQuantity
        """
        pc1 = u.Quantity(1.3705, "bar")
        pc2 = u.Quantity(0.10022, "bar^(-0.5)")
        pc = u.Quantity(jnp.matmul(fuel.gani_decomp, self._pck), "bar^(-0.5)")
        return (pc1 + qnp.power(pc + pc2, -2)).to(unit)

    def Vc(self, fuel: "Fuel", *, unit: str = "m^3/kmol") -> AbstractQuantity:
        """Calculate the critical volume of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the critical volume.
        :param unit: The unit for the returned critical volume. Default is "m^3/kmol".
        :type unit: str, optional
        :type fuel: Fuel
        :return: The critical volumes in the specified unit.
        :rtype: AbstractQuantity
        """
        vc0 = u.Quantity(-0.00435, "m^3/kmol")
        vc = u.Quantity(jnp.matmul(fuel.gani_decomp, self._vck), "m^3/kmol")
        return (vc0 + vc).to(unit)

    def Tb(self, fuel: "Fuel", *, unit: str = "K") -> AbstractQuantity:
        """Calculate the normal boiling point of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the normal boiling point.
        :type fuel: Fuel
        :param unit: The unit for the returned normal boiling point. Default is "K".
        :type unit: str, optional
        :return: The normal boiling points in the specified unit.
        :rtype: AbstractQuantity
        """
        tb0 = u.Quantity(204.359, "K")
        tb = jnp.log(jnp.matmul(fuel.gani_decomp, self._tbk))
        return convert_temperature((tb0 * tb), unit)

    def Tm(self, fuel: "Fuel", *, unit: str = "K") -> AbstractQuantity:
        """Calculate the melting point of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the melting point.
        :type fuel: Fuel
        :param unit: The unit for the returned melting point. Default is "K".
        :type unit: str, optional
        :return: The melting points in the specified unit.
        :rtype: AbstractQuantity
        """
        tm0 = u.Quantity(102.425, "K")
        tm = jnp.log(jnp.matmul(fuel.gani_decomp, self._tmk))
        return convert_temperature((tm0 * tm), unit)

    def Hf(self, fuel: "Fuel", *, unit: str = "kJ/mol") -> AbstractQuantity:
        """Calculate the heat of formation of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the heat of formation.
        :type fuel: Fuel
        :param unit: The unit for the returned heat of formation. Default is "kJ/mol".
        :type unit: str, optional
        :return: The heats of formation in the specified unit.
        :rtype: AbstractQuantity
        """
        hf0 = u.Quantity(10.835, "kJ/mol")
        hf = u.Quantity(jnp.matmul(fuel.gani_decomp, self._hfk), "kJ/mol")
        return (hf0 + hf).to(unit)

    def Gf(self, fuel: "Fuel", *, unit: str = "kJ/mol") -> AbstractQuantity:
        """Calculate the Gibbs free energy of formation of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the Gibbs free energy of formation.
        :type fuel: Fuel
        :param unit: The unit for the returned Gibbs free energy of formation. Default is "kJ/mol".
        :type unit: str, optional
        :return: The Gibbs free energies of formation in the specified unit.
        :rtype: AbstractQuantity
        """
        gf0 = u.Quantity(-14.828, "kJ/mol")
        gf = u.Quantity(jnp.matmul(fuel.gani_decomp, self._gfk), "kJ/mol")
        return (gf0 + gf).to(unit)

    def Hv_stp(self, fuel: "Fuel", *, unit: str = "kJ/mol") -> AbstractQuantity:
        """Calculate the heat of vaporization of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the heat of vaporization.
        :type fuel: Fuel
        :param unit: The unit for the returned heat of vaporization. Default is "kJ/mol".
        :type unit: str, optional
        :return: The heats of vaporization in the specified unit.
        :rtype: AbstractQuantity
        """
        hv0 = u.Quantity(6.829, "kJ/mol")
        hv = u.Quantity(jnp.matmul(fuel.gani_decomp, self._hvk), "kJ/mol")
        return (hv0 + hv).to(unit)

    def Cp_stp(self, fuel: "Fuel", *, unit: str = "J/(mol*K)") -> AbstractQuantity:
        """Calculate the heat capacity at STP of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the heat capacity at STP.
        :type fuel: Fuel
        :param unit: The unit for the returned heat capacity at STP. Default is "J/(mol*K)".
        :type unit: str, optional
        :return: The heat capacities at STP in the specified unit.
        :rtype: AbstractQuantity
        """
        cp0 = u.Quantity(19.7779, "J/(mol*K)")
        cp = u.Quantity(jnp.matmul(fuel.gani_decomp, self._CpAk), "J/(mol*K)")
        return (cp - cp0).to(unit)

    def Cp_B(self, fuel: "Fuel", *, unit: str = "J/(mol*K)") -> AbstractQuantity:
        """Calculate the temperature correction B for heat capacity of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the temperature correction for heat capacity.
        :type fuel: Fuel
        :param unit: The unit for the returned temperature correction for heat capacity. Default is "J/(mol*K)".
        :type unit: str, optional
        :return: The temperature corrections for heat capacity in the specified unit.
        :rtype: AbstractQuantity
        """
        cp_b = u.Quantity(jnp.matmul(fuel.gani_decomp, self._CpBk), "J/(mol*K)")
        return cp_b.to(unit)

    def Cp_C(self, fuel: "Fuel", *, unit: str = "J/(mol*K)") -> AbstractQuantity:
        """Calculate the temperature correction C for heat capacity of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the temperature correction for heat capacity.
        :type fuel: Fuel
        :param unit: The unit for the returned temperature correction for heat capacity. Default is "J/(mol*K)".
        :type unit: str, optional
        :return: The temperature corrections for heat capacity in the specified unit.
        :rtype: AbstractQuantity
        """
        cp_c = u.Quantity(jnp.matmul(fuel.gani_decomp, self._CpCk), "J/(mol*K)")
        return cp_c.to(unit)

    def Vm_stp(self, fuel: "Fuel", *, unit: str = "m^3/kmol") -> AbstractQuantity:
        """Calculate the molar liquid volume at STP of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the molar liquid volume at STP.
        :type fuel: Fuel
        :param unit: The unit for the returned molar liquid volume at STP. Default is "m^3/kmol".
        :type unit: str, optional
        :return: The molar liquid volumes at STP in the specified unit.
        :rtype: AbstractQuantity
        """
        vm0 = u.Quantity(0.01211, "m^3/kmol")
        vm = u.Quantity(jnp.matmul(fuel.gani_decomp, self._vmk), "m^3/kmol")
        return (vm0 + vm).to(unit)

    def omega(self, fuel: "Fuel") -> AbstractQuantity:
        """Calculate the acentric factor of fuel compounds using the Gani method.

        :param fuel: The fuel object for which to calculate the acentric factor.
        :type fuel: Fuel
        :return: The acentric factors.
        :rtype: AbstractQuantity
        """
        w0 = 0.4085
        w1 = 1.1507
        w2 = 1.0 / 0.5050
        wk = jnp.matmul(fuel.gani_decomp, self._wk)
        return u.Quantity(w0 * jnp.power(jnp.log(wk + w1), w2), "")
