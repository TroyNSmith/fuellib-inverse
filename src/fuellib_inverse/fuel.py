"""Fuel object."""

from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from .gcm.gani import GaniGCM
from .rd import mol
from .utils.units import convert_temperature

gani_gcm = GaniGCM()  # Initialize the Gani group contribution method

DEFAULT_FUEL_DIR = Path(__file__).parent / "data" / "fuelData"


def _check_valid_property(
    value: Array | AbstractQuantity, expected_length: int, name: str
) -> None:
    """
    Check if the length of the array matches the expected length.

    :param value: Array to check.
    :type value: Array | AbstractQuantity
    :param expected_length: Expected length of the array.
    :type expected_length: int
    :param name: Name of the variable for error messages.
    :type name: str
    :raises ValueError: If the array is not 1D.
    :raises ValueError: If the length of the array does not match the expected length.
    """
    value = qnp.array(value.value) if isinstance(value, AbstractQuantity) else value
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if value.shape[0] != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}, but has length {value.shape[0]}."
        )


def _atleast_col(x: AbstractQuantity) -> AbstractQuantity:
    """
    Add a trailing axis so an array of temperatures broadcasts against a per-compound axis; scalars pass through.

    Idempotent: only 1D arrays are expanded, so calling this again on an already-expanded
    array (e.g. because a caller already column-expanded T before passing it to another
    method that also calls this) is a no-op instead of adding another axis.

    :param x: Input array.
    :type x: Array
    :return: Array with at least one trailing axis.
    :rtype: Array
    """
    return x[:, None] if qnp.ndim(x) == 1 else x


class Fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.
    """

    name: str
    families: list[str]
    smiles: list[str]
    ref_compounds: list[str] | None

    gani_decomp: npt.NDArray[np.int_]

    # Critical properties of the fuel components
    _Y_0: Array  # Mass fractions of the compounds in the mixture
    _Tc: AbstractQuantity  # Critical temperatures in K
    _Pc: AbstractQuantity  # Critical pressures in Pa
    _Vc: AbstractQuantity  # Critical volumes in m^3/mol
    _Tb: AbstractQuantity  # Boiling points in K
    _Tm: AbstractQuantity  # Melting points in K
    _Hf: AbstractQuantity  # Heat of formation in J/mol
    _Gf: AbstractQuantity  # Gibbs free energy in J/mol
    _Hv_stp: AbstractQuantity  # Heat of vaporization at STP in J/mol
    _Cp_stp: AbstractQuantity  # Heat capacity at STP in J/mol/K
    _Cp_B: AbstractQuantity  # Temperature correction for heat capacity in J/mol/K
    _Cp_C: AbstractQuantity  # Temperature correction for heat capacity in J/mol/K
    _Vm_stp: AbstractQuantity  # Molar volume at STP in m^3/mol
    _omega: AbstractQuantity  # Acentric factor (dimensionless)
    # Derived properties
    _Lv_stp: AbstractQuantity  # Latent heat of vaporization at STP in J/mol
    _sigma: AbstractQuantity  # Surface tension in N/m

    def __init__(
        self,
        name: str,
        *,
        decompName: str | None = None,
        fuelDataDir: str | Path = DEFAULT_FUEL_DIR,
    ) -> None:
        """
        Initialize a Fuel object.

        :param name: Name of the fuel.
        :type name: str
        :param decompName: Name of the decomposition file (optional).
            Defaults to `{fuel}.{method}.csv` if not provided.
        :type decompName: str | None
        :param fuelDataDir: Directory containing fuel data files (optional).
            Defaults to `src/fuellib_inverse/data/fuelData` if not provided.
        :type fuelDataDir: str | Path | None

        """
        fuelDataDir = Path(fuelDataDir)
        self.name = name
        self.families = []
        self.smiles = []

        # Load GCxGC data for the fuel
        gcxgc_path = fuelDataDir / "gcData" / f"{name}.csv"
        df = pd.read_csv(gcxgc_path, skipinitialspace=True)
        required_cols = ["Family", "SMILES", "Weight %"]
        if not all(col in df.columns for col in required_cols):
            msg = f"Missing required columns in GCxGC data for fuel '{name}'. Required columns: {required_cols}"
            raise ValueError(msg)

        ## Parse the GCxGC data and store it in the Fuel object
        self.families = df["Family"].tolist()
        self.smiles = df["SMILES"].tolist()
        self.ref_compounds = (
            df["Reference Compound"].tolist()
            if "Reference Compound" in df.columns
            else None
        )

        wts = df["Weight %"].to_numpy(dtype=float)
        self._Y_0 = wts / np.sum(wts)  # Normalize weights to get mass fractions

        # Load group contribution decomposition data for the fuel
        gani_decomp_path = (
            fuelDataDir / "groupDecompositionData" / f"{name}.gani.csv"
            if decompName is None
            else fuelDataDir / "groupDecompositionData" / decompName
        )
        self.gani_decomp = gani_gcm.load_fuel_decomposition(gani_decomp_path)

        ## Initialize critical properties of the fuel components
        self._Tc = gani_gcm.Tc(self, unit="K")
        self._Pc = gani_gcm.Pc(self, unit="Pa")
        self._Vc = gani_gcm.Vc(self, unit="m^3/mol")
        self._Tb = gani_gcm.Tb(self, unit="K")
        self._Tm = gani_gcm.Tm(self, unit="K")
        self._Hf = gani_gcm.Hf(self, unit="J/mol")
        self._Gf = gani_gcm.Gf(self, unit="J/mol")
        self._Hv_stp = gani_gcm.Hv_stp(self, unit="J/mol")
        self._Cp_stp = gani_gcm.Cp_stp(self, unit="J/(mol*K)")
        self._Cp_B = gani_gcm.Cp_B(self, unit="J/(mol*K)")
        self._Cp_C = gani_gcm.Cp_C(self, unit="J/(mol*K)")
        self._Vm_stp = gani_gcm.Vm_stp(self, unit="m^3/mol")
        self._omega = gani_gcm.omega(self)

    # Basic properties of the mixture
    @property
    def num_compounds(self) -> int:
        """
        Return the number of compounds in the mixture.

        :return: Number of compounds.
        :rtype: int
        """
        return self.gani_decomp.shape[0]

    @cached_property
    def rdkit_mols(self) -> list[mol.Mol]:
        """
        Return a list of RDKit molecule objects for each component in the mixture.

        :return: List of RDKit molecule objects.
        :rtype: list[mol.Mol]
        """
        return [mol.from_smiles(s) for s in self.smiles]

    @cached_property
    def formulas(self) -> list[str]:
        """
        Return the chemical formulas of each component in the mixture.

        :return: Chemical formulas of each component.
        :rtype: list[str]
        """
        return [mol.hill_formula(m) for m in self.rdkit_mols]

    @cached_property
    def MW(self) -> AbstractQuantity:
        """
        Return the molecular weights of each component in the mixture.

        :return: Molecular weights of each component.
        :rtype: AbstractQuantity
        """
        masses = qnp.array([mol.molecular_weight(m) for m in self.rdkit_mols])
        return Quantity(masses, "g/mol")

    @cached_property
    def num_carbons(self) -> npt.NDArray[np.int_]:
        """
        Return the number of carbon atoms in each component of the mixture.

        :return: Number of carbon atoms.
        :rtype: npt.NDArray[np.int_]
        """
        return np.array([mol.count_element(m, "C") for m in self.rdkit_mols])

    @cached_property
    def num_hydrogens(self) -> npt.NDArray[np.int_]:
        """
        Return the number of hydrogen atoms in each component of the mixture.

        :return: Number of hydrogen atoms in each component of the mixture.
        :rtype: npt.NDArray[np.int_]
        """
        return np.array([mol.count_element(m, "H") for m in self.rdkit_mols])

    @cached_property
    def _hydrocarbon(self) -> npt.NDArray[np.bool_]:
        """
        Return a list indicating whether each component of the mixture is a hydrocarbon.

        :return: List of booleans indicating whether each component is a hydrocarbon.
        :rtype: npt.NDArray[np.bool_]
        """
        return np.array([mol.is_hydrocarbon(m) for m in self.rdkit_mols])

    @cached_property
    def _aromatic(self) -> npt.NDArray[np.bool_]:
        """
        Return a list indicating whether each component of the mixture is aromatic.

        :return: List of booleans indicating aromaticity of each component.
        :rtype: npt.NDArray[np.bool_]
        """
        return np.array([mol.has_aromatic(m) for m in self.rdkit_mols])

    @cached_property
    def _cyclic(self) -> npt.NDArray[np.bool_]:
        """
        Return a list indicating whether each component of the mixture is cyclic.

        :return: List of booleans indicating cyclicity of each component.
        :rtype: npt.NDArray[np.bool_]
        """
        return np.array([mol.has_ring(m) for m in self.rdkit_mols])

    @cached_property
    def _branched(self) -> npt.NDArray[np.bool_]:
        """
        Return a list indicating whether each component of the mixture is branched.

        :return: List of booleans indicating branching of each component.
        :rtype: npt.NDArray[np.bool_]
        """
        return np.array([mol.has_branch(m) for m in self.rdkit_mols])

    @cached_property
    def _alkene(self) -> npt.NDArray[np.bool_]:
        """
        Return a list indicating whether each component of the mixture is an alkene.

        :return: List of booleans indicating whether each component is an alkene.
        :rtype: npt.NDArray[np.bool_]
        """
        return np.array([mol.has_alkene_bond(m) for m in self.rdkit_mols])

    @cached_property
    def hydrocarbon_types(self) -> npt.NDArray[np.str_]:
        """
        Return a list indicating the hydrocarbon type of each component of the mixture.

        :return: List of hydrocarbon types for each component.
        :rtype: npt.NDArray[np.str_]
        """
        if not all(self._hydrocarbon):
            raise NotImplementedError(
                "hydrocarbon_types property is only available for mixtures where all components are hydrocarbons."
            )

        # "0123456789" sets the string length to 10, which is the max length of the hydrocarbon type strings
        # Array filled in descending order:
        # n-alkane < iso-alkane < alkene < cyclic < aromatic
        hc_types = np.array(["0123456789"] * self.num_compounds, dtype=np.str_)
        hc_types[:] = "n-alkane"  # Set default to n-alkane
        hc_types[self._branched] = "iso-alkane"
        hc_types[self._alkene] = "alkene"
        hc_types[self._cyclic] = "cyclic"
        hc_types[self._aromatic] = "aromatic"
        return hc_types

    def molar_liquid_vol(
        self, T: AbstractQuantity, *, unit: str = "m^3/mol"
    ) -> Quantity:
        """
        Calculate the molar liquid volume of fuel compounds over a range of temperatures.

        :param T: Temperature at which to calculate the molar liquid volume.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "m^3/mol".
        :type unit: str
        :return: Molar liquid volume of fuel compounds at the specified temperature.
        :rtype: Quantity
        """
        Tstp = Quantity(298.15, "K")

        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")

        # Strip units from T and broadcast for comparison
        condition = qnp.array(T.value) > qnp.array(Tc.value)
        x = -qnp.power(1 - (Tstp / Tc), 2.0 / 7.0)
        y = qnp.power(1 - (T / Tc), 2.0 / 7.0) + x

        phi: Array = qnp.where(condition, x, y).value  # ty: ignore[unresolved-attribute]

        z1 = 0.29056
        z2 = 0.08775
        z: Array = (z1 - z2 * self.omega).value  # ty: ignore[invalid-assignment]

        return (self.Vm_stp * qnp.power(z, phi)).to(unit)

    def density(self, T: AbstractQuantity, *, unit: str = "kg/m^3") -> AbstractQuantity:
        """
        Calculate the density of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the density.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "kg/m^3".
        :type unit: str
        :return: Density of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        return (self.MW / self.molar_liquid_vol(T, unit="m^3/mol")).to(unit)

    # Mixture property correlations
    def mean_molecular_weight(self, *, unit: str = "kg/mol") -> AbstractQuantity:
        """
        Calculate the mean molecular weight of the mixture.

        :return: Mean molecular weight of the mixture.
        :rtype: AbstractQuantity
        """
        return qnp.sum(self.MW / self.Y_0).to(unit)

    def mixture_density(
        self, T: AbstractQuantity, *, unit: str = "kg/m^3"
    ) -> AbstractQuantity:
        """
        Calculate the density of the mixture over a range of temperatures.

        :param T: Temperatures at which to calculate the density.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "kg/m^3".
        :type unit: str
        :return: Density of the mixture at the specified temperature.
        :rtype: AbstractQuantity
        """
        # molar_liquid_vol already handles T conversion and broadcasting, so we can skip it here
        return qnp.sum(self.Y_0 * (self.MW / self.molar_liquid_vol(T)), axis=-1).to(
            unit
        )

    # Critical property getters and setters
    @property
    def Y_0(self) -> Array:
        """
        Return the mass fractions of the compounds in the mixture.

        :return: Mass fractions of the compounds.
        :rtype: Array
        """
        return self._Y_0

    @Y_0.setter
    def Y_0(self, value: Array) -> None:
        """
        Set the mass fractions of the compounds in the mixture.

        :param value: Mass fractions of the compounds.
        :type value: Array
        """
        _check_valid_property(value, self.num_compounds, "Y_0")
        if not qnp.isclose(qnp.sum(value), 1.0, atol=5e-2):
            raise ValueError("Y_0 must sum to 1.00 +/- 0.05.")

        self._Y_0 = value

    @property
    def Tc(self) -> AbstractQuantity:
        """
        Return the critical temperatures of the compounds in the mixture.

        :return: Critical temperatures of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tc

    @Tc.setter
    def Tc(self, value: AbstractQuantity) -> None:
        """
        Set the critical temperatures of the compounds in the mixture.

        :param value: Critical temperatures of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tc")
        self._Tc = convert_temperature(value, "K")

    @property
    def Pc(self) -> AbstractQuantity:
        """
        Return the critical pressures of the compounds in the mixture.

        :return: Critical pressures of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Pc

    @Pc.setter
    def Pc(self, value: AbstractQuantity) -> None:
        """
        Set the critical pressures of the compounds in the mixture.

        :param value: Critical pressures of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Pc")
        self._Pc = value.to("Pa")

    @property
    def Vc(self) -> AbstractQuantity:
        """
        Return the critical volumes of the compounds in the mixture.

        :return: Critical volumes of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Vc

    @Vc.setter
    def Vc(self, value: AbstractQuantity) -> None:
        """
        Set the critical volumes of the compounds in the mixture.

        :param value: Critical volumes of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Vc")
        self._Vc = value.to("m^3/mol")

    @property
    def Tb(self) -> AbstractQuantity:
        """
        Return the boiling points of the compounds in the mixture.

        :return: Boiling points of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tb

    @Tb.setter
    def Tb(self, value: AbstractQuantity) -> None:
        """
        Set the boiling points of the compounds in the mixture.

        :param value: Boiling points of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tb")
        self._Tb = convert_temperature(value, "K")

    @property
    def Tm(self) -> AbstractQuantity:
        """
        Return the melting points of the compounds in the mixture.

        :return: Melting points of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tm

    @Tm.setter
    def Tm(self, value: AbstractQuantity) -> None:
        """
        Set the melting points of the compounds in the mixture.

        :param value: Melting points of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tm")
        self._Tm = convert_temperature(value, "K")

    @property
    def Hf(self) -> AbstractQuantity:
        """
        Return the heat of formation of the compounds in the mixture.

        :return: Heat of formation of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Hf

    @Hf.setter
    def Hf(self, value: AbstractQuantity) -> None:
        """
        Set the heat of formation of the compounds in the mixture.

        :param value: Heat of formation of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Hf")
        self._Hf = value.to("J/mol")

    @property
    def Gf(self) -> AbstractQuantity:
        """
        Return the Gibbs free energy of the compounds in the mixture.

        :return: Gibbs free energy of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Gf

    @Gf.setter
    def Gf(self, value: AbstractQuantity) -> None:
        """
        Set the Gibbs free energy of the compounds in the mixture.

        :param value: Gibbs free energy of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Gf")
        self._Gf = value.to("J/mol")

    @property
    def Hv_stp(self) -> AbstractQuantity:
        """
        Return the heat of vaporization at STP of the compounds in the mixture.

        :return: Heat of vaporization at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Hv_stp

    @Hv_stp.setter
    def Hv_stp(self, value: AbstractQuantity) -> None:
        """
        Set the heat of vaporization at STP of the compounds in the mixture.

        :param value: Heat of vaporization at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Hv_stp")
        self._Hv_stp = value.to("J/mol")

    @property
    def Cp_stp(self) -> AbstractQuantity:
        """
        Return the heat capacity at STP of the compounds in the mixture.

        :return: Heat capacity at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_stp

    @Cp_stp.setter
    def Cp_stp(self, value: AbstractQuantity) -> None:
        """
        Set the heat capacity at STP of the compounds in the mixture.

        :param value: Heat capacity at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_stp")
        self._Cp_stp = value.to("J/mol/K")

    @property
    def Cp_B(self) -> AbstractQuantity:
        """
        Return the temperature correction B for heat capacity of the compounds in the mixture.

        :return: Temperature correction B for heat capacity of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_B

    @Cp_B.setter
    def Cp_B(self, value: AbstractQuantity) -> None:
        """
        Set the temperature correction B for heat capacity of the compounds in the mixture.

        :param value: Temperature correction B for heat capacity of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_B")
        self._Cp_B = value.to("J/(mol*K)")

    @property
    def Cp_C(self) -> AbstractQuantity:
        """
        Return the temperature correction C for heat capacity of the compounds in the mixture.

        :return: Temperature correction C for heat capacity of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_C

    @Cp_C.setter
    def Cp_C(self, value: AbstractQuantity) -> None:
        """
        Set the temperature correction C for heat capacity of the compounds in the mixture.

        :param value: Temperature correction C for heat capacity of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_C")
        self._Cp_C = value.to("J/(mol*K)")

    @property
    def Vm_stp(self) -> AbstractQuantity:
        """
        Return the molar volume at STP of the compounds in the mixture.

        :return: Molar volume at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Vm_stp

    @Vm_stp.setter
    def Vm_stp(self, value: AbstractQuantity) -> None:
        """
        Set the molar volume at STP of the compounds in the mixture.

        :param value: Molar volume at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Vm_stp")
        self._Vm_stp = value.to("m^3/mol")

    @property
    def omega(self) -> AbstractQuantity:
        """
        Return the acentric factor of the compounds in the mixture.

        :return: Acentric factor of the compounds.
        :rtype: AbstractQuantity
        """
        return self._omega

    @omega.setter
    def omega(self, value: AbstractQuantity) -> None:
        """
        Set the acentric factor of the compounds in the mixture.

        :param value: Acentric factor of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "omega")
        self._omega = value
