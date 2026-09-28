"""Fuel class for managing fuel properties."""

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pyparsing as pp
from rdkit.Chem import Mol

from .correlation import components, mixture
from .data import (
    DEFAULT_FUELDATA_DIR,
    resolve_file_path,
    validate_decomp_name,
    validate_fuel_data_dir,
)
from .gcm import GCMRegistry
from .rdk import mol
from .utils import Units, logger, types


class Fuel:
    """Class for handling group contribution calculations.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name.
        Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored.
        If None, uses built-in embedded data.
    :type fuelDataDir: str, optional
    """

    #: Name of the fuel/mixture
    name: str
    #: Name of the decomposition file if different from name
    decompName: str | None = None
    #: Directory where the fuel data is stored
    fuelDataDir: Path = DEFAULT_FUELDATA_DIR

    def __init__(
        self,
        name: str,
        decompName: str | None = None,
        fuelDataDir: str | Path = DEFAULT_FUELDATA_DIR,
    ):
        """Initialize the Fuel object with parsed parameters.

        :param name: Name of the mixture as it appears in its gcData file.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name.
            Defaults to None.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored. If None, uses
            built-in embedded data.
        :type fuelDataDir: str, optional
        """
        logger.info("Initializing Fuel object with name: %s", name)

        self.name = name
        self.fuelDataDir = validate_fuel_data_dir(fuelDataDir)
        self.decompName = validate_decomp_name(self.fuelDataDir, decompName or name)

        logger.info(
            "Legacy property calls (e.g., fuel.Tc) set using the Gani GCM method.\nTo "
            "ensure forward compatibility, use "
            "``fuel.get_property(method_name, property_name)``."
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # (Mostly) Legacy functionalities for backwards compatibility                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Parsed data
    @property
    def gcxgc_data(self) -> pd.DataFrame:
        """Load and return the gcxgc data as a pandas DataFrame."""
        gcxgc_file = resolve_file_path(
            self.fuelDataDir / "gcData",
            old_name=f"{self.name}_init.csv",
            new_name=f"{self.name}.gcxgc.csv",
        )
        return pd.read_csv(gcxgc_file)

    @property
    def gcxgc_bins(self) -> list[tuple[str, int, str]]:
        """Hydrocarbon bins for the compounds. Shape: (num_compounds,)

        :return: List of tuples containing the bin name, carbon number, and family for
        each compound.
        :rtype: list[tuple[str, int, str]]
        """
        if "Bin" in self.gcxgc_data.columns:
            bins = self.gcxgc_data["Bin"].to_list()
        elif "Compound" in self.gcxgc_data.columns:
            logger.info(
                "Hydrocarbon bin information missing; using compound names as "
                "fallback.\nConsider renaming `Compound` column to `Bin` if appropriate."
            )
            bins = self.gcxgc_data["Compound"].to_list()
        else:
            msg = (
                "Hydrocarbon bin information missing and no compound names available "
                "in gc_data."
            )
            raise ValueError(msg)
        carbon_parser = pp.Suppress("C") + pp.Word(pp.nums).set_results_name(
            "carbon_number"
        )
        family_parser = pp.Word(pp.alphanums).set_results_name("family")
        parser = (carbon_parser + pp.Suppress("-") + family_parser) | (
            family_parser + pp.Suppress("-") + carbon_parser
        )

        def split_bin(bin_name: str) -> tuple[str, int, str]:
            parsed = parser.parse_string(bin_name)
            return bin_name, int(parsed.carbon_number), parsed.family.lower()

        return [split_bin(bin_name) for bin_name in bins]

    @property
    def compounds(self) -> list[str]:
        """Alias of gcxgc_bins for the compound (bin) names."""
        return [bin_name for bin_name, _, _ in self.gcxgc_bins]

    @property
    def hc_type(self) -> list[str]:
        """Alias of gcxgc_bins for the hydrocarbon family types."""
        return [family for _, _, family in self.gcxgc_bins]

    @property
    def fam(self) -> list[int]:
        Nij = self.gani_decomp.to_numpy()
        fam = []
        aromatics = 10  # starting index for aromatic groups
        num_aromatics = 5
        branching = 78  # starting index for branching groups (Group j (CH3)2CH through C(CH3)2C(CH3)2)
        num_branching = 5  # groups 78-82 inclusive
        cyclos = 83  # starting index for membered ring groups (3-7 membered rings)
        num_cyclos = 5
        olefins = 4  # starting index for double bound groups
        num_olefins = 6

        for i in range(self.num_compounds):
            # Check if aromatic: does it contain AC's?
            if sum(Nij[i, aromatics : aromatics + num_aromatics]) > 0:
                fam.append(1)
            # Check if cycloparaffin: does it contain rings?
            elif sum(Nij[i, cyclos : cyclos + num_cyclos]) > 0:
                fam.append(2)
            # Check if olefin: does it contain double bonds?
            elif sum(Nij[i, olefins : olefins + num_olefins]) > 0:
                fam.append(3)
            # Check for branching groups (CH, C quaternary carbons)
            elif sum(Nij[i, branching : branching + num_branching]) > 0:
                fam.append(0)
            else:
                # Only CH3 and CH2 -> n-alkane (linear)
                fam.append(0)

        return fam

    """    @property
    def fam(self) -> list[int]:
        ""Family classification for each compound.

        Classification mapping:
            Aromatic     : 1
            Cyclo-alkane : 2
            Alkene       : 3
            Otherwise    : 0
        ""
        fam = []
        for hc in self.hc_type:
            if "aromatic" in hc.lower():
                fam.append(1)
            elif "cyclo" in hc.lower():
                fam.append(2)
            elif "alkene" in hc.lower():
                fam.append(3)
            else:
                fam.append(0)
        return fam"""

    @property
    def smiles(self) -> list[str]:
        """Get the list of SMILES strings for the compounds. Shape: (num_compounds,)"""
        if "SMILES" in self.gcxgc_data.columns:
            return self.gcxgc_data["SMILES"].to_list()
        msg = "Required 'SMILES' column missing in gc_data."
        raise ValueError(msg)

    @property
    def Y_0(self) -> types.Array1D:
        """Normalized initial mass fraction (Y_0) for each compound. Shape: (num_compounds,)"""
        if "Weight %" in self.gcxgc_data.columns:
            wts = self.gcxgc_data["Weight %"].to_numpy().flatten().astype(float)
            return wts / wts.sum()
        msg = "Initial mass fractions (Y_0) not available: 'Weight %' column missing in gc_data."
        raise ValueError(msg)

    # GCM (Group Contribution Method) properties for the compounds
    @property
    def gani_decomp(self) -> pd.DataFrame:
        """Gani decomposition for the compounds. Shape: (num_compounds, num_groups)"""
        groupDecompFile = resolve_file_path(
            self.fuelDataDir / "groupDecompositionData",
            f"{self.decompName}.csv",
            f"{self.decompName}.gani.csv",
        )
        if not Path(groupDecompFile).exists():
            msg = f"Gani decomposition file not found: {groupDecompFile}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(groupDecompFile, header=0, index_col=0)
        missing = set(self.compounds) - set(df.index)
        if missing:
            msg = (
                f"Gani decomposition file ({groupDecompFile}) is missing compounds "
                f"present in the fuel mixture: {sorted(missing)}."
            )
            raise ValueError(msg)
        return df.loc[self.compounds]

    @cached_property
    def gcm_properties(self) -> dict[str, dict[str, types.Quantity1D]]:
        """Pre-computed GCM properties for the compounds. Shape: (num_properties, num_compounds)"""
        props: dict[str, dict[str, types.Quantity1D]] = {}
        for gcm in GCMRegistry.methods:
            props.update(gcm.predict_all(self))
        return props

    def get_property(self, method: str, property_name: str) -> types.Quantity1D:
        """
        Get a specific property from the GCM for each compound.

        :param method: The GCM method to use.
        :type method: str
        :param property_name: The name of the property to retrieve.
        :type property_name: str
        :return: Array of the requested property for each compound.
        :rtype: types.Quantity1D
        """
        method = method.lower()
        if method not in self.gcm_properties:
            msg = f"Method '{method}' not found in computed GCM properties."
            raise KeyError(msg)

        property_name = property_name.lower()
        if property_name not in self.gcm_properties[method]:
            msg = f"Property '{property_name}' not found in computed GCM properties."
            raise KeyError(msg)

        return self.gcm_properties[method][property_name]

    # Component classification and informatics
    @property
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit Mol objects for the compounds. Shape: (n_compounds,)"""
        return [mol.from_smiles(smiles) for smiles in self.smiles]

    @property
    def atom_counts(self) -> list[dict[str, int]]:
        """Atom counts for each compound. Shape: (num_compounds,)"""
        return [mol.atom_counts(mol) for mol in self.rdkit_mols]

    @property
    def nC(self) -> types.Array1D:
        """Number of carbon atoms for each compound. Shape: (num_compounds,)"""
        return np.array([c.get("C", 0) for c in self.atom_counts])

    @property
    def nH(self) -> types.Array1D:
        """Number of hydrogen atoms for each compound. Shape: (num_compounds,)"""
        return np.array([c.get("H", 0) for c in self.atom_counts])

    ## Legacy Fuel attributes for backward compatibility
    @cached_property
    def MW(self) -> types.Quantity1D:
        """Molecular weight for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "MW").to("kg/mol")
        mw = [mol.molecular_weight(m) for m in self.rdkit_mols]
        return Units.Quantity(mw, "g/mol").to("kg/mol")

    @property
    def Tc(self) -> types.Quantity1D:
        """Gani critical temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Tc").to("K")

    @property
    def Pc(self) -> types.Quantity1D:
        """Gani critical pressure (Pa) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Pc").to("Pa")

    @property
    def Vc(self) -> types.Quantity1D:
        """Gani critical volume (m^3/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Vc").to("m^3/mol")

    @property
    def Tb(self) -> types.Quantity1D:
        """Gani boiling temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Tb").to("K")

    @property
    def Tm(self) -> types.Quantity1D:
        """Gani melting temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Tm").to("K")

    @property
    def Hf(self) -> types.Quantity1D:
        """Gani enthalpy of formation (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Hf").to("J/mol")

    @property
    def Gf(self) -> types.Quantity1D:
        """Gani Gibbs free energy of formation (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Gf").to("J/mol")

    @property
    def Hv_stp(self) -> types.Quantity1D:
        """Gani enthalpy of vaporization at STP (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Hv_stp").to("J/mol")

    @property
    def omega(self) -> types.Quantity1D:
        """Gani acentric factor (dimensionless) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "omega").to("")

    @property
    def Vm_stp(self) -> types.Quantity1D:
        """Gani molar volume at STP (m^3/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Vm_stp").to("m^3/mol")

    @property
    def Cp_stp(self) -> types.Quantity1D:
        """Gani heat capacity at STP (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Cp_stp").to("J/mol/K")

    @property
    def Cp_B(self) -> types.Quantity1D:
        """Gani heat capacity correction (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Cp_B").to("J/mol/K")

    @property
    def Cp_C(self) -> types.Quantity1D:
        """Gani heat capacity correction (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_property("gani", "Cp_C").to("J/mol/K")

    @property
    def Lv_stp(self) -> types.Quantity1D:
        """Standard latent heat of vaporization for each compound. Shape: (n_compounds,)"""
        return components.latent_heat_vaporization_stp(self)

    @property
    def epsilonByKB(self) -> types.Quantity1D:
        """Epsilon divided by Boltzmann constant (K) for each compound. Shape: (n_compounds,)"""
        return components.epsilon_by_kb(self)

    @property
    def sigma(self) -> types.Quantity1D:
        """Sigma parameter (m) for each compound. Shape: (n_compounds,)"""
        return components.sigma(self)

    def molar_liquid_vol(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Molar liquid volume for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.molar_liquid_vol(self, T)

    def density(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Density for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.density(self, T)

    def viscosity_kinematic(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Kinematic viscosity for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.kinematic_viscosity(self, T)

    def viscosity_dynamic(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Dynamic viscosity for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.dynamic_viscosity(self, T)

    def Cp(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Molar specific heat capacity for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.molar_specific_heat_capacity(self, T)

    def Cl(
        self, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity1D | types.Quantity2D:
        """Liquid mass specific heat capacity for each compound at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return components.liquid_mass_specific_heat_capacity(self, T)

    def mixture_density(
        self, Yi: types.Array1D, T: types.Quantity0D | types.Quantity1D
    ) -> types.Quantity0D | types.Quantity1D:
        """Density for the mixture at a given temperature. Shape: (n_compounds,) or (n_compounds, n_temperatures)"""
        return mixture.density(self, Yi, T)
