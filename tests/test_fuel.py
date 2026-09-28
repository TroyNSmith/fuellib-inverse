"""Test module for the Fuel module."""

from pathlib import Path

import pytest

from fuellib.fuel import Fuel
from fuellib.utils import resolved_np as np


@pytest.fixture
def heptane() -> Fuel:
    return Fuel(name="heptane")


@pytest.fixture
def custom_path() -> Path:
    return Path(__file__).parent / "data/fuelData"


class TestFuelInitialization:
    """Test the initialization of the Fuel class."""

    def test__default_fueldata_dir(self) -> None:
        """Test the default fueldata directory properly resolves."""
        fuel = Fuel(name="heptane")
        assert fuel.fuelDataDir is not None

    def test__custom_fueldata_dir(self, custom_path: Path) -> None:
        """Test specifying a non-default fueldata directory."""
        fuel = Fuel(name="octane", fuelDataDir=custom_path)
        assert fuel.fuelDataDir == custom_path

    def test__invalid_custom_fueldata_dir(self, custom_path: Path) -> None:
        """Test specifying an invalid custom fueldata directory."""
        invalid_path = custom_path.parent
        with pytest.raises(FileNotFoundError):
            Fuel(name="heptane", fuelDataDir=invalid_path)

    def test__default_decomp_name_same_as_name(self) -> None:
        """Test the default decomposition name."""
        fuel = Fuel(name="heptane")
        assert fuel.decompName == "heptane"

    def test__default_decomp_name_different_from_name(self) -> None:
        """Test the default decomposition name when it differs from the fuel name."""
        fuel = Fuel(name="posf10325")
        assert fuel.decompName == "posf-cat-a"

    def test__custom_decomp_name_decomp_more_bins_than_fuel(self) -> None:
        """Test specifying a custom decomposition name where the decomp file has more entries than the fuel mixture."""
        custom_decomp_name = "posf-cat-a"
        fuel = Fuel(name="heptane", decompName=custom_decomp_name)
        assert fuel.decompName == custom_decomp_name
        assert len(fuel.gani_decomp) == 1

    def test__custom_decomp_name_decomp_missing_compounds(self) -> None:
        """Test specifying a custom decomposition name where the decomp file is missing compounds present in the fuel mixture."""
        custom_decomp_name = "heptane"
        fuel = Fuel(name="posf10325", decompName=custom_decomp_name)
        assert fuel.decompName == custom_decomp_name
        with pytest.raises(ValueError):
            _ = fuel.gani_decomp

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("heptane", 1), ("posf10325", 67)],
    )
    def test__gcxgc_data(self, fuel_name: str, num_compounds: int) -> None:
        """Test loading the gcxgc data for the fuel."""
        fuel = Fuel(name=fuel_name)
        gcxgc_data = fuel.gcxgc_data
        assert len(gcxgc_data) == num_compounds

    def test__gcxgc_bin_parsing(self) -> None:
        """Test parsing the gcxgc bin data for the fuel."""
        fuel = Fuel(name="heptane")
        (bin, nC, family) = fuel.gcxgc_bins[0]
        assert bin == "n-C07"
        assert nC == 7
        assert family == "n"

    @pytest.mark.parametrize(
        "fuel_name, num_compounds",
        [("heptane", 1), ("posf10325", 67)],
    )
    def test__initial_weight_fractions(
        self, fuel_name: str, num_compounds: int
    ) -> None:
        """Test the initial weight fractions (Y_0) for the fuel."""
        fuel = Fuel(name=fuel_name)
        Y_0 = fuel.Y_0
        assert Y_0 is not None
        assert len(Y_0) == num_compounds
        assert np.allclose(Y_0.sum(), 1.0)

    def test__gani_decomp(self) -> None:
        """Test loading the Gani decomposition data for the fuel."""
        fuel = Fuel(name="heptane")
        gani_decomp = fuel.gani_decomp
        assert gani_decomp is not None
        assert len(gani_decomp) == 1
        assert gani_decomp.shape[1] == 121
        assert fuel.Tc.units == "K"
        assert np.allclose(fuel.Tc.magnitude[-1], 549.856)

    def test__reordered_gani_decomp(self, custom_path: Path) -> None:
        """Test the reordered Gani decomposition data for the fuel."""
        fuel = Fuel(
            name="heptane-decane",
            decompName="heptane-decane",
            fuelDataDir=custom_path,
        )
        fuel_reordered = Fuel(
            name="heptane-decane",
            decompName="heptane-decane_reordered",
            fuelDataDir=custom_path,
        )
        assert fuel.gani_decomp.equals(fuel_reordered.gani_decomp)

    def test__abbreviated_gani_decomp(self, custom_path: Path) -> None:
        """Test loading the abbreviated Gani decomposition data for the fuel."""
        fuel = Fuel(name="octane", decompName="octane_abbr", fuelDataDir=custom_path)
        gani_decomp = fuel.gani_decomp
        assert gani_decomp is not None
        assert len(gani_decomp) == 1
        assert gani_decomp.shape[1] == 2
        assert np.allclose(fuel.Tc.magnitude, 577.94574)
