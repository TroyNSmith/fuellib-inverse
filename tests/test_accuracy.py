"""Test accuracy of fuellib property calculations."""

import unittest
from pathlib import Path

import pandas as pd
import pytest

from fuellib import Fuel, Units
from fuellib.correlation import mixture
from fuellib.utils import resolved_np as np

DATA_DIR = Path(__file__).parent / "data/accuracy"

RED = "\033[31m"  # ANSI escape code for red text
GREEN = "\033[32m"  # ANSI escape code for green text
BLUE = "\033[1;34m"  # ANSI escape code for blue text
RESET = "\033[0m"  # ANSI escape code to reset text color


def retrieve_data(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve the baseline and experimental data for a given fuel name.

    :param name: Name of the fuel.
    :type name: str
    :return: Baseline and experimental data as pandas DataFrames.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    base_df = pd.read_csv(DATA_DIR / f"{name}.base.csv", header=0)
    exp_df = pd.read_csv(DATA_DIR / f"{name}.exp.csv", header=0)
    return base_df, exp_df


class TestAccuracyCase(unittest.TestCase):
    def test__mixture_accuracy(self):

        fuel_names = [
            "heptane",
            "decane",
            "dodecane",
            "posf10264",
            "posf10325",
            "posf10289",
        ]
        properties = {
            "Density": mixture.density,
            "Viscosity": mixture.kinematic_viscosity,
            "VaporPressure": mixture.saturated_vapor_pressure,
            "SurfaceTension": mixture.surface_tension,
            "ThermalConductivity": mixture.thermal_conductivity,
        }

        for name in fuel_names:
            base_df, exp_df = retrieve_data(name)

            # Extract temperature units and values from the baseline DataFrame
            t_units_base = str(base_df["Temperature"][0])
            t_vals_base = base_df["Temperature"][1:].astype(float).to_numpy()
            t_base = Units.Q(t_vals_base, t_units_base).to("K")

            # Extract temperature units and values from the experimental DataFrame
            t_units_exp = str(exp_df["Temperature"][0])
            t_vals_exp = exp_df["Temperature"][1:].astype(float).to_numpy()
            t_exp = Units.Q(t_vals_exp, t_units_exp).to("K")

            self.assertTrue(np.allclose(t_base.magnitude, t_exp.magnitude))

            fuel = Fuel(name)
            passed_checks = 0
            total_checks = 0

            print(f"\n\n{BLUE}Accuracy Regression Check via MAPE:{RESET}")
            for prop, method in properties.items():
                with self.subTest(prop=prop):
                    total_checks += 1

                    # Extract property values and units from the baseline DataFrame
                    p_units_base = str(base_df[prop][0])
                    p_vals_base = base_df[prop][1:].astype(float).to_numpy()

                    # Extract property values and units from the experimental DataFrame
                    p_units_exp = str(exp_df[prop][0])
                    p_vals_exp = exp_df[prop][1:].astype(float).to_numpy()

                    # Determine valid indices from the baseline data (not NaN)
                    keep = ~np.isnan(p_vals_base)
                    ## Make sure all values are in the same units before comparison
                    base = Units.Q(p_vals_base[keep], p_units_base).to(p_units_base)
                    exp = Units.Q(p_vals_exp[keep], p_units_exp).to(p_units_base)
                    new = method(fuel, fuel.Y_0, t_base[keep]).to(p_units_base)

                    mape_base = np.mean(
                        np.abs((exp.magnitude - base.magnitude) / exp.magnitude) * 100
                    )
                    mape_new = np.mean(
                        np.abs((exp.magnitude - new.magnitude) / exp.magnitude) * 100
                    )

                    regression_ok = (mape_new <= mape_base) or np.allclose(
                        mape_new,
                        mape_base,
                        atol=5e-2,  # Accept 0.05% as a negligible difference
                    )
                    if regression_ok:
                        passed_checks += 1
                        print(
                            f"  {GREEN}"
                            f"✓ {prop}"
                            f"{RESET}"
                            f"\n    Baseline   = {mape_base:8.4f}%"
                            f"\n    New        = {mape_new:8.4f}%"
                            f"\n    Difference = {mape_new - mape_base:8.4f}%"
                            "\n"
                        )
                    else:
                        print(
                            f"  {RED}"
                            f"✗ {prop}  "
                            f"{RESET}"
                            f"\n    Baseline   = {mape_base:8.4f}%"
                            f"\n    New        = {mape_new:8.4f}%"
                            f"\n    Difference = {mape_new - mape_base:8.4f}%"
                            "\n"
                        )

                    self.assertTrue(
                        regression_ok,
                        f"{name} / {prop}: MAPE regressed from {mape_base:.4f}% (baseline) to {mape_new:.4f}%.",
                    )

        print(f"\n{passed_checks}/{total_checks} fuel-property checks passed")
