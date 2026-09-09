"""Test."""

from pathlib import Path

import numpy as np
import pandas as pd
from unxt import Quantity

from fuellib_inverse import Fuel, inverse

fuel = Fuel("jet-a")

data = {
    "SMILES": fuel.smiles,
    "Family": fuel.families,
    "Hydrocarbon Type": fuel.hydrocarbon_types,
    "MW": fuel.MW.value,
    "Tc": fuel.Tc.value,
    "Pc": fuel.Pc.value,
    "Vc": fuel.Vc.value,
    "Tb": fuel.Tb.value,
    "Tm": fuel.Tm.value,
    "Hf": fuel.Hf.value,
    "Gf": fuel.Gf.value,
    "Hv_stp": fuel.Hv_stp.value,
    "Vm_stp": fuel.Vm_stp.value,
    "Cp_stp": fuel.Cp_stp.value,
    "Cp_B": fuel.Cp_B.value,
    "Cp_C": fuel.Cp_C.value,
    "omega": fuel.omega.value,
}
units = {
    "SMILES": np.nan,
    "Family": np.nan,
    "Hydrocarbon Type": np.nan,
    "MW": "g/mol",
    "Tc": "K",
    "Pc": "Pa",
    "Vc": "m^3/mol",
    "Tb": "K",
    "Tm": "K",
    "Hf": "J/mol",
    "Gf": "J/mol",
    "Hv_stp": "J/mol",
    "Vm_stp": "m^3/mol",
    "Cp_stp": "J/(mol*K)",
    "Cp_B": "J/(mol*K)",
    "Cp_C": "J/(mol*K)",
    "omega": "dimensionless",
}

df = pd.DataFrame(data)
df = pd.concat([pd.DataFrame([units]), df], ignore_index=True)
df.set_index("SMILES", inplace=True)
df.to_csv(Path(__file__).parent / "jet-a.csv")
