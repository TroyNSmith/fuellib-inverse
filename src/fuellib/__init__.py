"""
FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`Fuel` for the main class and complete API documentation.
"""

from .fuel import Fuel
from .utils import Units

__all__ = ["Fuel", "Units"]
