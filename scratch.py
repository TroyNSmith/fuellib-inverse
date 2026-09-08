"""Scratch file for testing and debugging."""

from unxt import Quantity

from fuellib_inverse import Fuel

fuel = Fuel("jet-a")
t = Quantity(300, "K")

print(fuel.mixture_density(t))
