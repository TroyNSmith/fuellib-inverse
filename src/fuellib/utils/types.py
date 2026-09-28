"""Type definitions for the Fuellib project."""

from typing import Any, ClassVar, TypeVar, cast

try:
    import astropy.units as u
    import jax.numpy as resolved_np
    import unxt
    from jaxtyping import Array, Float

    _ = resolved_np.array([1])  # Check that jax initializes properly
    Array1D = Float[Array, "I"]
    Array2D = Float[Array, "I J"]

    _ = unxt.Q(1, "K")  # Check that unxt initializes properly

    # Enable temperature conversions for astropy units
    u.add_enabled_equivalencies(u.temperature())
    # Add additional units to astropy to ensure compatibility with pint
    atm = u.def_unit("atmosphere", 101325 * u.Pa)
    u.add_enabled_units([atm])
    # Enable aliases for astropy units to ensure compatibility with pint
    u.add_enabled_aliases({"celsius": u.deg_C, "atm": atm})

    class UnxtQuantity(unxt.Quantity):
        """Resolved quantity class using unxt."""

        def __init__(
            self,
            value: Array,
            unit: u.UnitBase | u.FunctionUnitBase,
        ):
            super().__init__(value, unit)

        @property
        def magnitude(self) -> Array | unxt.quantity.StaticValue:
            return self.value

        @property
        def units(self) -> u.UnitBase | u.FunctionUnitBase:
            return self.unit

        def to(self, unit: Any, /) -> "UnxtQuantity":
            return cast("UnxtQuantity", super().to(unit))

    # `AbstractQuantity` (not `Quantity`) since methods like `.to()`/`.uconvert()`
    # are annotated to return the base class, not `Self`.
    UnxtQuantityT = TypeVar("UnxtQuantityT", bound=UnxtQuantity)
    Quantity0D = Float[UnxtQuantity, ""]
    Quantity1D = Float[UnxtQuantity, "I"]
    Quantity2D = Float[UnxtQuantity, "I J"]

    has_jax = True


except (ImportError, Exception):  # noqa: BLE001
    import numpy as resolved_np
    import pint

    _ = resolved_np.array([1])  # Check that numpy initializes properly
    Array1D = resolved_np.ndarray[tuple[int,], resolved_np.dtype[resolved_np.float32]]
    Array2D = resolved_np.ndarray[
        tuple[int, int], resolved_np.dtype[resolved_np.float32]
    ]

    ureg = pint.UnitRegistry()

    PintQuantityT = TypeVar("PintQuantityT", bound=pint.Quantity)
    Quantity0D = pint.Quantity[float]
    Quantity1D = pint.Quantity[Array1D]
    Quantity2D = pint.Quantity[Array2D]

    has_jax = False

QuantityT = UnxtQuantityT if has_jax else PintQuantityT


class Units:
    """Wrapper class for the resolved units system."""

    Quantity: ClassVar[type[QuantityT]] = UnxtQuantity if has_jax else pint.Quantity
    Q: ClassVar[type[QuantityT]] = UnxtQuantity if has_jax else pint.Quantity


__all__ = [
    "Array1D",
    "Array2D",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
    "has_jax",
    "resolved_np",
]
