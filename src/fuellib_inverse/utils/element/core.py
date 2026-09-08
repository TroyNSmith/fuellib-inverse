"""Core element data interface."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Element:
    """Class representing a chemical element."""

    symbol: str  # Element symbol
    Z: int  # Atomic number
    A: int  # Mass number
    mass: float  # Atomic mass


ELEMENT_BY_NUMBER: dict[int, Element] = {}
ELEMENT_BY_SYMBOL: dict[str, Element] = {}


def _load_elements() -> None:
    """Load element data into the ELEMENT_BY_NUMBER and ELEMENT_BY_SYMBOL dictionaries."""
    # Example data; in practice, this would be loaded from a data source
    data_path = Path(__file__).with_name("elements_data.json")

    with data_path.open("r") as f:
        elements_data: list[dict[str, Any]] = json.load(f)

    for elem in elements_data:
        element = Element(**elem)
        ELEMENT_BY_NUMBER[element.Z] = element
        ELEMENT_BY_SYMBOL[element.symbol] = element


_load_elements()


def from_key(key: int | str) -> Element:
    """
    Retrieve an Element object by its atomic number or symbol.

    :param key: Atomic number (int) or element symbol (str).
    :return: Corresponding Element object.
    :raises KeyError: If the element is not found.
    """
    if isinstance(key, int):
        if key not in ELEMENT_BY_NUMBER:
            raise KeyError(f"Element with atomic number {key} not found.")
        return ELEMENT_BY_NUMBER[key]

    elif isinstance(key, str):
        if key not in ELEMENT_BY_SYMBOL:
            raise KeyError(f"Element with symbol '{key}' not found.")
        return ELEMENT_BY_SYMBOL[key]

    raise TypeError(
        "Key must be an integer (atomic number) or string (element symbol)."
    )


def symbol(key: int | str) -> str:
    """
    Retrieve the symbol of an element by its atomic number or symbol.

    :param key: Atomic number (int) or element symbol (str).
    :return: Symbol of the corresponding element.
    :raises KeyError: If the element is not found.
    """
    return from_key(key).symbol


def number(key: int | str) -> int:
    """
    Retrieve the atomic number of an element by its atomic number or symbol.

    :param key: Atomic number (int) or element symbol (str).
    :return: Atomic number of the corresponding element.
    :raises KeyError: If the element is not found.
    """
    return from_key(key).Z


def mass_number(key: int | str) -> int:
    """
    Retrieve the mass number of an element by its atomic number or symbol.

    :param key: Atomic number (int) or element symbol (str).
    :return: Mass number of the corresponding element.
    :raises KeyError: If the element is not found.
    """
    return from_key(key).A


def mass(key: int | str) -> float:
    """
    Retrieve the atomic mass of an element by its atomic number or symbol.

    :param key: Atomic number (int) or element symbol (str).
    :return: Atomic mass of the corresponding element.
    :raises KeyError: If the element is not found.
    """
    return from_key(key).mass
