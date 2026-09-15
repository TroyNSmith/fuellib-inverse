"""Loss functions for optimization of fuel mixture compositions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AromaticConstraint:
    """Constraint on the aromatic content of a fuel mixture."""

    name: str
    display: bool = False
