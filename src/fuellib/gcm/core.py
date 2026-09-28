"""Core Group Contribution Method registry."""

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel

from ..utils import types

if TYPE_CHECKING:
    from ..fuel import Fuel


__all__ = ["GCMRegistry"]


@runtime_checkable
class PropertyProtocol(Protocol):
    """Protocol for properties that can be predicted by GCMs."""

    __name__: str

    def __call__(self, fuel: "Fuel") -> types.Quantity1D:
        """Predict the property for the given fuel's components."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class GCM(BaseModel):
    """Base class for Core Group Contribution Methods."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    property_fns: dict[str, PropertyProtocol]

    def register_property(self, function: PropertyProtocol) -> PropertyProtocol:
        """Register a new property implementation for this GCM."""
        self.property_fns[function.__name__.lower()] = function
        return function

    def list_property_fns(self) -> list[str]:
        """List the names of all registered property functions."""
        return sorted(self.property_fns.keys())

    def get_property(self, property_name: str) -> PropertyProtocol:
        """Retrieve the property function for the specified property name."""
        property_name = property_name.lower()
        prop = self.property_fns.get(property_name)
        if prop is not None:
            return prop
        msg = f"Property '{property_name}' is not implemented by {self.name} GCM. Available property functions: {self.list_property_fns()}"
        raise ValueError(msg)

    def predict(self, property_name: str, fuel: "Fuel") -> types.Quantity1D:
        """Predict the specified property for the given fuel's components."""
        property_name = property_name.lower()
        if property_name not in self.property_fns:
            msg = f"Property '{property_name}' is not implemented by this GCM. Available property functions: {self.list_property_fns()}"
            raise ValueError(msg)
        return self.property_fns[property_name](fuel)

    def predict_all(self, fuel: "Fuel") -> dict[str, dict[str, types.Quantity1D]]:
        """Predict all properties for the given fuel's components using registered property functions."""
        return {
            self.name: {
                prop: self.predict(prop, fuel) for prop in self.list_property_fns()
            }
        }


class GCMRegistry:
    """Registry for Core Group Contribution Methods."""

    methods: ClassVar[list[GCM]] = []

    @classmethod
    def register(
        cls, gcm_name: str, property_fns: list[PropertyProtocol] | None = None
    ) -> GCM:
        """Register a new GCM with the given name and properties."""
        gcm = GCM.model_validate(
            {
                "name": gcm_name.lower(),
                "property_fns": {prop.__name__.lower(): prop for prop in property_fns}
                if property_fns is not None
                else {},
            }
        )
        cls.methods.append(gcm)
        return gcm

    @classmethod
    def list_methods(cls) -> list[str]:
        """List the names of all registered GCMs."""
        return [gcm.name for gcm in cls.methods]

    @classmethod
    def get_gcm(cls, gcm_name: str) -> GCM:
        """Retrieve a registered GCM by its name."""
        gcm_name = gcm_name.lower()
        for gcm in cls.methods:
            if gcm.name == gcm_name:
                return gcm
        msg = f"GCM with name '{gcm_name}' is not registered. Available GCMs: {cls.list_methods()}"
        raise ValueError(msg)
