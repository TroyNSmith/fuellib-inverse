"""Helpers for property correlations."""

from typing import Any

from ..utils import Units, types
from ..utils import resolved_np as np


def atleast_1d(array: types.Quantity0D | types.Quantity1D) -> types.Quantity1D:
    """Ensure the input array is at least 1-dimensional.

    :param array: Input array, which can be 0D or 1D.
    :type array: types.Quantity0D | types.Quantity1D
    :return: Array ensured to be at least 1-dimensional.
    :rtype: types.Quantity1D
    """
    arr = np.array(array.magnitude)
    return Units.Quantity(np.atleast_1d(arr), array.units)


def weight_to_mol_fraction(
    Y: types.Array1D, molar_masses: types.Array1D
) -> types.Array1D:
    """Convert weight fractions (Y) to mole fractions (X).

    :param Y: Weight fractions of the components.
    :type Y: types.Array1D
    :param molar_masses: Molar masses of the components.
    :type molar_masses: types.Array1D
    :return: Mole fractions of the components.
    :rtype: types.Array1D
    """
    mole_fractions = np.array(Y) / np.array(molar_masses)
    mole_fractions /= mole_fractions.sum()
    return mole_fractions


def arithmetic_mixing_rule(
    X: types.Array1D, values: types.Array1D | types.Array2D | Any
) -> float | types.Array1D:
    """Apply the (quadratic) arithmetic mixing rule to a set of property values.

    Computes ``sum_i sum_j X[i] * X[j] * (values[i] + values[j]) / 2`` over the
    leading (component) axis. ``values`` may carry additional trailing axes
    (e.g. a batch of temperatures), which are broadcast through unchanged.

    :param X: Mole fractions of the components.
    :type X: types.Array1D
    :param values: Property values of the components, with the component axis
        first (shape ``(n_components, ...)``).
    :type values: types.Array1D | types.Array2D
    :return: Mixed property value.
    :rtype: float | types.Array1D
    """
    values = np.array(values)
    X = np.array(X)
    n_components = X.shape[0]

    values_i = values.reshape((n_components, 1) + values.shape[1:])
    values_j = values.reshape((1, n_components) + values.shape[1:])
    pairwise_mean = (values_i + values_j) / 2

    X_outer = X[:, None] * X[None, :]
    weight = X_outer.reshape(X_outer.shape + (1,) * (values.ndim - 1))

    return np.sum(weight * pairwise_mean, axis=(0, 1))
