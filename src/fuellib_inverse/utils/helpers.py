"""Helper functions for FuelLib."""

import quaxed.numpy as qnp
from unxt import AbstractQuantity


def atleast_col(x: AbstractQuantity) -> AbstractQuantity:
    """
    Add a trailing axis so an array of temperatures broadcasts against a per-compound axis; scalars pass through.

    Idempotent: only 1D arrays are expanded, so calling this again on an already-expanded
    array (e.g. because a caller already column-expanded T before passing it to another
    method that also calls this) is a no-op instead of adding another axis.

    :param x: Input array.
    :type x: AbstractQuantity
    :return: Array with at least one trailing axis.
    :rtype: AbstractQuantity
    """
    return x[:, None] if qnp.ndim(x) == 1 else x
