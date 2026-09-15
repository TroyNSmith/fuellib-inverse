"""Core module for the FuelLib inverse package."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import quax
import quaxed.numpy as qnp
from jax import Array
from jax.typing import ArrayLike
from unxt import AbstractQuantity, Quantity

from ..utils.units import strip
from . import components, mixture
from .types import CriticalProperties, SolverConfig, _MinMax

logger = logging.getLogger(__name__)


def _bounds_penalty(value: AbstractQuantity, bounds: _MinMax) -> Array:
    """
    Squared-hinge penalty weighted by ``bounds["lambda_"]`` for a value constrained
    to ``[bounds["min"], bounds["max"]]``. Zero when the value is within bounds.

    :param value: Value to constrain.
    :type value: AbstractQuantity
    :param bounds: Mapping with "min", "max", and "lambda_" keys.
    :type bounds: _MinMax
    :return: Weighted squared penalty as a bare (unitless) scalar, so constraints of
        differing physical units can be summed into a single loss.
    :rtype: Array
    """
    value = value.to(bounds["min"].unit)
    zero = Quantity(0.0, bounds["min"].unit)
    below = qnp.maximum(bounds["min"] - value, zero)
    above = qnp.maximum(value - bounds["max"], zero)
    return bounds["lambda_"] * strip(below**2 + above**2)


@dataclass(frozen=True)
class Constraint:
    """
    A single named, physical constraint evaluated during optimization.

    Bundles the computed mixture property (`value`), the bounds it's penalized
    against (`bounds`), and the unit it should be reported in (`display_unit`).
    `_evaluate_constraints` turns a flat list of these into both the total loss
    and a human-readable dict of current values, so adding a new constraint to
    `solve` is just appending one more `Constraint` to the list built in
    `loss_fn` - no other code needs to change.
    """

    name: str
    value: AbstractQuantity
    bounds: _MinMax
    display_unit: str


def _evaluate_constraints(
    constraints: list[Constraint],
) -> tuple[Array, dict[str, Array]]:
    """
    Combine a list of `Constraint`s into a total weighted loss and a dict of their
    current values (each converted to its `display_unit`), keyed by name.

    :param constraints: Constraints to evaluate.
    :type constraints: list[Constraint]
    :return: Total loss and per-constraint current values.
    :rtype: tuple[Array, dict[str, Array]]
    """
    loss = jnp.zeros(())
    values = {}
    for c in constraints:
        values[c.name] = jnp.asarray(c.value.to(c.display_unit).value)
        loss = loss + _bounds_penalty(c.value, c.bounds)
    return loss, values


def _precompute_by_temperature(
    temperatures: dict[float, _MinMax],
    compute: Callable[[AbstractQuantity], AbstractQuantity],
) -> dict[float, AbstractQuantity]:
    """
    Evaluate `compute` once per temperature key, converting each key to a Kelvin
    `Quantity` first.

    :param temperatures: Mapping keyed by temperature (in Kelvin) to precompute a
        component property at, e.g. `config.constraints["volatility"]["density"]`.
    :type temperatures: dict[float, _MinMax]
    :param compute: Function computing a per-component property at a given temperature.
    :type compute: Callable[[AbstractQuantity], AbstractQuantity]
    :return: The precomputed property, keyed by temperature.
    :rtype: dict[float, AbstractQuantity]
    """
    return {T: compute(Quantity(T, "K")) for T in temperatures}


def _load_gcxgc_matches(
    node: dict[str, Any] | None, props: CriticalProperties
) -> list[tuple[str, int, _MinMax]]:
    """
    Match a GCxGC reference CSV's compounds (see `_GCxGCLeaf`) against
    `props.families`, building a `_MinMax` (`Weight % * (1 - rtol)` to
    `Weight % * (1 + rtol)`) for each matched compound. Reference rows whose
    "Compound" has no matching family are logged and skipped.

    :param node: Resolved "composition.gcxgc" constraint node, or None if no gcxgc
        constraint is configured.
    :type node: dict[str, Any] | None
    :param props: Critical properties of the fuel components (`props.families` is
        matched against the reference CSV's "Compound" column).
    :type props: CriticalProperties
    :return: List of `(compound, index into props.families, bounds)` for each
        matched reference compound.
    :rtype: list[tuple[str, int, _MinMax]]
    """
    if node is None:
        return []

    csv_path = node["csv_path"]
    df = pd.read_csv(csv_path)
    required_cols = ["Compound", "Weight %"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Missing required columns in GCxGC reference CSV '{csv_path}'. "
            f"Required columns: {required_cols}"
        )

    family_to_index = {family: i for i, family in enumerate(props.families)}
    rtol = node["rtol"]
    atol = node["atol"]
    matches: list[tuple[str, int, _MinMax]] = []
    for compound, weight_percent in zip(df["Compound"], df["Weight %"], strict=True):
        if compound not in family_to_index:
            logger.warning(
                "GCxGC reference compound '%s' not found in props.families; skipping.",
                compound,
            )
            continue
        w = float(weight_percent)
        bounds: _MinMax = {
            "lambda_": node["lambda_"],
            "min": Quantity(jnp.asarray(w * (1 - rtol) - atol), "dimensionless"),
            "max": Quantity(jnp.asarray(w * (1 + rtol) + atol), "dimensionless"),
        }
        matches.append((compound, family_to_index[compound], bounds))
    return matches


def _optimize(
    loss_fn: Callable[[ArrayLike], tuple[Array, dict[str, Array]]],
    Z0: Array,
    optimizer: optax.GradientTransformation,
    max_iterations: int,
    tolerance: float,
) -> Array:

    def _step(
        Z: Array, opt_state: optax.OptState
    ) -> tuple[Array, optax.OptState, Array, dict[str, Array]]:
        (loss, constraints), grads = jax.value_and_grad(loss_fn, has_aux=True)(Z)
        updates, opt_state = optimizer.update(grads, opt_state, Z)
        Z = jnp.asarray(optax.apply_updates(Z, updates))
        return Z, opt_state, loss, constraints

    # JIT the quaxified step: quax.quaxify lets the jitted function trace through
    # the unxt Quantity arithmetic in loss_fn (mixture densities/viscosities),
    # while jax.jit compiles away quax's dynamic-dispatch overhead.
    step = jax.jit(quax.quaxify(_step))

    Z = Z0
    opt_state = optimizer.init(Z)
    for i in range(max_iterations):
        prev_Z = Z
        Z, opt_state, _loss, _constraints = step(Z, opt_state)
        rmsd = jnp.sqrt(jnp.mean((Z - prev_Z) ** 2))
        logger.warning(
            "step %d: max deviation (Z) = %.6g", i, jnp.max(jnp.abs(Z - prev_Z))
        )
        if jnp.max(jnp.abs(Z - prev_Z)) < tolerance:
            logger.warning("Optimization converged after %d steps.", i + 1)
            break
    else:
        logger.warning(
            "Optimization did not converge within max_iterations=%d (max deviation (Z) = %.6g).",
            max_iterations,
            jnp.max(jnp.abs(Z - prev_Z)),
        )
    return Z


def solve(
    csv_path: str | Path, input_path: str | Path, output_path: str | Path | None = None
) -> Array:
    """
    Solve the inverse problem.

    :param csv_path: Path to the CSV file containing the fuel data.
    :type csv_path: str | Path
    :param input_path: Path to the input.txt-format file containing solver configuration.
    :type input_path: str | Path
    :param output_path: Path to save the optimized weight fractions of the fuel components.
    :type output_path: str | Path
    :return: Optimized weight fractions of the fuel components.
    :rtype: Array
    """
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(input_path).parent / "optimized_fuel.csv"
    )

    props = CriticalProperties.from_csv(csv_path)
    config = SolverConfig.from_input(input_path)

    # Pre-computed, per-constraint-temperature component properties.
    density_constraints = config.constraints["volatility"]["density"]
    component_densities = _precompute_by_temperature(
        density_constraints, lambda T: components.densities(T, props)
    )

    viscosity_constraints = config.constraints["fluidity"]["viscosity"]
    component_viscosities = _precompute_by_temperature(
        viscosity_constraints, lambda T: components.kinematic_viscosities(T, props)
    )

    aromatics_bounds = config.constraints["composition"]["aromatics"]["volume_percent"]
    gcxgc_matches = _load_gcxgc_matches(
        config.constraints.get("composition", {}).get("gcxgc"), props
    )

    def loss_fn(Z: ArrayLike) -> tuple[Array, dict[str, Array]]:
        """Total weighted constraint loss and per-constraint values."""
        Y = jax.nn.softmax(Z)
        X = components.mole_fractions(Y, props)
        V = components.volume_fractions(X, props)

        # NOTE: to add a new constraint, append a `Constraint(name, value,
        # bounds, display_unit)` below - it's automatically folded into the
        # loss and into the per-constraint values returned here (and printed
        # at the end of `solve`). No other code needs to change.
        constraints = [
            Constraint(
                name="aromatics_volume_percent",
                value=Quantity(
                    mixture.aromatics_content(V, props) * 100.0, "dimensionless"
                ),
                bounds=aromatics_bounds,
                display_unit="dimensionless",
            ),
            *(
                Constraint(
                    name=f"density_kg_per_m3_at_{T:g}K",
                    value=mixture.density(X, component_densities[T]),
                    bounds=bounds,
                    display_unit="kg/m^3",
                )
                for T, bounds in density_constraints.items()
            ),
            *(
                Constraint(
                    name=f"kinematic_viscosity_mm2_per_s_at_{T:g}K",
                    value=mixture.kinematic_viscosity(X, component_viscosities[T]),
                    bounds=bounds,
                    display_unit="mm^2/s",
                )
                for T, bounds in viscosity_constraints.items()
            ),
            *(
                Constraint(
                    name=f"gcxgc_weight_percent_{compound}",
                    value=Quantity(Y[idx] * 100.0, "dimensionless"),
                    bounds=bounds,
                    display_unit="dimensionless",
                )
                for compound, idx, bounds in gcxgc_matches
            ),
        ]
        return _evaluate_constraints(constraints)

    Z = _optimize(
        loss_fn,
        Z0=jnp.zeros(props.num_compounds),
        optimizer=optax.adam(config.optimization["learning_rate"]),
        max_iterations=config.optimization["max_iterations"],
        tolerance=config.optimization["tolerance"],
    )

    Y = jax.nn.softmax(Z)

    out = {
        "Family": props.families,
        "SMILES": props.smiles,
        "Weight %": Y * 100.0,
    }
    out_df = pd.DataFrame(out)
    out_df.to_csv(output_path, index=False)

    _, final_constraints = loss_fn(Z)
    for name, value in final_constraints.items():
        print(f"{name}: {value:.4g}")

    return Y
