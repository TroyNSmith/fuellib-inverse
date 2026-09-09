"""Core module for the FuelLib inverse package."""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.typing import ArrayLike

from . import components, mixture
from .types import CriticalProperties, SolverConfig, _MinMax


def _bounds_penalty(value: ArrayLike, bounds: _MinMax) -> Array:
    """
    Squared-hinge penalty weighted by ``bounds["lambda_"]`` for a value constrained
    to ``[bounds["min"], bounds["max"]]``. Zero when the value is within bounds.

    :param value: Value to constrain.
    :type value: ArrayLike
    :param bounds: Mapping with "min", "max", and "lambda_" keys.
    :type bounds: _MinMax
    :return: Weighted squared penalty.
    :rtype: Array
    """
    below = jnp.maximum(bounds["min"] - value, 0.0)
    above = jnp.maximum(value - bounds["max"], 0.0)
    return bounds["lambda_"] * (below**2 + above**2)


def solve(csv_path: str | Path, json_path: str | Path) -> Array:
    """
    Solve the inverse problem.

    :param csv_path: Path to the CSV file containing the fuel data.
    :type csv_path: str | Path
    :param json_path: Path to the JSON file containing solver configuration.
    :type json_path: str | Path
    :return: Optimized weight fractions of the fuel components.
    :rtype: Array
    """
    props = CriticalProperties.from_csv(csv_path)
    config = SolverConfig.from_json(json_path)

    optimizer = optax.adam(config.optimization["learning_rate"])

    # Y = softmax(Z) always satisfies Y >= 0 and sum(Y) == 1.
    Z = jnp.zeros(props.num_compounds)

    # Pre-computed properties
    density_constraints = config.constraints["volatility"]["density"]
    component_densities = {
        T: components.densities(
            jnp.asarray(T), props.Tc, props.Vm_stp, props.omega, props.MW
        )
        for T in density_constraints
    }

    viscosity_constraints = config.constraints["fluidity"]["viscosity"]
    component_viscosities = {
        T: components.kinematic_viscosities(jnp.asarray(T), props.Tb)
        for T in viscosity_constraints
    }

    # NOTE: additional constraints (e.g. other composition targets) should be
    # added here, keyed by name, and referenced from `loss_fn` below.
    constraint_bounds = {
        "aromatics_volume_percent": config.constraints["composition"]["aromatics"][
            "volume_percent"
        ]
    }

    def loss_fn(Z: ArrayLike) -> tuple[Array, dict[str, Array]]:
        """Total weighted constraint loss and per-constraint values."""
        Y = jax.nn.softmax(Z)
        X = components.mole_fractions(Y, props.MW)
        V = components.volume_fractions(X, props.Vm_stp)

        constraints = {}
        loss = 0.0

        # Aromatics content
        V_aromatics = mixture.aromatics_content(V, props.hydrocarbon_types) * 100.0
        constraints["aromatics_volume_percent"] = V_aromatics
        loss += _bounds_penalty(
            V_aromatics, constraint_bounds["aromatics_volume_percent"]
        )

        # Density
        for T, bounds in density_constraints.items():
            mixture_density = mixture.density(X, component_densities[T])
            constraints[f"density_kg_per_m3_at_{T:g}K"] = mixture_density
            loss += _bounds_penalty(mixture_density, bounds)

        # Kinematic viscosity
        for T, bounds in viscosity_constraints.items():
            mixture_viscosity = mixture.kinematic_viscosity(
                T, X, component_viscosities[T]
            )
            constraints[f"kinematic_viscosity_m2_per_s_at_{T:g}K"] = mixture_viscosity
            loss += _bounds_penalty(mixture_viscosity, bounds)

        return loss, constraints

    @jax.jit
    def step(
        Z: Array, opt_state: optax.OptState
    ) -> tuple[Array, optax.OptState, Array, dict[str, Array]]:
        (loss, constraints), grads = jax.value_and_grad(loss_fn, has_aux=True)(Z)
        updates, opt_state = optimizer.update(grads, opt_state, Z)
        Z = jnp.asarray(optax.apply_updates(Z, updates))
        return Z, opt_state, loss, constraints

    opt_state = optimizer.init(Z)
    tolerance = config.optimization["tolerance"]
    for _ in range(config.optimization["max_iterations"]):
        prev_Z = Z
        Z, opt_state, _loss, _constraints = step(Z, opt_state)
        if jnp.max(jnp.abs(Z - prev_Z)) < tolerance:
            break

    Y = jax.nn.softmax(Z)
    X = components.mole_fractions(Y, props.MW)
    V = components.volume_fractions(X, props.Vm_stp)

    aromatics_content = mixture.aromatics_content(V, props.hydrocarbon_types) * 100.0
    print(f"Aromatics content: {aromatics_content:.4f} vol%")

    for T in density_constraints:
        mixture_density = mixture.density(X, component_densities[T])
        print(f"Density at {T:.2f} K: {mixture_density:.4f} kg/m^3")

    for T in viscosity_constraints:
        mixture_viscosity = mixture.kinematic_viscosity(T, X, component_viscosities[T])
        print(f"Kinematic viscosity at {T:.2f} K: {mixture_viscosity:.4e} m^2/s")

    return Y
