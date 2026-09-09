"""Scratch file for testing and debugging."""

import jax
import jax.numpy as jnp
import optax
from jax import Array
from unxt import Quantity

from fuellib_inverse import Fuel

fuel = Fuel("jet-a")
t = Quantity(300, "K")

print(fuel.mixture_density(t))


def solve_Y_0_for_targets(
    fuel: Fuel,
    *,
    target_density: Quantity,
    target_nu_m40: Quantity,
    target_nu_m25: Quantity,
    T_density: Quantity,
    Y_init: Array | None = None,
    w_density: float = 1.0,
    w_nu_m40: float = 1.0,
    w_nu_m25: float = 1.0,
    reg_weight: float = 1e-2,
    lr: float = 1e-2,
    n_steps: int = 2000,
) -> Array:
    """
    Find mass fractions `Y_0` that jointly match `fuel.mixture_density(T_density)`,
    `fuel.mixture_kinematic_viscosity(-40 degC)`, and `fuel.mixture_kinematic_viscosity(-25 degC)`
    to the given targets.

    The per-compound densities `rho_i = MW_i / molar_liquid_vol_i(T_density)` and the
    per-compound kinematic viscosities `nu_i` at -40 degC and -25 degC do not depend on
    `Y_0` (only on temperature and fixed critical properties), so they are computed once
    and the rest of the problem is solved as pure JAX. `Y_0` is parametrized as
    `softmax(z)` of unconstrained logits `z` so every candidate is automatically a valid
    mass-fraction vector (non-negative, sums to 1). The optimizer (optax Adam) minimizes

        w_density * (density - target_density)^2
        + w_nu_m40 * (nu_m40 - target_nu_m40)^2
        + w_nu_m25 * (nu_m25 - target_nu_m25)^2
        + reg_weight * ||Y - Y_init||^2

    where `nu_m40`/`nu_m25` are computed with the Kendall-Monroe mixing rule (same as
    `fuel.mixture_kinematic_viscosity(..., correlation="Kendall-Monroe")`), and the
    regularization term keeps the solution close to `Y_init`, since matching three
    scalar properties does not uniquely determine `Y_0` for mixtures with more than
    three compounds.

    :param fuel: Fuel object providing component molecular weights, molar volumes, and
        boiling points.
    :type fuel: Fuel
    :param target_density: Desired mixture density at temperature `T_density`.
    :type target_density: Quantity
    :param target_nu_m40: Desired mixture kinematic viscosity at -40 degC.
    :type target_nu_m40: Quantity
    :param target_nu_m25: Desired mixture kinematic viscosity at -25 degC.
    :type target_nu_m25: Quantity
    :param T_density: Temperature at which the target density should be achieved.
    :type T_density: Quantity
    :param Y_init: Initial guess for the mass fractions. Defaults to `fuel.Y_0`.
    :type Y_init: Array | None
    :param w_density: Weight of the density loss term.
    :type w_density: float
    :param w_nu_m40: Weight of the -40 degC viscosity loss term.
    :type w_nu_m40: float
    :param w_nu_m25: Weight of the -25 degC viscosity loss term.
    :type w_nu_m25: float
    :param reg_weight: Weight of the L2 regularization pulling `Y` toward `Y_init`.
    :type reg_weight: float
    :param lr: Learning rate for the optax Adam optimizer.
    :type lr: float
    :param n_steps: Number of gradient-descent steps to take.
    :type n_steps: int
    :return: Optimized mass fractions summing to 1.
    :rtype: Array
    """
    # rho_i and nu_i are independent of Y_0, so precompute them once, outside the
    # optimization, and strip units so the inner loop is pure JAX.
    rho_i = jnp.asarray((fuel.MW / fuel.molar_liquid_vol(T_density)).to("kg/m^3").value)
    MW_i = jnp.asarray(fuel.MW.to("g/mol").value)
    nu_i_m40 = jnp.asarray(
        fuel.viscosity_kinematic(Quantity(-40.0, "Celsius")).to("mm^2/s").value
    )
    nu_i_m25 = jnp.asarray(
        fuel.viscosity_kinematic(Quantity(-25.0, "Celsius")).to("mm^2/s").value
    )

    target_rho = jnp.asarray(target_density.to("kg/m^3").value)
    target_nu40 = jnp.asarray(target_nu_m40.to("mm^2/s").value)
    target_nu25 = jnp.asarray(target_nu_m25.to("mm^2/s").value)

    Y_init = jnp.asarray(fuel.Y_0 if Y_init is None else Y_init)
    z0 = jnp.log(jnp.clip(Y_init, 1e-12, None))  # logits whose softmax is Y_init

    def kendall_monroe_nu(Y: Array, nu_i: Array) -> Array:
        X = (Y / MW_i) / jnp.sum(Y / MW_i)
        return jnp.exp(jnp.sum(X * jnp.log(nu_i)))

    def loss_fn(z: Array) -> Array:
        Y = jax.nn.softmax(z)
        density = jnp.sum(Y * rho_i)
        nu_m40 = kendall_monroe_nu(Y, nu_i_m40)
        nu_m25 = kendall_monroe_nu(Y, nu_i_m25)
        data_term = (
            w_density * (density - target_rho) ** 2
            + w_nu_m40 * (nu_m40 - target_nu40) ** 2
            + w_nu_m25 * (nu_m25 - target_nu25) ** 2
        )
        reg_term = reg_weight * jnp.sum((Y - Y_init) ** 2)
        return data_term + reg_term

    optimizer = optax.adam(lr)

    @jax.jit
    def run(z0: Array) -> Array:
        opt_state = optimizer.init(z0)

        def step(
            _, carry: tuple[Array, optax.OptState]
        ) -> tuple[Array, optax.OptState]:
            z, opt_state = carry
            grads = jax.grad(loss_fn)(z)
            updates, opt_state = optimizer.update(grads, opt_state, z)
            z = optax.apply_updates(z, updates)
            return z, opt_state

        z, _ = jax.lax.fori_loop(0, n_steps, step, (z0, opt_state))
        return jax.nn.softmax(z)

    return run(z0)


target_density = Quantity(775.0, "kg/m^3")
target_nu_m40 = Quantity(7.85, "mm^2/s")
target_nu_m25 = Quantity(4.680, "mm^2/s")
Y_solution = solve_Y_0_for_targets(
    fuel,
    target_density=target_density,
    target_nu_m40=target_nu_m40,
    target_nu_m25=target_nu_m25,
    T_density=t,
)

fuel.Y_0 = Y_solution
print(fuel.mixture_density(T=t))
print(fuel.mixture_kinematic_viscosity(Quantity(-40.0, "Celsius")))
print(fuel.mixture_kinematic_viscosity(Quantity(-25.0, "Celsius")))
print(fuel.Y_0)
