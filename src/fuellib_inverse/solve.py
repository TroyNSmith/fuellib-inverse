"""Module for solving the inverse problem using optimization parameters."""

import jax
import jax.numpy as jnp
import optax

from .parse import OptimizationParameters


def solve(
    pars: OptimizationParameters, *, Y0: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Solve the inverse problem using the given optimization parameters."""
    Y0 = Y0 or jnp.zeros(pars.num_compounds)
    Y = jax.nn.softmax(Y0)

    loss_fn = lambda Y: sum([c.loss(Y) for c in pars.constraints])

    @jax.jit
    def step(
        Y: jnp.ndarray, opt_state: optax.OptState
    ) -> tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(Y)
        updates, opt_state = pars.optimizer.update(grads, opt_state, Y)
        Y = jnp.asarray(optax.apply_updates(Y, updates))
        return Y, opt_state, loss

    opt_state = pars.optimizer.init(Y)
    for i in range(pars.max_iter):
        prev_Y = Y
        Y, opt_state, loss = step(Y, opt_state)
        if jnp.max(jnp.abs(Y - prev_Y)) < pars.tolerance:
            print(f"Optimization converged after {i + 1} steps.")
            break
        print(
            f"Step {i + 1}: loss = {loss:.6g}, max deviation = {jnp.max(jnp.abs(Y - prev_Y)):.6g}"
        )
    else:
        print(
            f"Optimization did not converge within max_iterations={pars.max_iter} (max deviation (Y) = {jnp.max(jnp.abs(Y - prev_Y)):.6g})."
        )

    return Y
