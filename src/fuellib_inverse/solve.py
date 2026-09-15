"""Module for solving the inverse problem using optimization parameters."""

import logging
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from fuellib import fuel

from .parse import OptimizationParameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()

formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define ANSI codes
BLUE = "\033[34m"
GREEN = "\033[1;32m"
RED = "\033[1;31m"
STOP = "\033[0m"


def write_csv(filename: str | Path, Y: jnp.ndarray, fuel: fuel) -> None:
    """Write the mass fraction vector to a CSV file."""
    df = pd.DataFrame({"Family": fuel.compounds, "Weight %": Y})
    df.to_csv(filename, index=False)


def solve(
    pars: OptimizationParameters,
    *,
    Y0: jnp.ndarray | None = None,
    out: str | Path | None = None,
    verbosity: Literal[0, 1, 2, 3, 4, 5] = 2,
) -> jnp.ndarray:
    """Solve the inverse problem using the given optimization parameters.

    :param pars: Optimization parameters, including constraints and solver settings.
    :param Y0: Optional initial guess for the (pre-softmax) mass-fraction logits.
    :param verbosity: Logging verbosity level.
        ``0`` disables logging
        ``1`` enables DEBUG
        ``2`` enables INFO
        ``3`` enables WARNING
        ``4`` enables ERROR
        ``5`` enables CRITICAL
    :return: Optimized mass fraction vector.
    """
    handler.setLevel(verbosity * 10)

    Z0 = Y0 if Y0 is not None else jnp.zeros(pars.fuel.num_compounds)
    Z = Z0

    loss_fn = lambda Z: sum([c.loss(jax.nn.softmax(Z)) for c in pars.constraints])

    @jax.jit
    def step(
        Z: jnp.ndarray, opt_state: optax.OptState
    ) -> tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(Z)
        updates, opt_state = pars.optimizer.update(grads, opt_state, Z)
        Z = jnp.asarray(optax.apply_updates(Z, updates))
        return Z, opt_state, loss

    opt_state = pars.optimizer.init(Z)
    Y = jax.nn.softmax(Z)
    for i in range(pars.max_iter):
        prev_Y = Y
        Z, opt_state, loss = step(Z, opt_state)
        Y = jax.nn.softmax(Z)
        max_deviation = jnp.max(jnp.abs(Y - prev_Y))
        if max_deviation < pars.tolerance:
            logger.info(f"\n{GREEN}Optimization converged after %d steps.{STOP}", i + 1)
            break
        logger.debug(
            "Step %d: loss = %.6g, max deviation = %.6g",
            i + 1,
            loss,
            max_deviation,
        )
    else:
        logger.warning(
            f"{RED}Optimization did not converge within max_iterations=%d "
            f"(max deviation (Y) = %.6g).{STOP}",
            pars.max_iter,
            max_deviation,
        )

    for c in pars.constraints:
        temp = getattr(c, "temperature", None)
        units = getattr(c.target_range, "units", "dimensionless")
        if temp is not None:
            logger.info(
                f"\n{BLUE}Final value for %s constraint{STOP}:\nTemperature: %.2f %s\nValue: %.6g %s",
                c.property,
                temp.magnitude,
                temp.units,
                c.value(Y),
                units,
            )
        else:
            logger.info(
                f"\n{BLUE}Final value for constraint %s{STOP}:\nTemperature: N/A\nValue: %.6f %s",
                c.property,
                c.value(Y),
                units,
            )

    logger.info(
        f"\n{BLUE}Final optimized mass fraction vector (sum: %.2f):{STOP}\n%s",
        jnp.sum(Y),
        Y,
    )

    if out:
        write_csv(out, Y, pars.fuel)

    return Y
