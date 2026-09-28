"""Scratch file for testing solve_inverse_problem."""

from pathlib import Path

from fuellib_inverse import OptimizationParameters, solve

cwd = Path(__file__).parent
pars = OptimizationParameters.from_txt(cwd / "data/input.txt")

Y = solve(pars, out=cwd / "data/output.csv", seed=2124432)
