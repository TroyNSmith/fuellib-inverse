"""Scratch file for testing solve_inverse_problem."""

from pathlib import Path

from fuellib_inverse.inverse import solve

solve(Path(__file__).parent / "jet-a.csv", Path(__file__).parent / "input.json")
