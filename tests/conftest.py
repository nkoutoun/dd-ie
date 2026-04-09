"""Shared test fixtures for dd_ie."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def balanced_panel() -> pd.DataFrame:
    """Generate a balanced panel: 20 units x 8 periods with known DGP.

    The data-generating process follows the simulation setup from
    Giesselmann & Schmidt-Catran (2022), Section "Simulation Setup":
        y_it = x_it + z_it + 1.0 * x_it * z_it + u_i + e_it
    where u_i is a unit-specific effect.
    """
    rng = np.random.default_rng(42)
    n_units, n_time = 20, 8
    rows = []

    for unit in range(1, n_units + 1):
        unit_effect = rng.normal(0, 1)
        x_mean = rng.normal(0, 1)
        z_mean = rng.normal(0, 1)

        for t in range(1, n_time + 1):
            x = x_mean + rng.normal(0, 1)
            z = z_mean + rng.normal(0, 1)
            e = rng.normal(0, 0.5)
            y = x + z + 1.0 * x * z + unit_effect + e
            rows.append(
                {
                    "unit_id": unit,
                    "time_id": t,
                    "y": y,
                    "x": x,
                    "z": z,
                    "control1": rng.normal(0, 1),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def unbalanced_panel() -> pd.DataFrame:
    """Generate an unbalanced panel: varying period counts per unit."""
    rng = np.random.default_rng(123)
    rows = []

    for unit in range(1, 11):
        n_periods = rng.integers(3, 10)
        for t in range(1, n_periods + 1):
            rows.append(
                {
                    "unit_id": unit,
                    "time_id": t,
                    "y": rng.normal(0, 1),
                    "x": rng.normal(0, 1),
                    "z": rng.normal(0, 1),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def panel_no_within_variation() -> pd.DataFrame:
    """Panel where 'x' has no within-unit variation (time-invariant)."""
    rng = np.random.default_rng(99)
    rows = []

    for unit in range(1, 6):
        x_val = rng.normal(0, 1)
        for t in range(1, 5):
            rows.append(
                {
                    "unit_id": unit,
                    "time_id": t,
                    "y": rng.normal(0, 1),
                    "x": x_val,  # constant within unit
                    "z": rng.normal(0, 1),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def simple_panel() -> pd.DataFrame:
    """Minimal panel for deterministic calculation checks."""
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time_id": [1, 2, 3, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "z": [2.0, 4.0, 6.0, 1.0, 3.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
