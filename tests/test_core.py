"""Tests for core functionality of dd_ie package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dd_ie import DoubleDemeanAnalysis, create_double_demeaned_interaction
from dd_ie._types import ComparisonResult


class TestDoubleDemeanAnalysis:
    """Tests for the DoubleDemeanAnalysis class."""

    def test_init_basic(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
            w_vars=["control1"],
        )
        assert analysis.unit_var == "unit_id"
        assert analysis.y_var == "y"
        assert analysis.x_var == "x"
        assert analysis.z_var == "z"
        assert analysis.w_vars == ["control1"]
        assert len(analysis.data) == 160

    def test_missing_variables_raises_valueerror(self) -> None:
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time_id": [1, 2, 1, 2],
                "y": [1, 2, 3, 4],
            }
        )
        with pytest.raises(ValueError, match="Variables not found in data"):
            DoubleDemeanAnalysis(
                data=df,
                unit_var="unit_id",
                time_var="time_id",
                y_var="y",
                x_var="missing_x",
                z_var="missing_z",
            )

    def test_repr(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        r = repr(analysis)
        assert "DoubleDemeanAnalysis" in r
        assert "y='y'" in r
        assert "n_obs=160" in r

    def test_init_without_controls(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        assert analysis.w_vars == []


class TestCreateDoubleDemeanedInteraction:
    """Tests for create_double_demeaned_interaction."""

    def test_basic_functionality(self, simple_panel: pd.DataFrame) -> None:
        data = simple_panel.set_index(["unit_id", "time_id"])
        result = create_double_demeaned_interaction(data, "x", "z", "unit_id", verbose=False)

        expected_cols = ["x", "z", "mean_x", "mean_z", "dm_x", "dm_z", "int_x_z", "dd_int_x_z"]
        for col in expected_cols:
            assert col in result.columns, f"Column {col} not found"

        # Check demeaned variables have zero mean within units
        for unit in result.index.get_level_values(0).unique():
            unit_data = result.loc[unit]
            assert abs(unit_data["dm_x"].mean()) < 1e-10
            assert abs(unit_data["dm_z"].mean()) < 1e-10

    def test_interaction_calculation(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time_id": [1, 2, 1, 2],
                "x": [1.0, 3.0, 2.0, 4.0],  # Unit 1 mean = 2, Unit 2 mean = 3
                "z": [2.0, 4.0, 1.0, 3.0],  # Unit 1 mean = 3, Unit 2 mean = 2
            }
        )
        data = data.set_index(["unit_id", "time_id"])
        result = create_double_demeaned_interaction(data, "x", "z", "unit_id", verbose=False)

        # Unit 1: dm_x = [-1, 1], dm_z = [-1, 1], dd_int = [1, 1]
        unit1 = result.loc[1]
        np.testing.assert_array_almost_equal(unit1["dd_int_x_z"].values, [1.0, 1.0])

        # Unit 2: dm_x = [-1, 1], dm_z = [-1, 1], dd_int = [1, 1]
        unit2 = result.loc[2]
        np.testing.assert_array_almost_equal(unit2["dd_int_x_z"].values, [1.0, 1.0])

    def test_standard_interaction_is_product(self, simple_panel: pd.DataFrame) -> None:
        data = simple_panel.set_index(["unit_id", "time_id"])
        result = create_double_demeaned_interaction(data, "x", "z", "unit_id", verbose=False)

        expected = data["x"] * data["z"]
        pd.testing.assert_series_equal(result["int_x_z"], expected, check_names=False)


class TestEstimateFEModels:
    """Tests for estimate_fe_models (via the full pipeline)."""

    def test_returns_comparison_result(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
            w_vars=["control1"],
        )
        result = analysis.run_analysis(verbose=False, run_hausman=False)
        assert isinstance(result.comparison, ComparisonResult)
        assert isinstance(result.comparison.table, pd.DataFrame)
        assert len(result.comparison.table) > 0

    def test_comparison_table_includes_interaction_row(
        self, balanced_panel: pd.DataFrame
    ) -> None:
        """Regression test: the interaction row must appear in the comparison."""
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, run_hausman=False)

        variables = result.comparison.table["Variable"].tolist()
        assert "int_x_z" in variables, (
            f"Interaction row missing from comparison table. Variables: {variables}"
        )


class TestPerformHausmanTest:
    """Tests for perform_hausman_test (via the full pipeline)."""

    def test_hausman_returns_result(self, balanced_panel: pd.DataFrame) -> None:
        from dd_ie._types import HausmanResult

        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, run_hausman=True)

        assert result.hausman is not None
        assert isinstance(result.hausman, HausmanResult)
        assert result.hausman.statistic >= 0
        assert 0 <= result.hausman.p_value <= 1
        assert result.hausman.degrees_of_freedom > 0
        assert result.hausman.conclusion in ("SYSTEMATIC_BIAS", "NO_SYSTEMATIC_BIAS")

    def test_hausman_skipped_when_disabled(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, run_hausman=False)
        assert result.hausman is None
