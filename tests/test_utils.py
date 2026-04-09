"""Tests for utility functions of dd_ie package."""

from __future__ import annotations

import pandas as pd
import pytest

from dd_ie.utils import (
    _convert_to_numeric,
    check_within_unit_variation,
    filter_units_by_time_periods,
    summarize_panel_structure,
    validate_panel_data,
)


class TestConvertToNumeric:
    """Tests for the _convert_to_numeric helper."""

    def test_numeric_passthrough(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        result = _convert_to_numeric(s)
        pd.testing.assert_series_equal(result, s)

    def test_object_to_numeric(self) -> None:
        s = pd.Series(["1", "2", "3"])
        result = _convert_to_numeric(s)
        assert pd.api.types.is_numeric_dtype(result)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_categorical_numeric(self) -> None:
        s = pd.Categorical([1, 2, 3])
        result = _convert_to_numeric(pd.Series(s))
        assert pd.api.types.is_numeric_dtype(result)

    def test_unconvertible_becomes_nan(self) -> None:
        s = pd.Series(["a", "b", "c"])
        result = _convert_to_numeric(s)
        assert result.isnull().all()


class TestValidatePanelData:
    """Tests for validate_panel_data."""

    def test_valid_panel_data(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 3, 3],
                "time_id": [1, 2, 1, 2, 1, 2],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        result = validate_panel_data(data, "unit_id", "time_id")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6

    def test_missing_unit_variable(self) -> None:
        data = pd.DataFrame({"time_id": [1, 2], "value": [10, 20]})
        with pytest.raises(ValueError, match="Required columns missing"):
            validate_panel_data(data, "missing_unit", "time_id")

    def test_missing_time_variable(self) -> None:
        data = pd.DataFrame({"unit_id": [1, 1], "value": [10, 20]})
        with pytest.raises(ValueError, match="Required columns missing"):
            validate_panel_data(data, "unit_id", "missing_time")

    def test_missing_values_in_unit_var(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, None, 2],
                "time_id": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        with pytest.raises(ValueError, match="Missing values found in unit variable"):
            validate_panel_data(data, "unit_id", "time_id")

    def test_missing_values_in_time_var(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time_id": [1, None, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        with pytest.raises(ValueError, match="Missing values found in time variable"):
            validate_panel_data(data, "unit_id", "time_id")


class TestCheckWithinUnitVariation:
    """Tests for check_within_unit_variation."""

    def test_variable_with_variation(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "time_id": [1, 2, 3, 1, 2, 3],
                "variable": [1, 2, 3, 4, 5, 6],
            }
        )
        data = data.set_index(["unit_id", "time_id"])
        result = check_within_unit_variation(data, "variable")

        assert result["total_units"] == 2
        assert result["units_with_variation"] == 2
        assert result["units_without_variation"] == 0
        assert result["pct_with_variation"] == 1.0
        assert result["meets_threshold"] is True

    def test_variable_without_variation(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "time_id": [1, 2, 3, 1, 2, 3],
                "variable": [1, 1, 1, 2, 2, 2],
            }
        )
        data = data.set_index(["unit_id", "time_id"])
        result = check_within_unit_variation(data, "variable")

        assert result["units_with_variation"] == 0
        assert result["units_without_variation"] == 2
        assert result["pct_with_variation"] == 0.0
        assert result["meets_threshold"] is False

    def test_missing_variable_raises(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time_id": [1, 2, 1, 2],
                "other_var": [1, 2, 3, 4],
            }
        )
        data = data.set_index(["unit_id", "time_id"])
        with pytest.raises(ValueError, match="not found"):
            check_within_unit_variation(data, "missing_variable")

    def test_custom_threshold(self) -> None:
        # 1 of 2 units has variation = 50%
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time_id": [1, 2, 1, 2],
                "variable": [1, 2, 3, 3],
            }
        )
        data = data.set_index(["unit_id", "time_id"])

        result_low = check_within_unit_variation(data, "variable", min_variation_threshold=0.3)
        assert result_low["meets_threshold"] is True

        result_high = check_within_unit_variation(data, "variable", min_variation_threshold=0.8)
        assert result_high["meets_threshold"] is False


class TestFilterUnitsByTimePeriods:
    """Tests for filter_units_by_time_periods."""

    def test_filtering_units(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 3],
                "time_id": [1, 2, 3, 1, 2, 1],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        result = filter_units_by_time_periods(data, "unit_id", min_periods=3)

        assert len(result) == 3
        assert result["unit_id"].nunique() == 1
        assert result["unit_id"].iloc[0] == 1

    def test_no_filtering_needed(self) -> None:
        data = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "time_id": [1, 2, 3, 1, 2, 3],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
        result = filter_units_by_time_periods(data, "unit_id", min_periods=3)
        assert len(result) == len(data)


class TestSummarizePanelStructure:
    """Tests for summarize_panel_structure."""

    def test_balanced_panel(self, balanced_panel: pd.DataFrame) -> None:
        result = summarize_panel_structure(balanced_panel, "unit_id", "time_id")
        assert result["n_units"] == 20
        assert result["n_observations"] == 160
        assert result["is_balanced"] is True
        assert result["min_periods"] == result["max_periods"]

    def test_unbalanced_panel(self, unbalanced_panel: pd.DataFrame) -> None:
        result = summarize_panel_structure(unbalanced_panel, "unit_id", "time_id")
        assert result["n_units"] == 10
        assert result["is_balanced"] is False
