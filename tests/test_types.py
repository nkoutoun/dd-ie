"""Tests for result dataclasses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dd_ie._types import AnalysisResult, ComparisonResult, HausmanResult


class TestHausmanResult:
    """Tests for HausmanResult dataclass."""

    def test_to_dict(self) -> None:
        result = HausmanResult(
            statistic=3.14,
            p_value=0.42,
            degrees_of_freedom=3,
            coefficient_differences=np.array([0.1, -0.2, 0.05]),
            common_variables=["x", "z", "int_x_z"],
            conclusion="NO_SYSTEMATIC_BIAS",
            positive_definite=True,
        )
        d = result.to_dict()
        assert d["hausman_statistic"] == 3.14
        assert d["p_value"] == 0.42
        assert d["degrees_of_freedom"] == 3
        assert d["conclusion"] == "NO_SYSTEMATIC_BIAS"
        assert d["positive_definite"] is True

    def test_frozen(self) -> None:
        result = HausmanResult(
            statistic=1.0,
            p_value=0.5,
            degrees_of_freedom=1,
            coefficient_differences=np.array([0.1]),
            common_variables=["x"],
            conclusion="NO_SYSTEMATIC_BIAS",
            positive_definite=True,
        )
        with pytest.raises(AttributeError):
            result.statistic = 999.0  # type: ignore[misc]


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_to_dict(self) -> None:
        table = pd.DataFrame({"Variable": ["x"], "Difference": [0.1]})
        result = ComparisonResult(table=table, interaction_difference=0.05)
        d = result.to_dict()
        assert "table" in d
        assert d["interaction_difference"] == 0.05

    def test_frozen(self) -> None:
        table = pd.DataFrame({"Variable": ["x"]})
        result = ComparisonResult(table=table, interaction_difference=0.0)
        with pytest.raises(AttributeError):
            result.interaction_difference = 1.0  # type: ignore[misc]


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_to_dict_with_hausman(self) -> None:
        hausman = HausmanResult(
            statistic=2.0,
            p_value=0.3,
            degrees_of_freedom=2,
            coefficient_differences=np.array([0.1, -0.1]),
            common_variables=["x", "z"],
            conclusion="NO_SYSTEMATIC_BIAS",
            positive_definite=True,
        )
        comparison = ComparisonResult(
            table=pd.DataFrame({"Variable": ["x"]}),
            interaction_difference=0.01,
        )
        result = AnalysisResult(
            standard_results=None,
            dd_results=None,
            comparison=comparison,
            hausman=hausman,
            processed_data=pd.DataFrame(),
        )
        d = result.to_dict()
        assert "comparison_df" in d
        assert d["hausman_test"]["hausman_statistic"] == 2.0

    def test_to_dict_without_hausman(self) -> None:
        comparison = ComparisonResult(
            table=pd.DataFrame({"Variable": ["x"]}),
            interaction_difference=0.0,
        )
        result = AnalysisResult(
            standard_results=None,
            dd_results=None,
            comparison=comparison,
            hausman=None,
            processed_data=pd.DataFrame(),
        )
        d = result.to_dict()
        assert d["hausman_test"] is None
