"""Integration tests for the dd_ie analysis pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from dd_ie import DoubleDemeanAnalysis
from dd_ie._types import AnalysisResult, HausmanResult


class TestFullPipeline:
    """End-to-end tests for the analysis pipeline."""

    def test_returns_analysis_result(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
            w_vars=["control1"],
        )
        result = analysis.run_analysis(verbose=False)

        assert isinstance(result, AnalysisResult)
        assert result.standard_results is not None
        assert result.dd_results is not None
        assert result.comparison is not None
        assert isinstance(result.hausman, HausmanResult)
        assert isinstance(result.processed_data, pd.DataFrame)

    def test_without_hausman(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, run_hausman=False)

        assert isinstance(result, AnalysisResult)
        assert result.hausman is None

    def test_without_controls(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False)
        assert isinstance(result, AnalysisResult)

    def test_without_centering(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, center_variables=False)
        assert isinstance(result, AnalysisResult)

    def test_comparison_includes_all_variables(
        self, balanced_panel: pd.DataFrame
    ) -> None:
        """The comparison table must include x, z, and the interaction."""
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False, run_hausman=False)

        variables = set(result.comparison.table["Variable"].tolist())
        assert "x" in variables
        assert "z" in variables
        assert "int_x_z" in variables

    def test_to_dict_backward_compat(self, balanced_panel: pd.DataFrame) -> None:
        """AnalysisResult.to_dict() produces the v0.1.0 dict structure."""
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False)
        d = result.to_dict()

        assert "standard_results" in d
        assert "dd_results" in d
        assert "comparison_df" in d
        assert "hausman_test" in d
        assert "processed_data" in d

    def test_silent_mode_no_stdout(
        self, balanced_panel: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """verbose=False should produce no stdout output."""
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        analysis.run_analysis(verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_unbalanced_panel(self, unbalanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=unbalanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False)
        assert isinstance(result, AnalysisResult)

    def test_results_stored_on_instance(self, balanced_panel: pd.DataFrame) -> None:
        analysis = DoubleDemeanAnalysis(
            data=balanced_panel,
            unit_var="unit_id",
            time_var="time_id",
            y_var="y",
            x_var="x",
            z_var="z",
        )
        result = analysis.run_analysis(verbose=False)
        assert analysis.results is result
