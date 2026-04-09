"""Result types for double demeaning analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HausmanResult:
    """Result of the Hausman test for systematic differences between estimators.

    Parameters
    ----------
    statistic : float
        Chi-square test statistic.
    p_value : float
        P-value from the chi-square distribution.
    degrees_of_freedom : int
        Number of coefficients tested.
    coefficient_differences : numpy.ndarray
        Vector of coefficient differences (dd_IE - FE_IE).
    common_variables : list[str]
        Names of the common coefficients tested.
    conclusion : str
        Either ``"SYSTEMATIC_BIAS"`` or ``"NO_SYSTEMATIC_BIAS"``.
    positive_definite : bool
        Whether the variance difference matrix was positive definite.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    coefficient_differences: np.ndarray
    common_variables: list[str]
    conclusion: str
    positive_definite: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "hausman_statistic": self.statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "coefficient_differences": self.coefficient_differences,
            "common_variables": self.common_variables,
            "conclusion": self.conclusion,
            "positive_definite": self.positive_definite,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison between standard FE and double-demeaned FE coefficients.

    Parameters
    ----------
    table : pandas.DataFrame
        DataFrame with columns ``Variable``, ``Std_FE_Coef``, ``Std_FE_SE``,
        ``DD_Coef``, ``DD_SE``, ``Difference``.
    interaction_difference : float
        Coefficient difference for the interaction term specifically.
    """

    table: pd.DataFrame
    interaction_difference: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "table": self.table,
            "interaction_difference": self.interaction_difference,
        }


@dataclass
class AnalysisResult:
    """Complete results from a double demeaning analysis.

    Parameters
    ----------
    standard_results : object
        PanelEffectsResults from the standard FE model.
    dd_results : object
        PanelEffectsResults from the double-demeaned FE model.
    comparison : ComparisonResult
        Coefficient comparison between the two models.
    hausman : HausmanResult or None
        Hausman test results, or ``None`` if the test was not run.
    processed_data : pandas.DataFrame
        The processed panel data including demeaned variables and interactions.
    """

    standard_results: Any
    dd_results: Any
    comparison: ComparisonResult
    hausman: HausmanResult | None
    processed_data: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (backward-compatible with v0.1.0)."""
        return {
            "standard_results": self.standard_results,
            "dd_results": self.dd_results,
            "comparison_df": self.comparison.table,
            "hausman_test": self.hausman.to_dict() if self.hausman else None,
            "processed_data": self.processed_data,
        }
