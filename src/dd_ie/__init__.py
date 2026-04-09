"""dd_ie: Double Demeaning for Fixed Effects Interactions.

A Python implementation of the double demeaning technique for unbiased
estimation of interactions in fixed effects regression models.

Reference:
    Giesselmann, M., & Schmidt-Catran, A. W. (2022). Interactions in Fixed
    Effects Regression Models. *Sociological Methods & Research*, 51(3),
    1100-1127.
"""

from __future__ import annotations

from dd_ie._types import AnalysisResult, ComparisonResult, HausmanResult
from dd_ie.core import (
    DoubleDemeanAnalysis,
    create_double_demeaned_interaction,
    estimate_fe_models,
    perform_hausman_test,
)
from dd_ie.utils import (
    check_within_unit_variation,
    filter_units_by_time_periods,
    prepare_panel_data,
    summarize_panel_structure,
    validate_panel_data,
)

__version__ = "0.2.0"
__author__ = "Nikolaos Koutounidis"
__email__ = "nikolaos.koutounidis@ugent.be"

__all__ = [
    "AnalysisResult",
    "ComparisonResult",
    "DoubleDemeanAnalysis",
    "HausmanResult",
    "check_within_unit_variation",
    "create_double_demeaned_interaction",
    "estimate_fe_models",
    "filter_units_by_time_periods",
    "perform_hausman_test",
    "prepare_panel_data",
    "summarize_panel_structure",
    "validate_panel_data",
]
