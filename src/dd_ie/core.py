"""Core functionality for double demeaning analysis in fixed effects models.

This module implements the methodology from:

    Giesselmann, M., & Schmidt-Catran, A. W. (2022). Interactions in Fixed
    Effects Regression Models. *Sociological Methods & Research*, 51(3),
    1100-1127.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from scipy import stats

from dd_ie._logging import configure_verbosity, get_logger
from dd_ie._types import AnalysisResult, ComparisonResult, HausmanResult
from dd_ie.utils import _convert_to_numeric, check_within_unit_variation, validate_panel_data

logger = get_logger()


class DoubleDemeanAnalysis:
    """Perform double demeaning analysis on panel data.

    Implements the methodology from Giesselmann & Schmidt-Catran (2022) for
    unbiased estimation of interactions in fixed effects regression models.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel dataset with unit and time identifiers.
    unit_var : str
        Name of the unit identifier variable.
    time_var : str
        Name of the time identifier variable.
    y_var : str
        Name of the dependent variable.
    x_var : str
        First interacting variable.
    z_var : str
        Second interacting variable.
    w_vars : list[str] or None, optional
        List of control variable names (default: None).

    Raises
    ------
    ValueError
        If any specified variable is not found in *data*.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_var: str,
        time_var: str,
        y_var: str,
        x_var: str,
        z_var: str,
        w_vars: list[str] | None = None,
    ) -> None:
        self.unit_var = unit_var
        self.time_var = time_var
        self.y_var = y_var
        self.x_var = x_var
        self.z_var = z_var
        self.w_vars = w_vars or []

        # Validate that all analysis variables exist
        all_vars = [y_var, x_var, z_var] + self.w_vars
        missing = [v for v in all_vars if v not in data.columns]
        if missing:
            raise ValueError(f"Variables not found in data: {missing}")

        # Validate panel structure
        self.data = validate_panel_data(data, unit_var, time_var)
        self._prepare_data()

        # Results storage
        self.results: AnalysisResult | None = None

    def _prepare_data(self) -> None:
        """Prepare data for analysis including type conversion and indexing."""
        analysis_vars = [self.y_var, self.x_var, self.z_var] + self.w_vars
        for var in analysis_vars:
            if var in self.data.columns and not pd.api.types.is_numeric_dtype(self.data[var]):
                original_dtype = self.data[var].dtype
                self.data[var] = _convert_to_numeric(self.data[var])
                if not pd.api.types.is_numeric_dtype(self.data[var]):
                    logger.warning("Could not convert '%s' to numeric", var)
                else:
                    logger.debug(
                        "Converted '%s': %s -> %s", var, original_dtype, self.data[var].dtype
                    )

        if not isinstance(self.data.index, pd.MultiIndex):
            self.data = self.data.set_index([self.unit_var, self.time_var])

        logger.info(
            "Data prepared: %d observations, %d units",
            len(self.data),
            self.data.index.get_level_values(0).nunique(),
        )

    def run_analysis(
        self,
        center_variables: bool = True,
        run_hausman: bool = True,
        verbose: bool = True,
    ) -> AnalysisResult:
        """Run the complete double demeaning analysis.

        Parameters
        ----------
        center_variables : bool, optional
            Whether to apply grand mean centering (default: True).
        run_hausman : bool, optional
            Whether to run the Hausman test (default: True).
        verbose : bool, optional
            Whether to print detailed output (default: True).

        Returns
        -------
        AnalysisResult
            Dataclass containing analysis results.
        """
        configure_verbosity(verbose)

        if verbose:
            self._print_header()

        # Step 1: Data preparation checks
        if verbose:
            logger.info("Step 1: Data Preparation")
            self._print_data_info()

        # Step 2: Grand mean centering
        df_work = self.data.copy()
        if center_variables:
            if verbose:
                logger.info("Step 2: Grand Mean Centering")
            df_work = self._apply_grand_mean_centering(df_work, verbose)
        elif verbose:
            logger.info("Step 2: Skipping grand mean centering")

        # Step 3: Create double demeaned interaction
        if verbose:
            logger.info("Step 3: Double Demeaning Implementation")
        df_dd = create_double_demeaned_interaction(
            df_work, self.x_var, self.z_var, self.unit_var, verbose
        )

        # Step 4: Model estimation and comparison
        if verbose:
            logger.info("Step 4: Model Estimation and Comparison")
        standard_results, dd_results, comparison = estimate_fe_models(
            df_dd, self.y_var, self.x_var, self.z_var, self.w_vars, verbose
        )

        # Step 5: Hausman test
        hausman_result: HausmanResult | None = None
        if run_hausman:
            if verbose:
                logger.info("Step 5: Hausman Test for Systematic Differences")
            hausman_result = perform_hausman_test(
                standard_results, dd_results, self.x_var, self.z_var, verbose
            )

        self.results = AnalysisResult(
            standard_results=standard_results,
            dd_results=dd_results,
            comparison=comparison,
            hausman=hausman_result,
            processed_data=df_dd,
        )

        if verbose:
            logger.info("ANALYSIS COMPLETE")

        return self.results

    def _print_header(self) -> None:
        """Print analysis header."""
        logger.info("STARTING DOUBLE DEMEANING ANALYSIS")
        logger.info(
            "Dataset: %d observations, %d variables", len(self.data), len(self.data.columns)
        )
        logger.info(
            "Panel structure: %s (units) x %s (time)", self.unit_var, self.time_var
        )
        logger.info(
            "Analysis: %s ~ %s x %s + controls", self.y_var, self.x_var, self.z_var
        )

    def _print_data_info(self) -> None:
        """Print data preparation information."""
        panel_info = self.data.groupby(level=0).size()
        units_insufficient = int((panel_info <= 2).sum())

        logger.info(
            "  Panel: %d units, %d-%d periods per unit (mean %.1f)",
            len(panel_info),
            panel_info.min(),
            panel_info.max(),
            panel_info.mean(),
        )

        if units_insufficient > 0:
            logger.warning(
                "%d units have <= 2 periods. "
                "Double demeaning requires T > 2. Consider filtering.",
                units_insufficient,
            )

    def _apply_grand_mean_centering(
        self, df: pd.DataFrame, verbose: bool
    ) -> pd.DataFrame:
        """Apply grand mean centering to analysis variables."""
        variables_to_center = [self.y_var, self.x_var, self.z_var] + self.w_vars

        for var in variables_to_center:
            if var in df.columns:
                original_mean = df[var].mean()
                df[var] = df[var] - original_mean
                if verbose:
                    logger.info(
                        "  %s: mean before = %.5f, mean after = %.10f",
                        var,
                        original_mean,
                        df[var].mean(),
                    )

        return df

    def __repr__(self) -> str:
        return (
            f"DoubleDemeanAnalysis(y={self.y_var!r}, "
            f"x={self.x_var!r}, z={self.z_var!r}, "
            f"w={self.w_vars!r}, "
            f"n_obs={len(self.data)}, "
            f"n_units={self.data.index.get_level_values(0).nunique()})"
        )


def create_double_demeaned_interaction(
    df: pd.DataFrame,
    x_var: str,
    z_var: str,
    unit_var: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create the double-demeaned interaction term.

    This is the core innovation of Giesselmann & Schmidt-Catran (2022):

    1. First demean each variable within units.
    2. Then create the interaction from the demeaned variables.

    The resulting interaction term, when used in a fixed effects model (which
    performs an additional within-unit demeaning), yields the dd-IE estimator
    from equation (12) of the paper.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel data with MultiIndex ``[unit, time]``.
    x_var : str
        First interacting variable.
    z_var : str
        Second interacting variable.
    unit_var : str
        Unit identifier (used for grouping).
    verbose : bool, optional
        Whether to print detailed output (default: True).

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns: ``mean_{x}``, ``mean_{z}``,
        ``dm_{x}``, ``dm_{z}``, ``int_{x}_{z}``, ``dd_int_{x}_{z}``.
    """
    df_dd = df.copy()

    if verbose:
        logger.info("CREATING DOUBLE DEMEANED INTERACTION")

    # Step 1: Create within-unit means for each variable
    for var in [x_var, z_var]:
        mean_name = f"mean_{var}"
        df_dd[mean_name] = df_dd.groupby(level=0)[var].transform("mean")
        if verbose:
            logger.info("  %s: unit means calculated -> %s", var, mean_name)

    # Step 2: Create demeaned variables (within-unit deviations)
    for var in [x_var, z_var]:
        dm_name = f"dm_{var}"
        df_dd[dm_name] = df_dd[var] - df_dd[f"mean_{var}"]
        if verbose:
            logger.info(
                "  %s -> %s: mean = %.10f (should be ~0), std = %.5f",
                var,
                dm_name,
                df_dd[dm_name].mean(),
                df_dd[dm_name].std(),
            )

    # Step 3: Create interaction terms
    dm_x = f"dm_{x_var}"
    dm_z = f"dm_{z_var}"
    interaction_name = f"int_{x_var}_{z_var}"
    dd_interaction_name = f"dd_{interaction_name}"

    # Standard interaction (X * Z)
    df_dd[interaction_name] = df_dd[x_var] * df_dd[z_var]

    # Double demeaned interaction (dm_X * dm_Z)
    df_dd[dd_interaction_name] = df_dd[dm_x] * df_dd[dm_z]

    if verbose:
        logger.info(
            "  Double demeaned interaction: %s = %s * %s",
            dd_interaction_name,
            dm_x,
            dm_z,
        )
        logger.info(
            "  Mean of dd interaction: %.10f, Std: %.5f",
            df_dd[dd_interaction_name].mean(),
            df_dd[dd_interaction_name].std(),
        )
        correlation = df_dd[interaction_name].corr(df_dd[dd_interaction_name])
        logger.info("  Correlation with standard interaction: %.5f", correlation)

    return df_dd


def estimate_fe_models(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    z_var: str,
    w_vars: list[str],
    verbose: bool = True,
) -> tuple[object, object, ComparisonResult]:
    """Estimate standard FE and double-demeaned FE models and compare results.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel data with interaction terms created by
        :func:`create_double_demeaned_interaction`.
    y_var : str
        Dependent variable name.
    x_var : str
        First interacting variable.
    z_var : str
        Second interacting variable.
    w_vars : list[str]
        Control variable names.
    verbose : bool, optional
        Whether to print detailed output (default: True).

    Returns
    -------
    tuple[object, object, ComparisonResult]
        Standard FE results, double-demeaned FE results, and comparison.
    """
    if verbose:
        logger.info("CHECKING WITHIN-UNIT VARIATION")

    all_vars = [y_var, x_var, z_var] + w_vars
    filtered_w_vars: list[str] = []

    for var in all_vars:
        if var not in df.columns:
            continue
        variation_check = check_within_unit_variation(df, var)
        units_with = variation_check["units_with_variation"]
        total = variation_check["total_units"]
        pct = 100 * units_with / total  # type: ignore[operator]

        if verbose:
            logger.info("  %s: %d/%d units have variation (%.1f%%)", var, units_with, total, pct)

        if var in w_vars and pct < 5.0:
            if verbose:
                logger.info("    EXCLUDED: %s - insufficient within-unit variation for FE", var)
        elif var in w_vars:
            filtered_w_vars.append(var)
        elif pct < 10.0 and verbose:
            logger.warning("  %s has limited within-unit variation (%.1f%%)", var, pct)

    # Model specifications
    interaction_std = f"int_{x_var}_{z_var}"
    interaction_dd = f"dd_int_{x_var}_{z_var}"

    exog_vars_std = [x_var, z_var, interaction_std] + filtered_w_vars
    exog_vars_dd = [x_var, z_var, interaction_dd] + filtered_w_vars

    if verbose:
        logger.info("Variables in analysis:")
        logger.info("  Dependent: %s", y_var)
        logger.info("  Interactors: %s x %s", x_var, z_var)
        logger.info("  Controls: %s", filtered_w_vars)

    # Standard FE model
    if verbose:
        logger.info("STANDARD FIXED EFFECTS MODEL")
        logger.info("  Model: %s ~ %s + entity_fe", y_var, " + ".join(exog_vars_std))
        logger.info(
            "  Note: conventional approach, may be biased when both variables vary within units."
        )

    standard_model = PanelOLS(df[y_var], df[exog_vars_std], entity_effects=True)
    standard_results = standard_model.fit(cov_type="clustered", cluster_entity=True, debiased=True)

    if verbose:
        logger.info("\n%s", standard_results.summary.tables[1])

    # Double demeaned FE model
    if verbose:
        logger.info("DOUBLE DEMEANED FIXED EFFECTS MODEL")
        logger.info("  Model: %s ~ %s + entity_fe", y_var, " + ".join(exog_vars_dd))
        logger.info("  Note: unbiased within-unit interaction estimator (dd-IE).")

    dd_model = PanelOLS(df[y_var], df[exog_vars_dd], entity_effects=True)
    dd_results = dd_model.fit(cov_type="clustered", cluster_entity=True, debiased=True)

    if verbose:
        logger.info("\n%s", dd_results.summary.tables[1])

    comparison = _create_comparison_table(standard_results, dd_results, x_var, z_var, verbose)

    return standard_results, dd_results, comparison


def _create_comparison_table(
    standard_results: Any,
    dd_results: Any,
    x_var: str,
    z_var: str,
    verbose: bool,
) -> ComparisonResult:
    """Create coefficient comparison table between standard and dd FE models."""
    std_params = standard_results.params    std_se = standard_results.std_errors    dd_params = dd_results.params    dd_se = dd_results.std_errors
    int_std = f"int_{x_var}_{z_var}"
    int_dd = f"dd_int_{x_var}_{z_var}"

    # Build explicit variable mapping: (display_name, std_name, dd_name)
    var_mapping: list[tuple[str, str, str]] = []
    for var in std_params.index:
        if var == int_std:
            if int_dd in dd_params.index:
                var_mapping.append((var, var, int_dd))
        elif var in dd_params.index:
            var_mapping.append((var, var, var))

    rows = []
    interaction_diff = 0.0
    for display, std_name, dd_name in var_mapping:
        diff = float(std_params[std_name] - dd_params[dd_name])
        rows.append(
            {
                "Variable": display,
                "Std_FE_Coef": float(std_params[std_name]),
                "Std_FE_SE": float(std_se[std_name]),
                "DD_Coef": float(dd_params[dd_name]),
                "DD_SE": float(dd_se[dd_name]),
                "Difference": diff,
            }
        )
        if display == int_std:
            interaction_diff = diff

    comparison_df = pd.DataFrame(rows)

    if verbose and len(comparison_df) > 0:
        logger.info("MODEL COMPARISON")
        header = (
            f"{'Variable':<20} {'Std FE Coef':>12} {'Std FE SE':>10} "
            f"{'DD Coef':>12} {'DD SE':>10} {'Difference':>12}"
        )
        logger.info(header)
        logger.info("-" * len(header))

        for _, row in comparison_df.iterrows():
            logger.info(
                "%-20s %12.5f %10.5f %12.5f %10.5f %12.5f",
                row["Variable"],
                row["Std_FE_Coef"],
                row["Std_FE_SE"],
                row["DD_Coef"],
                row["DD_SE"],
                row["Difference"],
            )

        logger.info(
            "KEY FINDING - Interaction effect difference: %.6f", interaction_diff
        )

    return ComparisonResult(table=comparison_df, interaction_difference=interaction_diff)


def perform_hausman_test(
    standard_results: Any,
    dd_results: Any,
    x_var: str,
    z_var: str,
    verbose: bool = True,
) -> HausmanResult | None:
    """Perform Hausman test for systematic differences between estimators.

    Tests whether the standard FE interaction estimator (FE-IE) differs
    systematically from the double-demeaned estimator (dd-IE).

    Following the Hausman test convention:
    - *b* (consistent under H0 and Ha): dd-IE coefficients.
    - *B* (efficient under H0, inconsistent under Ha): FE-IE coefficients.

    Parameters
    ----------
    standard_results : object
        Results from the standard FE model.
    dd_results : object
        Results from the double-demeaned FE model.
    x_var : str
        First interacting variable name.
    z_var : str
        Second interacting variable name.
    verbose : bool, optional
        Whether to print detailed output (default: True).

    Returns
    -------
    HausmanResult or None
        Test results, or ``None`` if the test could not be computed.
    """
    try:
        if verbose:
            logger.info("HAUSMAN TEST FOR SYSTEMATIC DIFFERENCES")
            logger.info("  H0: No systematic difference between estimators")
            logger.info("  Ha: Standard FE estimator is biased")

        b_std = standard_results.params        b_dd = dd_results.params        V_std = standard_results.cov        V_dd = dd_results.cov
        int_std_name = f"int_{x_var}_{z_var}"
        int_dd_name = f"dd_int_{x_var}_{z_var}"

        # Build coefficient vectors with proper interaction mapping
        common_vars: list[str] = []
        b_std_mapped: list[float] = []
        b_dd_mapped: list[float] = []

        for var in b_std.index:
            if var == int_std_name and int_dd_name in b_dd.index:
                common_vars.append(var)
                b_std_mapped.append(float(b_std[var]))
                b_dd_mapped.append(float(b_dd[int_dd_name]))
            elif var in b_dd.index and var != int_std_name:
                common_vars.append(var)
                b_std_mapped.append(float(b_std[var]))
                b_dd_mapped.append(float(b_dd[var]))

        if len(common_vars) == 0:
            logger.error("No common coefficients found for Hausman test")
            return None

        # Extract variance submatrices
        std_indices = [
            b_std.index.get_loc(int_std_name) if var == int_std_name else b_std.index.get_loc(var)
            for var in common_vars
        ]
        dd_indices = [
            b_dd.index.get_loc(int_dd_name) if var == int_std_name else b_dd.index.get_loc(var)
            for var in common_vars
        ]

        V_std_sub = V_std.iloc[std_indices, std_indices].values
        V_dd_sub = V_dd.iloc[dd_indices, dd_indices].values

        b_std_vec = np.array(b_std_mapped)
        b_dd_vec = np.array(b_dd_mapped)

        if verbose:
            logger.info("  Testing %d common coefficients: %s", len(common_vars), common_vars)

        # dd_IE is consistent (b), FE_IE is efficient under H0 (B)
        diff = b_dd_vec - b_std_vec
        V_diff = V_dd_sub - V_std_sub

        # Check positive definiteness
        eigenvals = np.linalg.eigvals(V_diff)
        pos_def = bool(np.all(eigenvals > 1e-10))

        if verbose:
            logger.info("")
            logger.info(
                "             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))"
            )
            logger.info("             |     dd_IE        FE_IE        Difference       Std. err.")
            logger.info("-------------+----------------------------------------------------------------")

            for i, var in enumerate(common_vars):
                var_display = var[:12] if len(var) <= 12 else var[:9] + "~" + var[-2:]
                if not pos_def and V_diff[i, i] <= 0:
                    se_display = "          ."
                else:
                    se_diff = np.sqrt(abs(V_diff[i, i]))
                    se_display = f"{se_diff:>11.4f}"
                logger.info(
                    "%12s | %11.5f   %11.5f   %11.5f   %s",
                    var_display,
                    b_dd_vec[i],
                    b_std_vec[i],
                    diff[i],
                    se_display,
                )

            logger.info("-------------+----------------------------------------------------------------")
            logger.info("  b = Consistent under H0 and Ha; obtained from dd_IE.")
            logger.info("  B = Inconsistent under Ha, efficient under H0; obtained from FE_IE.")

        # Compute test statistic
        hausman_stat: float
        if pos_def:
            try:
                V_diff_inv = np.linalg.inv(V_diff)
                hausman_stat = float(diff.T @ V_diff_inv @ diff)
            except np.linalg.LinAlgError:
                pos_def = False

        if not pos_def:
            # Robust computation for non-positive definite matrices
            # Approach 1: Eigendecomposition
            eigenvals_real, eigenvecs = np.linalg.eigh(V_diff)
            max_eigenval: float = float(np.max(np.abs(eigenvals_real)))
            tolerance = max_eigenval * len(V_diff) * np.finfo(float).eps
            valid_mask = eigenvals_real > tolerance

            hausman_stat_1 = np.inf
            if np.sum(valid_mask) > 0:
                eigenvals_pos = eigenvals_real[valid_mask]
                eigenvecs_pos = eigenvecs[:, valid_mask]
                V_diff_ginv = eigenvecs_pos @ np.diag(1 / eigenvals_pos) @ eigenvecs_pos.T
                hausman_stat_1 = float(diff.T @ V_diff_ginv @ diff)

            # Approach 2: SVD
            U, s, Vh = np.linalg.svd(V_diff, full_matrices=False)
            tolerance_2 = np.max(s) * 1e-10
            s_inv = np.where(s > tolerance_2, 1 / s, 0)
            V_diff_ginv_2 = (Vh.T * s_inv) @ Vh
            hausman_stat_2 = float(diff.T @ V_diff_ginv_2 @ diff)

            # Choose the more stable result
            if hausman_stat_1 != np.inf and not np.isnan(hausman_stat_1):
                hausman_stat = hausman_stat_1
            else:
                hausman_stat = hausman_stat_2

            # Fallback only for clearly invalid results
            if hausman_stat < 0 or np.isnan(hausman_stat) or np.isinf(hausman_stat):
                V_diff_ginv_3 = np.linalg.pinv(V_diff, rcond=1e-10)
                hausman_stat = float(diff.T @ V_diff_ginv_3 @ diff)

        degrees_of_freedom = len(common_vars)
        p_value = float(1 - stats.chi2.cdf(hausman_stat, df=degrees_of_freedom))

        conclusion = "SYSTEMATIC_BIAS" if p_value < 0.05 else "NO_SYSTEMATIC_BIAS"

        if verbose:
            logger.info("")
            logger.info("  Test: H0 = difference in coefficients not systematic")
            logger.info(
                "  chi2(%d) = (b-B)'[(V_b-V_B)^(-1)](b-B) = %.2f",
                degrees_of_freedom,
                hausman_stat,
            )
            logger.info("  Prob > chi2 = %.4f", p_value)
            if not pos_def:
                logger.info("  (V_b-V_B is not positive definite)")

            if p_value < 0.05:
                logger.info(
                    "  REJECT H0 at 5%% level (p = %.4f < 0.05). "
                    "Evidence of systematic differences. "
                    "Prefer double-demeaned estimator.",
                    p_value,
                )
            else:
                logger.info(
                    "  FAIL TO REJECT H0 at 5%% level (p = %.4f >= 0.05). "
                    "No systematic differences detected. "
                    "Standard FE is more efficient.",
                    p_value,
                )

        return HausmanResult(
            statistic=hausman_stat,
            p_value=p_value,
            degrees_of_freedom=degrees_of_freedom,
            coefficient_differences=diff,
            common_variables=common_vars,
            conclusion=conclusion,
            positive_definite=pos_def,
        )

    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.error("Hausman test failed: %s", exc)
        return None
