"""Utility functions for data validation and preparation in double demeaning analysis."""

from __future__ import annotations

import pandas as pd

from dd_ie._logging import get_logger

logger = get_logger()


def _convert_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, handling categorical and object types.

    Parameters
    ----------
    series : pandas.Series
        The series to convert.

    Returns
    -------
    pandas.Series
        Numeric series, with unconvertible values set to NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    if series.dtype.name == "category":
        if series.cat.categories.dtype.kind in "biufc":
            return pd.to_numeric(series, errors="coerce")
        return series.cat.codes.astype(float)

    return pd.to_numeric(series, errors="coerce")


def validate_panel_data(data: pd.DataFrame, unit_var: str, time_var: str) -> pd.DataFrame:
    """Validate panel data structure.

    Parameters
    ----------
    data : pandas.DataFrame
        Input panel dataset.
    unit_var : str
        Name of the unit identifier variable.
    time_var : str
        Name of the time identifier variable.

    Returns
    -------
    pandas.DataFrame
        Validated panel data (unmodified).

    Raises
    ------
    ValueError
        If required variables are missing or contain null values.
    """
    missing_cols = [col for col in [unit_var, time_var] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    if data[unit_var].isnull().any():
        raise ValueError(f"Missing values found in unit variable '{unit_var}'")
    if data[time_var].isnull().any():
        raise ValueError(f"Missing values found in time variable '{time_var}'")

    panel_structure = data.groupby(unit_var)[time_var].count()
    units_with_single_obs = (panel_structure == 1).sum()

    if units_with_single_obs > 0:
        logger.warning(
            "%d units have only 1 observation. "
            "Fixed effects models require multiple observations per unit.",
            units_with_single_obs,
        )

    unique_periods = panel_structure.unique()
    is_balanced = len(unique_periods) == 1

    logger.info("Panel Data Summary:")
    logger.info("  Total observations: %d", len(data))
    logger.info("  Number of units: %d", data[unit_var].nunique())
    logger.info(
        "  Time periods per unit: %d-%d", panel_structure.min(), panel_structure.max()
    )
    logger.info("  Panel type: %s", "Balanced" if is_balanced else "Unbalanced")

    return data


def check_within_unit_variation(
    data: pd.DataFrame, variable: str, min_variation_threshold: float = 0.05
) -> dict[str, object]:
    """Check within-unit variation for a variable in panel data.

    This is crucial for fixed effects models as variables with no within-unit
    variation will be perfectly collinear with the fixed effects.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel data with MultiIndex ``[unit, time]`` or regular columns.
    variable : str
        Variable name to check.
    min_variation_threshold : float, optional
        Minimum fraction of units that must have variation (default: 0.05).

    Returns
    -------
    dict[str, object]
        Dictionary with variation statistics including keys ``variable``,
        ``total_units``, ``units_with_variation``, ``units_without_variation``,
        ``pct_with_variation``, ``meets_threshold``, ``threshold_used``,
        ``overall_std``, ``within_unit_std``, ``between_unit_std``,
        ``avg_periods_per_unit``.

    Raises
    ------
    ValueError
        If the variable is not found in the data.
    """
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' not found in data")

    if isinstance(data.index, pd.MultiIndex):
        grouped = data.groupby(level=0)[variable]
    else:
        unit_col = data.columns[0]
        grouped = data.groupby(unit_col)[variable]

    unit_stats = grouped.agg(["count", "std", "min", "max"]).fillna(0)

    units_with_variation = int(
        ((unit_stats["std"] > 1e-10) & (unit_stats["count"] > 1)).sum()
    )
    total_units = len(unit_stats)
    pct_with_variation = units_with_variation / total_units if total_units > 0 else 0.0

    return {
        "variable": variable,
        "total_units": total_units,
        "units_with_variation": units_with_variation,
        "units_without_variation": total_units - units_with_variation,
        "pct_with_variation": pct_with_variation,
        "meets_threshold": pct_with_variation >= min_variation_threshold,
        "threshold_used": min_variation_threshold,
        "overall_std": float(data[variable].std()),
        "within_unit_std": float(unit_stats["std"].mean()),
        "between_unit_std": float(grouped.mean().std()),
        "avg_periods_per_unit": float(unit_stats["count"].mean()),
    }


def prepare_panel_data(
    data: pd.DataFrame, unit_var: str, time_var: str, variables: list[str]
) -> pd.DataFrame:
    """Prepare panel data for analysis including filtering and type conversion.

    Parameters
    ----------
    data : pandas.DataFrame
        Input panel dataset.
    unit_var : str
        Unit identifier variable.
    time_var : str
        Time identifier variable.
    variables : list[str]
        List of analysis variables to prepare.

    Returns
    -------
    pandas.DataFrame
        Prepared panel data with numeric types and no missing values.
    """
    all_vars = [unit_var, time_var] + variables
    available_vars = [var for var in all_vars if var in data.columns]

    missing = set(all_vars) - set(available_vars)
    if missing:
        logger.warning("Variables not found in data: %s", missing)

    df_prep = data[available_vars].copy()

    # Handle missing values
    missing_before = df_prep.isnull().sum().sum()
    if missing_before > 0:
        logger.info("Missing Data Handling:")
        logger.info("  Total missing values: %d", missing_before)

        missing_by_var = df_prep.isnull().sum()
        for var, count in missing_by_var[missing_by_var > 0].items():
            pct = 100 * count / len(df_prep)
            logger.info("  %s: %d (%.1f%%)", var, count, pct)

        df_prep = df_prep.dropna()
        dropped = len(data) - len(df_prep)
        logger.info("  Observations dropped: %d", dropped)
        logger.info("  Final sample size: %d", len(df_prep))

    # Data type conversion
    numeric_vars = [var for var in variables if var in df_prep.columns]
    for var in numeric_vars:
        original_dtype = df_prep[var].dtype
        if not pd.api.types.is_numeric_dtype(df_prep[var]):
            df_prep[var] = _convert_to_numeric(df_prep[var])
            new_dtype = df_prep[var].dtype
            if original_dtype != new_dtype:
                logger.info("  %s: %s -> %s", var, original_dtype, new_dtype)

            new_missing = df_prep[var].isnull().sum()
            if new_missing > 0:
                logger.warning(
                    "%d values in '%s' became NaN during conversion", new_missing, var
                )

    # Final cleanup
    if df_prep.isnull().sum().sum() > 0:
        df_prep = df_prep.dropna()
        logger.info("  Final sample after type conversion: %d", len(df_prep))

    return df_prep


def filter_units_by_time_periods(
    data: pd.DataFrame, unit_var: str, min_periods: int = 3
) -> pd.DataFrame:
    """Filter panel data to keep only units with sufficient time periods.

    Double demeaning requires T > 2 for proper identification.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel dataset.
    unit_var : str
        Unit identifier variable.
    min_periods : int, optional
        Minimum number of time periods required (default: 3).

    Returns
    -------
    pandas.DataFrame
        Filtered dataset containing only units with at least *min_periods* periods.
    """
    unit_periods = data.groupby(unit_var).size()
    units_to_keep = unit_periods[unit_periods >= min_periods].index
    units_dropped = len(unit_periods) - len(units_to_keep)
    obs_dropped = data[~data[unit_var].isin(units_to_keep)].shape[0]

    logger.info("Filtering Units by Time Periods:")
    logger.info("  Minimum periods required: %d", min_periods)
    logger.info("  Units dropped: %d", units_dropped)
    logger.info("  Observations dropped: %d", obs_dropped)

    filtered_data = data[data[unit_var].isin(units_to_keep)].copy()
    logger.info(
        "  Final sample: %d observations, %d units",
        len(filtered_data),
        filtered_data[unit_var].nunique(),
    )

    return filtered_data


def summarize_panel_structure(
    data: pd.DataFrame, unit_var: str, time_var: str
) -> dict[str, object]:
    """Provide comprehensive summary of panel data structure.

    Parameters
    ----------
    data : pandas.DataFrame
        Panel dataset.
    unit_var : str
        Unit identifier variable.
    time_var : str
        Time identifier variable.

    Returns
    -------
    dict[str, object]
        Dictionary with keys ``n_observations``, ``n_units``, ``n_periods``,
        ``min_periods``, ``max_periods``, ``mean_periods``,
        ``units_insufficient``, ``is_balanced``, ``total_missing``,
        ``meets_requirements``.
    """
    n_obs = len(data)
    n_units = data[unit_var].nunique()
    n_periods = data[time_var].nunique()

    periods_per_unit = data.groupby(unit_var)[time_var].nunique()
    min_t = int(periods_per_unit.min())
    max_t = int(periods_per_unit.max())
    mean_t = float(periods_per_unit.mean())

    is_balanced = len(periods_per_unit.unique()) == 1
    units_insufficient = int((periods_per_unit <= 2).sum())

    total_missing = int(data.isnull().sum().sum())

    logger.info("Panel Data Summary:")
    logger.info("  Total observations: %d", n_obs)
    logger.info("  Number of units: %d", n_units)
    logger.info("  Number of time periods: %d", n_periods)
    logger.info("  Periods per unit: %d-%d (mean %.1f)", min_t, max_t, mean_t)
    logger.info("  Balanced panel: %s", "Yes" if is_balanced else "No")

    if units_insufficient > 0:
        pct = 100 * units_insufficient / n_units
        logger.warning(
            "%d units (%.1f%%) have <= 2 periods. "
            "Double demeaning requires T > 2.",
            units_insufficient,
            pct,
        )

    if total_missing > 0:
        logger.info("  Total missing values: %d", total_missing)
    else:
        logger.info("  No missing data detected.")

    return {
        "n_observations": n_obs,
        "n_units": n_units,
        "n_periods": n_periods,
        "min_periods": min_t,
        "max_periods": max_t,
        "mean_periods": mean_t,
        "units_insufficient": units_insufficient,
        "is_balanced": is_balanced,
        "total_missing": total_missing,
        "meets_requirements": units_insufficient == 0 and total_missing == 0,
    }
