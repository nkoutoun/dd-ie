# dd-ie: Double Demeaning for Fixed Effects Interactions

[![CI](https://github.com/nkoutoun/dd-ie/actions/workflows/ci.yml/badge.svg)](https://github.com/nkoutoun/dd-ie/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ddinteract)](https://pypi.org/project/ddinteract/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typed](https://img.shields.io/badge/typing-typed-blue)](https://peps.python.org/pep-0561/)

A Python implementation of the double demeaning technique for unbiased estimation of interactions in fixed effects regression models.

## The Problem

Standard fixed effects interactions (`X x Z`) may be biased when both variables vary within units and correlate with unobserved unit-specific moderators. The standard FE interaction estimator (FE-IE) uses between-unit differences in the effects of the interacted variables, making it susceptible to omitted variable bias from time-constant unobserved moderators.

## The Solution

**Double demeaning** (dd-IE) provides an unbiased within-unit interaction estimator:

1. Demean each variable within units: `X* = X - mean_i(X)`, `Z* = Z - mean_i(Z)`
2. Create the interaction from demeaned variables: `X* x Z*`
3. Use this in a standard FE regression (which demeans it again)

This eliminates all between-unit heterogeneity from the interaction term.

## Installation

```bash
pip install ddinteract
```

The PyPI distribution is named `ddinteract`, but the import name remains `dd_ie`.

For development:

```bash
git clone https://github.com/nkoutoun/dd-ie.git
cd dd-ie
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from dd_ie import DoubleDemeanAnalysis

# Load panel data
df = pd.read_csv("your_data.csv")

# Run analysis
analysis = DoubleDemeanAnalysis(
    data=df,
    unit_var="unit_id",      # Unit identifier
    time_var="time_id",      # Time identifier
    y_var="outcome",         # Dependent variable
    x_var="treatment",       # First interacting variable
    z_var="moderator",       # Second interacting variable
    w_vars=["control1"],     # Control variables (optional)
)

results = analysis.run_analysis()

# Access structured results
print(results.comparison.table)          # Coefficient comparison
print(results.hausman.statistic)         # Hausman test statistic
print(results.hausman.p_value)           # Hausman p-value
print(results.hausman.conclusion)        # "SYSTEMATIC_BIAS" or "NO_SYSTEMATIC_BIAS"
```

## Model Comparison

### Standard FE (potentially biased)

```
Y = b_x * X + b_z * Z + b_xz * (X * Z) + g * W + a_i + e
```

### Double Demeaned FE (unbiased)

```
Y = b_x * X + b_z * Z + b_xz * (X* x Z*) + g * W + a_i + e
where X* = X - mean_i(X), Z* = Z - mean_i(Z)
```

## Result Types

The package returns typed dataclasses for all results:

- **`AnalysisResult`** -- complete pipeline output with `.standard_results`, `.dd_results`, `.comparison`, `.hausman`, `.processed_data`
- **`ComparisonResult`** -- coefficient comparison `.table` (DataFrame) and `.interaction_difference`
- **`HausmanResult`** -- test `.statistic`, `.p_value`, `.degrees_of_freedom`, `.conclusion`, `.positive_definite`

All result types have a `.to_dict()` method for backward compatibility.

## Advanced Usage

```python
from dd_ie import (
    create_double_demeaned_interaction,
    estimate_fe_models,
    perform_hausman_test,
)

# Step-by-step analysis
df_processed = create_double_demeaned_interaction(df, "x", "z", "unit_id")
standard_results, dd_results, comparison = estimate_fe_models(
    df_processed, "y", "x", "z", ["control1"]
)
hausman = perform_hausman_test(standard_results, dd_results, "x", "z")
```

## Data Requirements

- **Panel structure**: unit and time identifiers
- **Minimum periods**: T > 2 per unit for identification
- **Within-unit variation**: both X and Z must vary within units
- **Format**: long format with one row per unit-time observation

## Interpreting Results

- **Hausman test p >= 0.05**: no systematic bias detected; standard FE is more efficient
- **Hausman test p < 0.05**: systematic bias detected; prefer the double-demeaned estimator

## Citation

If you use this package, please cite the original methodology:

```bibtex
@article{giesselmann2022interactions,
  title={Interactions in Fixed Effects Regression Models},
  author={Giesselmann, Marco and Schmidt-Catran, Alexander W},
  journal={Sociological Methods \& Research},
  volume={51},
  number={3},
  pages={1100--1127},
  year={2022},
  publisher={SAGE Publications},
  doi={10.1177/0049124120914934}
}
```

## License

MIT License
