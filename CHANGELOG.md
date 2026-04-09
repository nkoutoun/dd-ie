# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-09

### Fixed
- **Comparison table now includes the interaction row.** In v0.1.0, the interaction coefficient was silently omitted from the comparison table due to a variable name mapping bug.
- **Hausman test no longer caps the test statistic at 100.** Large but valid chi-square values are now preserved. The fallback to generalized inverse now only triggers for negative, NaN, or infinite values.
- **Variable shadowing:** renamed internal `df` variable to `degrees_of_freedom` in the Hausman test to avoid shadowing the built-in.

### Added
- **Typed result dataclasses:** `AnalysisResult`, `ComparisonResult`, `HausmanResult` replace raw dictionaries. All have a `.to_dict()` method for backward compatibility.
- **`py.typed` marker** for PEP 561 compliance.
- **`__repr__`** on `DoubleDemeanAnalysis` for better debugging.
- **Early variable validation** in `DoubleDemeanAnalysis.__init__` -- raises `ValueError` with a clear message if analysis variables are missing.
- `tests/test_integration.py` -- end-to-end pipeline tests.
- `tests/test_types.py` -- dataclass tests.
- `tests/conftest.py` -- shared test fixtures.
- GitHub Actions CI workflow (test matrix: Python 3.9-3.13).
- GitHub Actions publish workflow (PyPI trusted publishing on version tags).

### Changed
- **`print()` replaced with `logging`** module throughout. Configure via `logging.getLogger("dd_ie")`. The `verbose` parameter still works as before.
- **`check_within_unit_variation` now raises `ValueError`** for missing variables instead of returning a dict with an `"error"` key. **(Breaking)**
- **Restructured to `src/` layout** with `pyproject.toml` (Hatchling build backend).
- Replaced `setup.py` with `pyproject.toml`.
- Replaced `black`+`flake8` with `ruff`.
- Minimum Python version raised from 3.8 to 3.9.
- Broad `except Exception` replaced with specific exception types.
- Deduplicated type conversion logic into `_convert_to_numeric()`.

### Removed
- `setup.py` (replaced by `pyproject.toml`).
- Emoji characters from all log messages.

## [0.1.0] - 2025-01-26

### Added
- Initial release of dd_ie package.
- `DoubleDemeanAnalysis` class for complete double demeaning workflow.
- Core functions: `create_double_demeaned_interaction`, `estimate_fe_models`, `perform_hausman_test`.
- Utility functions for panel data validation and preparation.
- Hausman test with robust handling of non-positive definite matrices.
- Test suite with pytest.
- MIT license.

## References

Giesselmann, M., & Schmidt-Catran, A. W. (2022). Interactions in Fixed Effects
Regression Models. *Sociological Methods & Research*, 51(3), 1100-1127.
