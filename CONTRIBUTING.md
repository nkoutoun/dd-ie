# Contributing to dd-ie

Thank you for your interest in contributing to dd-ie!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/dd-ie.git
   cd dd-ie
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development

### Running Tests

```bash
pytest
pytest --cov=dd_ie --cov-report=term-missing
```

### Linting and Type Checking

```bash
ruff check src/dd_ie tests
mypy src/dd_ie
```

### Code Style

- Follow PEP 8, enforced by `ruff`.
- Use type annotations for all function parameters and return values.
- Use `from __future__ import annotations` at the top of every module.
- Include docstrings (NumPy style) for all public functions and classes.

## Project Structure

```
dd-ie/
├── src/dd_ie/          # Package source code
│   ├── __init__.py     # Public API
│   ├── core.py         # Main analysis classes and functions
│   ├── utils.py        # Data validation and preparation
│   ├── _types.py       # Result dataclasses
│   ├── _logging.py     # Logger configuration
│   └── py.typed        # PEP 561 marker
├── tests/              # Test suite (pytest)
├── docs/examples/      # Example notebooks
├── pyproject.toml      # Build configuration (Hatchling)
└── README.md
```

## Pull Request Process

1. Create a descriptive branch name: `feature/add-bootstrap-tests` or `fix/hausman-edge-case`.
2. Write tests for your changes.
3. Ensure all tests pass and linting is clean.
4. Update documentation and CHANGELOG.md if needed.
5. Submit a pull request with a clear description.

## Questions?

Open an issue or contact nikolaos.koutounidis@ugent.be.
