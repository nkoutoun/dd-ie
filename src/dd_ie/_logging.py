"""Logging configuration for the dd_ie package."""

from __future__ import annotations

import logging


def get_logger() -> logging.Logger:
    """Return the package-level logger."""
    return logging.getLogger("dd_ie")


def configure_verbosity(verbose: bool) -> None:
    """Set up a console handler on the ``dd_ie`` logger.

    Parameters
    ----------
    verbose : bool
        If ``True``, set the logger to INFO level.
        If ``False``, set it to WARNING level.
    """
    logger = get_logger()
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
