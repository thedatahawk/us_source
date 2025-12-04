"""
Utility functions for OECD ICIO Analysis Package.
"""

from .helpers import (
    df_melt,
    generate_sector_groups,
    validate_matrix_dimensions,
    check_matrix_singularity,
    safe_matrix_inverse,
)
from .logging_config import setup_logger

__all__ = [
    'df_melt',
    'generate_sector_groups',
    'validate_matrix_dimensions',
    'check_matrix_singularity',
    'safe_matrix_inverse',
    'setup_logger',
]
