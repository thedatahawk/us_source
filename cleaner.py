"""
Data cleaning and preprocessing utilities for OECD ICIO data.

This module provides functions to clean, validate, and preprocess ICIO data
before analysis, including handling missing values, outliers, and extracting
country-sector components.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


def extract_country_sector(
    identifier: str,
    separator: str = '_'
) -> Tuple[str, str]:
    """
    Extract country and sector codes from combined identifier.

    OECD ICIO uses combined identifiers like 'usa_c10t12' where:
    - 'usa' is the country code
    - 'c10t12' is the industry code

    Parameters
    ----------
    identifier : str
        Combined country-sector identifier (e.g., 'usa_c10t12').
    separator : str, optional
        Separator character, by default '_'.

    Returns
    -------
    country : str
        Country code (e.g., 'usa').
    sector : str
        Sector/industry code (e.g., 'c10t12').

    Examples
    --------
    >>> country, sector = extract_country_sector('usa_c10t12')
    >>> print(country, sector)
    usa c10t12
    """
    parts = identifier.split(separator, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid identifier format: {identifier}")
    return parts[0], parts[1]


def clean_icio_data(
    df: pd.DataFrame,
    remove_negatives: bool = False,
    fill_missing: bool = True,
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Clean and preprocess ICIO DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ICIO DataFrame.
    remove_negatives : bool, optional
        Whether to replace negative values with zero, by default False.
    fill_missing : bool, optional
        Whether to fill missing values, by default True.
    fill_value : float, optional
        Value to use for filling missing data, by default 0.0.

    Returns
    -------
    pd.DataFrame
        Cleaned ICIO DataFrame.

    Notes
    -----
    OECD ICIO data is generally clean, but this function provides safeguards
    against common data quality issues that can arise during processing or
    from alternative data sources.
    """
    df_clean = df.copy()

    # Handle missing values
    if fill_missing:
        n_missing = df_clean.isna().sum().sum()
        if n_missing > 0:
            logger.warning(f"Filling {n_missing} missing values with {fill_value}")
            df_clean.fillna(fill_value, inplace=True)

    # Handle negative values (rarely occur but can cause issues in log operations)
    if remove_negatives:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_neg = (df_clean[col] < 0).sum()
            if n_neg > 0:
                logger.warning(f"Replacing {n_neg} negative values in {col}")
                df_clean.loc[df_clean[col] < 0, col] = 0.0

    return df_clean


def validate_accounting_identity(
    T: pd.DataFrame,
    X: pd.Series,
    F: pd.Series,
    tolerance: float = 0.01
) -> bool:
    """
    Validate the fundamental IO accounting identity: X = T.sum(axis=1) + F.

    This is the "use accounting identity" from Baldwin et al. (2022):
    Gross output equals intermediate use plus final demand.

    Parameters
    ----------
    T : pd.DataFrame
        Intermediate use matrix.
    X : pd.Series
        Gross output vector.
    F : pd.Series
        Final demand vector.
    tolerance : float, optional
        Maximum relative error allowed, by default 0.01 (1%).

    Returns
    -------
    bool
        True if identity holds within tolerance, False otherwise.

    Notes
    -----
    The use accounting identity is fundamental to IO analysis:
        X_i = Σ_j T_ij + F_i

    where X_i is gross output of sector i, T_ij is intermediate sales from i to j,
    and F_i is final demand for sector i's output.

    References
    ----------
    Baldwin, Freeman, & Theodorakopoulos (2022): Section 2.1 discusses
    the use and cost accounting identities in IO tables.
    """
    T_row_sum = T.sum(axis=1)
    X_computed = T_row_sum + F

    # Compute relative error
    relative_error = np.abs(X_computed - X) / (X + 1e-10)

    # Check if all errors are within tolerance
    if (relative_error > tolerance).any():
        n_violations = (relative_error > tolerance).sum()
        max_error = relative_error.max()
        logger.error(
            f"Accounting identity violated for {n_violations} sectors "
            f"(max error: {max_error:.2%})"
        )
        return False

    logger.debug("Accounting identity validated successfully")
    return True


def filter_manufacturing_sectors(
    df: pd.DataFrame,
    sector_col: str = 's',
    manufacturing_codes: Optional[list] = None
) -> pd.DataFrame:
    """
    Filter DataFrame to include only manufacturing sectors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sector codes.
    sector_col : str, optional
        Column name containing sector codes, by default 's'.
    manufacturing_codes : list, optional
        List of manufacturing sector codes. If None, uses OECD standard codes.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only manufacturing sectors.
    """
    if manufacturing_codes is None:
        # Use config for manufacturing codes (supports both old and new formats)
        import config
        manufacturing_codes = config.MANUFACTURING_CODES_ALL

    df_filtered = df[df[sector_col].isin(manufacturing_codes)].copy()
    logger.debug(f"Filtered to {len(df_filtered)} manufacturing sectors")

    return df_filtered
