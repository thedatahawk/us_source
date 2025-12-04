"""
Helper utility functions for OECD ICIO data manipulation and matrix operations.

This module provides reusable functions for data reshaping, sector classification,
and numerically robust matrix operations following best practices from
Baldwin, Freeman, & Theodorakopoulos (2022, 2023).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


def df_melt(
    df: pd.DataFrame,
    indicator_name: str = 'value',
    index_name: str = 'index'
) -> pd.DataFrame:
    """
    Melt a square matrix DataFrame into long format for bilateral analysis.

    This function converts wide-format inter-country input-output matrices
    into long format with explicit 'selling' and 'sourcing' country-sector pairs.
    This format is essential for bilateral exposure analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Square matrix with country-sector pairs as both index and columns.
    indicator_name : str, optional
        Name for the value column in melted DataFrame, by default 'value'.
    index_name : str, optional
        Name for the index (not currently used), by default 'index'.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with columns ['selling', 'sourcing', indicator_name],
        sorted by selling then sourcing.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create simple 2x2 matrix
    >>> df = pd.DataFrame([[1, 2], [3, 4]],
    ...                   index=['usa_c10t12', 'chn_c10t12'],
    ...                   columns=['usa_c10t12', 'chn_c10t12'])
    >>> melted = df_melt(df, 'FPEM')
    >>> print(melted.head())

    Notes
    -----
    The function preserves the index name 'selling' to maintain consistency
    with ICIO literature where rows represent selling sectors and columns
    represent buying/sourcing sectors.
    """
    # Preserve original index name or set default
    df_copy = df.copy()
    df_copy.index.name = 'selling'

    # Melt to long format, keeping index
    df_melted = pd.melt(df_copy, ignore_index=False)

    # Reset index to make 'selling' a column (use drop=False to keep the index as a column)
    df_melted = df_melted.reset_index()

    # Rename columns for clarity
    df_melted.columns = ['selling', 'sourcing', indicator_name]

    # Sort for consistent ordering
    df_melted.sort_values(['selling', 'sourcing'], inplace=True)

    return df_melted


def generate_sector_groups(
    df: pd.DataFrame,
    industry_col: str = 's'
) -> pd.DataFrame:
    """
    Add sector group classifications to DataFrame for aggregation analysis.

    This function maps fine-grained OECD ICIO industry codes to broader
    sector groupings (e.g., agriculture, mining, manufacturing, utilities, services).
    Follows standard OECD ICIO sector classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing industry codes.
    industry_col : str, optional
        Name of column containing industry codes, by default 's'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional 'sgroup' column containing sector group codes.

    Notes
    -----
    Sector groups follow OECD classification:
    - 01t03: Agriculture, forestry, fishing
    - 05t09: Mining and quarrying
    - 10t33: Manufacturing
    - 35t43: Utilities, water, construction
    - 45t98: Services

    Examples
    --------
    >>> df = pd.DataFrame({'s': ['c10t12', 'a01_02', 'k']})
    >>> df = generate_sector_groups(df)
    >>> print(df)
    """
    # Initialize empty sector group column
    df['sgroup'] = ""

    # Use config for sector groupings to support both old and new formats
    import config

    # Agriculture, forestry, fishing (01-03)
    df.loc[df[industry_col].isin(config.SECTOR_GROUPS["01t03"]), 'sgroup'] = "01t03"

    # Mining and quarrying (05-09)
    df.loc[df[industry_col].isin(config.SECTOR_GROUPS["05t09"]), 'sgroup'] = "05t09"

    # Manufacturing (10-33)
    df.loc[df[industry_col].isin(config.SECTOR_GROUPS["10t33"]), 'sgroup'] = "10t33"

    # Utilities, water, construction (35-43)
    df.loc[df[industry_col].isin(config.SECTOR_GROUPS["35t43"]), 'sgroup'] = "35t43"

    # Services (45-98)
    df.loc[df[industry_col].isin(config.SECTOR_GROUPS["45t98"]), 'sgroup'] = "45t98"

    return df


def validate_matrix_dimensions(
    *matrices: np.ndarray,
    expected_shape: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Validate that matrices have compatible dimensions for operations.

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of matrices to validate.
    expected_shape : tuple of int, optional
        Expected (rows, cols) shape, by default None (just check square).

    Returns
    -------
    bool
        True if all matrices have compatible dimensions, False otherwise.

    Examples
    --------
    >>> A = np.eye(3)
    >>> B = np.ones((3, 3))
    >>> validate_matrix_dimensions(A, B, expected_shape=(3, 3))
    True
    """
    if not matrices:
        logger.warning("No matrices provided for validation")
        return False

    # Check all matrices are 2D
    for i, mat in enumerate(matrices):
        if mat.ndim != 2:
            logger.error(f"Matrix {i} has {mat.ndim} dimensions (expected 2)")
            return False

    # Check all matrices have same shape
    ref_shape = matrices[0].shape
    for i, mat in enumerate(matrices[1:], start=1):
        if mat.shape != ref_shape:
            logger.error(
                f"Matrix {i} has shape {mat.shape}, expected {ref_shape}"
            )
            return False

    # Check square if required
    if expected_shape is None:
        # Just check square
        if ref_shape[0] != ref_shape[1]:
            logger.error(f"Matrices are not square: {ref_shape}")
            return False
    else:
        # Check exact shape match
        if ref_shape != expected_shape:
            logger.error(
                f"Matrices have shape {ref_shape}, expected {expected_shape}"
            )
            return False

    return True


def check_matrix_singularity(
    matrix: np.ndarray,
    tolerance: float = 1e-10,
    max_condition_number: float = 1e15
) -> Tuple[bool, Optional[float]]:
    """
    Check if a matrix is singular or ill-conditioned.

    A matrix is considered singular (non-invertible) if its determinant is
    approximately zero or if its condition number is too large. This check
    is critical before attempting matrix inversion in Leontief calculations.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check.
    tolerance : float, optional
        Tolerance for considering determinant as zero, by default 1e-10.
    max_condition_number : float, optional
        Maximum acceptable condition number, by default 1e15.

    Returns
    -------
    is_singular : bool
        True if matrix is singular or ill-conditioned, False otherwise.
    condition_number : float or None
        Condition number of the matrix, or None if matrix is exactly singular.

    Notes
    -----
    The condition number κ(A) measures how sensitive the matrix inverse is
    to perturbations in the input. A large condition number indicates an
    ill-conditioned matrix where small errors in data can lead to large
    errors in the inverse.

    For Leontief inverse L = (I-A)^(-1), singularity indicates that the
    input-output system is not productive (some sector requires more inputs
    than it produces).

    Examples
    --------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> is_sing, cond = check_matrix_singularity(A)
    >>> print(f"Singular: {is_sing}, Condition: {cond}")
    """
    if matrix.shape[0] != matrix.shape[1]:
        logger.error("Matrix is not square")
        return True, None

    # Check determinant
    try:
        det = np.linalg.det(matrix)
        if abs(det) < tolerance:
            logger.warning(f"Matrix is singular (det ~= {det:.2e})")
            return True, None
    except np.linalg.LinAlgError:
        logger.error("Failed to compute determinant")
        return True, None

    # Check condition number
    try:
        cond = np.linalg.cond(matrix)
        if cond > max_condition_number:
            logger.warning(
                f"Matrix is ill-conditioned (cond = {cond:.2e} > {max_condition_number:.2e})"
            )
            return True, cond
    except np.linalg.LinAlgError:
        logger.error("Failed to compute condition number")
        return True, None

    return False, cond


def safe_matrix_inverse(
    matrix: np.ndarray,
    use_pseudo_inverse: bool = True,
    tolerance: float = 1e-10,
    max_condition_number: float = 1e15
) -> Tuple[np.ndarray, bool]:
    """
    Compute matrix inverse with robust error handling.

    This function attempts standard matrix inversion and falls back to
    pseudo-inverse (Moore-Penrose inverse) if the matrix is singular or
    ill-conditioned. This is essential for Leontief inverse calculation
    where numerical issues can arise from data quality problems.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to invert.
    use_pseudo_inverse : bool, optional
        Whether to use pseudo-inverse as fallback, by default True.
    tolerance : float, optional
        Tolerance for singularity check, by default 1e-10.
    max_condition_number : float, optional
        Maximum acceptable condition number, by default 1e15.

    Returns
    -------
    inverse : np.ndarray
        Inverse (or pseudo-inverse) of the input matrix.
    used_pseudo : bool
        True if pseudo-inverse was used, False if standard inverse succeeded.

    Raises
    ------
    np.linalg.LinAlgError
        If matrix is singular and pseudo-inverse is not allowed.

    Notes
    -----
    The pseudo-inverse A⁺ satisfies:
    - AA⁺A = A
    - A⁺AA⁺ = A⁺
    - (AA⁺)ᵀ = AA⁺ (AA⁺ is symmetric)
    - (A⁺A)ᵀ = A⁺A (A⁺A is symmetric)

    For invertible matrices, A⁺ = A⁻¹. For singular matrices, A⁺ provides
    a "best fit" solution in the least-squares sense.

    References
    ----------
    Baldwin, Freeman, & Theodorakopoulos (2022, 2023): Emphasize importance
    of robust matrix operations in ICIO analysis.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_inv, used_pseudo = safe_matrix_inverse(A)
    >>> print(f"Used pseudo-inverse: {used_pseudo}")
    """
    # Validate input
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for inversion")

    # Check for singularity
    is_singular, cond_num = check_matrix_singularity(
        matrix, tolerance, max_condition_number
    )

    # Try standard inversion first
    if not is_singular:
        try:
            inverse = np.linalg.inv(matrix.astype(float))
            logger.debug(f"Standard inversion successful (κ = {cond_num:.2e})")
            return inverse, False
        except np.linalg.LinAlgError as e:
            logger.warning(f"Standard inversion failed: {e}")
            if not use_pseudo_inverse:
                raise

    # Fall back to pseudo-inverse
    if use_pseudo_inverse:
        logger.info("Using pseudo-inverse (Moore-Penrose inverse)")
        inverse = np.linalg.pinv(matrix.astype(float))
        return inverse, True
    else:
        raise np.linalg.LinAlgError(
            "Matrix is singular and pseudo-inverse is not allowed"
        )


def safe_division(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Perform element-wise division with safe handling of division by zero.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator array.
    denominator : np.ndarray
        Denominator array.
    fill_value : float, optional
        Value to use where denominator is zero, by default 0.0.

    Returns
    -------
    np.ndarray
        Result of numerator / denominator with fill_value for zero denominators.

    Notes
    -----
    This is essential for computing technical coefficients A = T / X where
    some sectors may have zero gross output.

    Examples
    --------
    >>> num = np.array([1, 2, 3])
    >>> den = np.array([2, 0, 3])
    >>> result = safe_division(num, den, fill_value=0.0)
    >>> print(result)  # [0.5, 0.0, 1.0]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.where(
            np.abs(denominator) > 1e-10,
            numerator / denominator,
            fill_value
        )
    return result
