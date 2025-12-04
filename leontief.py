"""
Leontief Input-Output Analysis Module.

This module implements the core Leontief inverse calculation and related
input-output analysis techniques. The Leontief inverse matrix L = (I-A)^(-1)
is fundamental to understanding supply chain linkages and exposure.

The mathematical framework follows:
    X = AX + F  (use accounting identity)
    X = (I-A)^(-1) F = LF  (solving for gross output)

where:
- X: Gross output vector (total production by each sector)
- A: Technical coefficients matrix (input requirements per unit output)
- F: Final demand vector (consumption, investment, exports)
- L: Leontief inverse matrix (total requirements matrix)

References
----------
Baldwin, Freeman, & Theodorakopoulos (2022): "Horses for Courses"
    Comprehensive exposition of IO analysis for supply chain measurement.
Baldwin, Freeman, & Theodorakopoulos (2023): "Hidden Exposure"
    Application to US-China supply chain analysis.
Leontief, W. (1986): Input-Output Economics, 2nd edition.
Miller & Blair (2009): Input-Output Analysis: Foundations and Extensions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings

from utils.logging_config import setup_logger
from utils.helpers import (
    safe_matrix_inverse,
    safe_division,
    validate_matrix_dimensions,
)

logger = setup_logger(__name__)


class LeontiefAnalyzer:
    """
    Analyzer for Leontief input-output calculations.

    This class encapsulates the computation of the Leontief inverse matrix
    and related supply chain metrics. It provides robust numerical methods
    and comprehensive validation.

    The Leontief inverse L = (I-A)^(-1) captures the total (direct + indirect)
    requirements of all sectors needed to satisfy one unit of final demand.
    Each element ℓ_ij represents the gross output of sector i required to
    produce one dollar of final output in sector j.

    Attributes
    ----------
    T : np.ndarray or pd.DataFrame
        Intermediate use matrix (sector-to-sector flows).
    X : np.ndarray or pd.Series
        Gross output vector.
    F : np.ndarray or pd.Series
        Final demand vector.
    A : np.ndarray
        Technical coefficients matrix (computed).
    L : np.ndarray
        Leontief inverse matrix (computed).
    csnames : pd.Index, optional
        Country-sector identifiers for labeling.

    Examples
    --------
    >>> analyzer = LeontiefAnalyzer(T, X, F, csnames)
    >>> L = analyzer.compute_leontief_inverse()
    >>> LF = analyzer.compute_total_requirements()
    """

    def __init__(
        self,
        T: pd.DataFrame,
        X: pd.Series,
        F: pd.Series,
        csnames: Optional[pd.Index] = None,
        validate: bool = True
    ):
        """
        Initialize Leontief analyzer.

        Parameters
        ----------
        T : pd.DataFrame
            Intermediate use matrix. T[i,j] represents sales from sector i
            to sector j for use as intermediate inputs.
        X : pd.Series
            Gross output vector. X[i] is total production of sector i.
        F : pd.Series
            Final demand vector. F[i] is final demand for sector i output.
        csnames : pd.Index, optional
            Country-sector identifiers for rows/columns, by default None.
        validate : bool, optional
            Whether to validate input data, by default True.

        Notes
        -----
        The intermediate use matrix T is also called the "transactions matrix"
        or "flows matrix". It captures business-to-business (B2B) transactions.

        Final demand F includes household consumption, government spending,
        investment, and net exports. These are business-to-consumer (B2C)
        transactions.

        Gross output X represents total production (sales) by each sector,
        equal to intermediate sales plus final sales: X = T.sum(axis=1) + F
        """
        self.T = T
        self.X = X
        self.F = F
        self.csnames = csnames if csnames is not None else T.index

        # Convert to numpy for efficient computation
        self.T_array = T.to_numpy() if isinstance(T, pd.DataFrame) else T
        self.X_array = X.to_numpy() if isinstance(X, pd.Series) else X
        self.F_array = F.to_numpy() if isinstance(F, pd.Series) else F

        # Validate inputs
        if validate:
            self._validate_inputs()

        # Initialize computed matrices (will be calculated on demand)
        self.A = None
        self.L = None
        self.L_used_pseudo = False

        logger.info(f"LeontiefAnalyzer initialized with {len(self.X_array)} sectors")

    def _validate_inputs(self) -> None:
        """
        Validate input data dimensions and accounting identities.

        Checks:
        1. T is square matrix
        2. X and F have same length as T dimensions
        3. Use accounting identity: X ≈ T.sum(axis=1) + F
        4. No negative values in X (gross output must be non-negative)

        Raises
        ------
        ValueError
            If validation fails.
        """
        n = self.T_array.shape[0]

        # Check dimensions
        if self.T_array.shape[0] != self.T_array.shape[1]:
            raise ValueError(f"T must be square, got shape {self.T_array.shape}")

        if len(self.X_array) != n:
            raise ValueError(f"X length {len(self.X_array)} != T dimensions {n}")

        if len(self.F_array) != n:
            raise ValueError(f"F length {len(self.F_array)} != T dimensions {n}")

        # Check use accounting identity: X = T.sum(axis=1) + F
        T_row_sum = self.T_array.sum(axis=1)
        X_computed = T_row_sum + self.F_array
        relative_error = np.abs(X_computed - self.X_array) / (self.X_array + 1e-10)

        if (relative_error > 0.01).any():  # 1% tolerance
            n_errors = (relative_error > 0.01).sum()
            max_error = relative_error.max()
            logger.warning(
                f"Use accounting identity violated for {n_errors} sectors "
                f"(max relative error: {max_error:.2%})"
            )

        # Check for negative values
        if (self.X_array < 0).any():
            n_neg = (self.X_array < 0).sum()
            logger.warning(f"Found {n_neg} negative values in gross output X")

        logger.debug("Input validation completed")

    def compute_technical_coefficients(
        self,
        method: str = 'column'
    ) -> np.ndarray:
        """
        Compute technical coefficients matrix A.

        The technical coefficients matrix A represents the input requirements
        per unit of output. Element a_ij shows how much input from sector i
        is needed to produce one dollar of output in sector j.

        There are two conventions:
        - Column normalization (standard): A = T / X (broadcast over columns)
        - Row normalization (Ghosh): B = T / X (broadcast over rows)

        This function implements column normalization, which is standard in
        Leontief analysis.

        Parameters
        ----------
        method : str, optional
            Normalization method: 'column' (Leontief) or 'row' (Ghosh),
            by default 'column'.

        Returns
        -------
        np.ndarray
            Technical coefficients matrix A.

        Notes
        -----
        Mathematical definition:
            a_ij = T_ij / X_j  (column normalization)

        Interpretation: If X_j increases by $1, sector j will purchase $a_ij
        more inputs from sector i.

        The matrix satisfies: T = A ⊙ X' where ⊙ is element-wise multiplication
        and X' is broadcast across columns.

        Division by zero is handled by replacing with zero (sectors with no
        output don't purchase inputs).

        References
        ----------
        Baldwin et al. (2022), Section 2.2.3: Detailed exposition of technical
        coefficients and their economic interpretation.
        """
        if method == 'column':
            # Standard Leontief: A = T / X (column normalization)
            # a_ij = T_ij / X_j
            # Broadcast X across columns: each column j divided by X_j
            A = safe_division(self.T_array, self.X_array[np.newaxis, :], fill_value=0.0)

        elif method == 'row':
            # Ghosh approach: B = T / X (row normalization)
            # b_ij = T_ij / X_i
            # Broadcast X across rows: each row i divided by X_i
            A = safe_division(self.T_array, self.X_array[:, np.newaxis], fill_value=0.0)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'column' or 'row'.")

        # Replace any NaN or inf values with zero
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        self.A = A
        logger.debug(f"Computed technical coefficients matrix (method={method})")
        logger.debug(f"  A sparsity: {(A == 0).sum() / A.size:.1%}")
        logger.debug(f"  A range: [{A.min():.4f}, {A.max():.4f}]")

        return A

    def compute_leontief_inverse(
        self,
        use_pseudo_inverse: bool = True,
        tolerance: float = 1e-10,
        max_condition_number: float = 1e15
    ) -> np.ndarray:
        """
        Compute the Leontief inverse matrix L = (I-A)^(-1).

        The Leontief inverse is the most important matrix in input-output analysis.
        It captures the total (direct + indirect) linkages between all sectors.

        Mathematical background:
        Starting from X = AX + F, we solve for X:
            X - AX = F
            (I - A)X = F
            X = (I - A)^(-1) F = LF

        The inverse can be expressed as an infinite series:
            L = (I-A)^(-1) = I + A + A² + A³ + ...

        This series converges if and only if the spectral radius of A is less
        than 1, which is equivalent to the economy being "productive"
        (not requiring more inputs than it produces).

        Parameters
        ----------
        use_pseudo_inverse : bool, optional
            Whether to use pseudo-inverse if (I-A) is singular, by default True.
        tolerance : float, optional
            Tolerance for singularity detection, by default 1e-10.
        max_condition_number : float, optional
            Maximum acceptable condition number, by default 1e15.

        Returns
        -------
        np.ndarray
            Leontief inverse matrix L.

        Raises
        ------
        np.linalg.LinAlgError
            If matrix is singular and pseudo_inverse is False.

        Notes
        -----
        Interpretation of elements:
        - Diagonal elements ℓ_ii > 1: Sector i needs more than $1 of its own
          output (direct + indirect) to produce $1 of final output
        - Off-diagonal ℓ_ij: Dollar amount of sector i output needed (total)
          to produce $1 of final output in sector j

        The column sums of L show the "output multiplier": total output across
        all sectors required to satisfy $1 of final demand in a given sector.

        Computational considerations:
        - For large ICIO tables (3000+ sectors), inversion can be slow
        - Numerical stability is critical: check condition number
        - Pseudo-inverse provides fallback for singular/ill-conditioned cases

        References
        ----------
        Baldwin et al. (2022), Section 2.2.4: "The Leontief inverse matrix"
        Miller & Blair (2009), Chapter 2: Mathematical foundations
        """
        # Compute A if not already done
        if self.A is None:
            self.compute_technical_coefficients()

        # Compute (I - A)
        n = self.A.shape[0]
        I = np.eye(n)
        I_minus_A = I - self.A

        # Invert (I-A) to get Leontief inverse L
        L, used_pseudo = safe_matrix_inverse(
            I_minus_A,
            use_pseudo_inverse=use_pseudo_inverse,
            tolerance=tolerance,
            max_condition_number=max_condition_number
        )

        self.L = L
        self.L_used_pseudo = used_pseudo

        # Log diagnostics
        logger.info("Computed Leontief inverse matrix")
        logger.info(f"  Used pseudo-inverse: {used_pseudo}")
        logger.info(f"  L diagonal range: [{np.diag(L).min():.3f}, {np.diag(L).max():.3f}]")
        logger.info(f"  L off-diagonal range: [{L[~np.eye(n, dtype=bool)].min():.3f}, "
                   f"{L[~np.eye(n, dtype=bool)].max():.3f}]")

        # Check if all diagonal elements > 1 (sanity check)
        diag_L = np.diag(L)
        if not np.all(diag_L >= 1.0 - tolerance):
            n_violations = np.sum(diag_L < 1.0 - tolerance)
            logger.warning(
                f"Leontief inverse has {n_violations} diagonal elements < 1.0 "
                "(possible data quality issue)"
            )

        return L

    def compute_total_requirements(self) -> np.ndarray:
        """
        Compute total requirements matrix LF.

        The matrix LF shows the total gross output from each sector (rows)
        required to satisfy final demand in each sector (columns), accounting
        for all direct and indirect linkages through the supply chain.

        Returns
        -------
        np.ndarray
            Total requirements matrix LF, where (LF)_ij is the gross output
            of sector i required to produce the final demand in sector j.

        Notes
        -----
        Mathematical formulation:
            LF = (I-A)^(-1) F

        This is computed element-wise: (LF)_ij = ℓ_ij * F_j

        The row sums of LF equal gross output X (by definition):
            X = LF · 1 = L(F · 1) = LF_total

        where F · 1 is the row-sum of final demand across all destinations.

        For bilateral exposure analysis, we often look at columns of LF:
        Column j shows all the sectoral outputs needed to satisfy final
        demand in sector j.

        Examples
        --------
        >>> analyzer = LeontiefAnalyzer(T, X, F)
        >>> LF = analyzer.compute_total_requirements()
        >>> # Total US vehicle output needed for US final demand:
        >>> us_vehicles_for_us = LF.loc['usa_c29', 'usa_c29']
        """
        if self.L is None:
            self.compute_leontief_inverse()

        # Compute LF: element-wise multiplication of L by F (broadcast across columns)
        LF = self.L * self.F_array[np.newaxis, :]

        logger.debug("Computed total requirements matrix LF")

        return LF

    def compute_face_value_exposure(self) -> np.ndarray:
        """
        Compute face value exposure matrix (I+A)F.

        Face value exposure captures only direct (tier-1) bilateral linkages,
        without accounting for indirect supply chain connections. This is the
        "level-2 answer" in Baldwin et al. (2022) terminology.

        Returns
        -------
        np.ndarray
            Face value matrix (I+A)F.

        Notes
        -----
        Mathematical formulation:
            FV = (I + A)F

        Interpretation:
        - I * F: Direct final demand (sales to consumers)
        - A * F: Direct intermediate purchases (tier-1 suppliers only)

        This contrasts with the Leontief-based "look-through" exposure LF which
        includes all tiers of suppliers (suppliers of suppliers, etc.).

        Hidden exposure is defined as:
            HE = LF - (I+A)F

        This represents the supply chain exposure "hidden" behind direct
        bilateral trade relationships.

        References
        ----------
        Baldwin et al. (2023), Section I.A: "Face Value versus Look Through Exposure"
        """
        if self.A is None:
            self.compute_technical_coefficients()

        n = self.A.shape[0]
        I = np.eye(n)
        I_plus_A = I + self.A

        # Compute (I+A)F: element-wise multiplication
        face_value = I_plus_A * self.F_array[np.newaxis, :]

        logger.debug("Computed face value exposure matrix (I+A)F")

        return face_value

    def compute_output_multipliers(self) -> np.ndarray:
        """
        Compute output multipliers from Leontief inverse.

        The output multiplier for sector j measures the total gross output
        across all sectors required to satisfy $1 of final demand in sector j.

        Returns
        -------
        np.ndarray
            Vector of output multipliers (one per sector).

        Notes
        -----
        The output multiplier is the column sum of the Leontief inverse:
            m_j = Σ_i ℓ_ij

        Interpretation: If final demand for sector j increases by $1, total
        economy-wide output will increase by $m_j (including direct and
        indirect effects).

        Sectors with high multipliers have extensive backward linkages
        (they rely heavily on inputs from other sectors).

        Examples
        --------
        >>> analyzer = LeontiefAnalyzer(T, X, F)
        >>> multipliers = analyzer.compute_output_multipliers()
        >>> print(f"Vehicles multiplier: {multipliers['usa_c29']:.2f}")
        """
        if self.L is None:
            self.compute_leontief_inverse()

        # Column sums of L
        multipliers = self.L.sum(axis=0)

        logger.debug(f"Output multipliers range: [{multipliers.min():.2f}, {multipliers.max():.2f}]")

        return multipliers

    def to_dataframe(
        self,
        matrix: np.ndarray,
        name: str = 'value'
    ) -> pd.DataFrame:
        """
        Convert numpy matrix to labeled DataFrame.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix to convert.
        name : str, optional
            Name for the matrix (used in logging), by default 'value'.

        Returns
        -------
        pd.DataFrame
            DataFrame with country-sector labels for rows and columns.
        """
        df = pd.DataFrame(
            matrix,
            index=self.csnames,
            columns=self.csnames
        )
        logger.debug(f"Converted {name} matrix to DataFrame")
        return df
