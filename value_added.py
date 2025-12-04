"""
Value-Added Decomposition Module.

This module implements value-added decomposition techniques for tracing the
origin of value creation in global supply chains. The value-added approach
answers the question: "Where was the work done?" rather than "Where did
goods cross borders?"

Mathematical Framework:
    VA = V̂ L

where:
- VA: Value-added matrix (value-added content of gross output)
- V̂: Diagonal matrix of value-added-to-gross-output ratios (v_i = VA_i / X_i)
- L: Leontief inverse matrix

Element VA_ij represents the value added from sector i embodied in one dollar
of final output from sector j.

References
----------
Baldwin, Freeman, & Theodorakopoulos (2022): Section 3.2
    "Where was the work done? The value-added approach"
Johnson & Noguera (2012): "Accounting for intermediates:
    Production sharing and trade in value added"
Koopman, Wang, & Wei (2014): "Tracing value-added and double
    counting in gross exports"
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from utils.logging_config import setup_logger
from utils.helpers import safe_division

logger = setup_logger(__name__)


class ValueAddedDecomposition:
    """
    Analyzer for value-added decomposition in input-output systems.

    This class computes value-added matrices and decomposes trade flows
    into domestic and foreign value-added components.

    Attributes
    ----------
    L : np.ndarray
        Leontief inverse matrix.
    X : np.ndarray
        Gross output vector.
    VA : np.ndarray
        Value added vector (by sector).
    csnames : pd.Index
        Country-sector identifiers.

    Examples
    --------
    >>> from analysis.leontief import LeontiefAnalyzer
    >>> leontief = LeontiefAnalyzer(T, X, F)
    >>> L = leontief.compute_leontief_inverse()
    >>> va_decomp = ValueAddedDecomposition(L, X, VA, csnames)
    >>> va_matrix = va_decomp.compute_value_added_matrix()
    """

    def __init__(
        self,
        L: np.ndarray,
        X: np.ndarray,
        VA: np.ndarray,
        csnames: Optional[pd.Index] = None
    ):
        """
        Initialize value-added decomposition analyzer.

        Parameters
        ----------
        L : np.ndarray
            Leontief inverse matrix (total requirements matrix).
        X : np.ndarray
            Gross output vector.
        VA : np.ndarray
            Value added vector (payments to primary factors: labor, capital, etc.).
        csnames : pd.Index, optional
            Country-sector identifiers, by default None.

        Notes
        -----
        Value added represents payments to primary factors of production:
        - Compensation of employees (wages and salaries)
        - Operating surplus (profits, capital income)
        - Mixed income (for unincorporated enterprises)
        - Taxes less subsidies on production

        In ICIO tables, value added is often provided as a row in the table.
        Alternatively, it can be computed as: VA = X - T.sum(axis=0)
        (gross output minus intermediate purchases).
        """
        self.L = L
        self.X = X
        self.VA = VA
        self.csnames = csnames

        # Compute value-added-to-gross-output ratios
        self.va_ratios = safe_division(VA, X, fill_value=0.0)

        logger.info("ValueAddedDecomposition initialized")
        logger.debug(f"  VA/X ratio range: [{self.va_ratios.min():.3f}, {self.va_ratios.max():.3f}]")

    def compute_value_added_matrix(self) -> np.ndarray:
        """
        Compute value-added matrix VA = V̂ L.

        The value-added matrix shows the value added from each source sector
        (rows) embodied in the final output of each destination sector (columns).

        Returns
        -------
        np.ndarray
            Value-added matrix VA.

        Notes
        -----
        Mathematical formulation:
            VA = V̂ L

        where V̂ is the diagonal matrix of value-added ratios:
            v_i = VA_i / X_i

        Interpretation of elements:
            VA_ij = v_i * ℓ_ij

        This represents the value added in sector i embodied in one dollar
        of final demand for sector j's output.

        Column sums equal 1 (approximately):
            Σ_i VA_ij ≈ 1

        This reflects that all value in final output must come from value
        added somewhere in the supply chain (no value is created from nothing).

        References
        ----------
        Baldwin et al. (2022), equation after "VA is a matrix that is critical..."
        """
        # Create diagonal matrix of VA/X ratios
        V_hat = np.diag(self.va_ratios)

        # Compute VA = V̂ L
        VA_matrix = V_hat @ self.L

        logger.debug("Computed value-added matrix")
        logger.debug(f"  Column sums range: [{VA_matrix.sum(axis=0).min():.3f}, "
                    f"{VA_matrix.sum(axis=0).max():.3f}] (should be ≈ 1.0)")

        return VA_matrix

    def decompose_domestic_foreign_va(
        self,
        VA_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose value added into domestic and foreign components.

        For each sector, separate the value added that originates domestically
        versus from foreign countries.

        Parameters
        ----------
        VA_matrix : np.ndarray, optional
            Precomputed value-added matrix. If None, will be computed.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'domestic_va': Domestic value-added matrix
            - 'foreign_va': Foreign value-added matrix
            - 'dva_share': Domestic VA as share of total
            - 'fva_share': Foreign VA as share of total

        Notes
        -----
        For a country-sector pair 'country1_sector1', domestic VA includes
        all value added from any sector within 'country1', while foreign VA
        includes value added from all other countries.

        This decomposition is essential for understanding:
        - Import content of exports
        - Domestic vs foreign sourcing
        - GVC participation indices

        Examples
        --------
        >>> va_decomp = ValueAddedDecomposition(L, X, VA)
        >>> decomposition = va_decomp.decompose_domestic_foreign_va()
        >>> dva_share = decomposition['dva_share']
        >>> print(f"USA vehicles domestic VA share: {dva_share['usa_c29']:.1%}")
        """
        if VA_matrix is None:
            VA_matrix = self.compute_value_added_matrix()

        n = VA_matrix.shape[0]

        # Extract country codes from csnames
        # Assumes format: 'country_sector' (e.g., 'usa_c10t12')
        if self.csnames is None:
            logger.warning("csnames not provided, cannot decompose by country")
            return {}

        countries = [cs.split('_')[0] for cs in self.csnames]

        # Initialize domestic and foreign VA matrices
        domestic_va = np.zeros_like(VA_matrix)
        foreign_va = np.zeros_like(VA_matrix)

        # For each column (buying sector)
        for j in range(n):
            country_j = countries[j]

            # For each row (selling sector)
            for i in range(n):
                country_i = countries[i]

                if country_i == country_j:
                    # Same country → domestic VA
                    domestic_va[i, j] = VA_matrix[i, j]
                else:
                    # Different country → foreign VA
                    foreign_va[i, j] = VA_matrix[i, j]

        # Compute shares
        total_va = domestic_va + foreign_va
        dva_share = safe_division(domestic_va.sum(axis=0), total_va.sum(axis=0), fill_value=0.0)
        fva_share = safe_division(foreign_va.sum(axis=0), total_va.sum(axis=0), fill_value=0.0)

        logger.info("Decomposed value added into domestic and foreign components")
        logger.info(f"  Average domestic VA share: {dva_share.mean():.1%}")
        logger.info(f"  Average foreign VA share: {fva_share.mean():.1%}")

        return {
            'domestic_va': domestic_va,
            'foreign_va': foreign_va,
            'dva_share': dva_share,
            'fva_share': fva_share,
        }

    def compute_vax_ratio(
        self,
        exports: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute VAX ratio (value-added to gross exports ratio).

        The VAX ratio measures the domestic value-added content of exports,
        answering: "How much of our export value is actually created here?"

        Parameters
        ----------
        exports : np.ndarray, optional
            Vector of gross exports by sector. If None, estimates from final demand.

        Returns
        -------
        np.ndarray
            VAX ratio for each sector.

        Notes
        -----
        The VAX ratio is defined as:
            VAX_j = (Domestic VA in exports of j) / (Gross exports of j)

        A VAX ratio of 0.7 means 70% of the export value was created domestically,
        while 30% represents foreign value added (imported inputs).

        Low VAX ratios indicate high import content, common in:
        - Assembly operations (e.g., electronics assembly)
        - Re-export hubs (e.g., Hong Kong, Singapore)
        - Sectors with complex global value chains

        References
        ----------
        Johnson & Noguera (2012): Original VAX concept
        Koopman, Wang, & Wei (2014): Refined measures
        """
        VA_matrix = self.compute_value_added_matrix()

        # Decompose into domestic and foreign
        decomp = self.decompose_domestic_foreign_va(VA_matrix)
        domestic_va_total = decomp['domestic_va'].sum(axis=0)

        # Use exports if provided, otherwise use proxy from data
        if exports is None:
            # Rough proxy: assume exports proportional to gross output
            logger.warning("Exports not provided, using gross output as proxy")
            exports = self.X

        # Compute VAX ratio
        vax_ratio = safe_division(domestic_va_total, exports, fill_value=0.0)

        logger.debug(f"VAX ratio range: [{vax_ratio.min():.2f}, {vax_ratio.max():.2f}]")

        return vax_ratio

    def to_dataframe(
        self,
        matrix: np.ndarray,
        name: str = 'VA'
    ) -> pd.DataFrame:
        """
        Convert value-added matrix to labeled DataFrame.

        Parameters
        ----------
        matrix : np.ndarray
            Value-added matrix to convert.
        name : str, optional
            Name for logging, by default 'VA'.

        Returns
        -------
        pd.DataFrame
            DataFrame with country-sector labels.
        """
        if self.csnames is None:
            logger.warning("No csnames provided, using numeric indices")
            return pd.DataFrame(matrix)

        df = pd.DataFrame(
            matrix,
            index=self.csnames,
            columns=self.csnames
        )
        logger.debug(f"Converted {name} to DataFrame")
        return df
