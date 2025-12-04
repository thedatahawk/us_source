"""
Supply Chain Exposure Calculator Module.

This module implements the Foreign Production Exposure Measure (FPEM) suite
of indicators developed by Baldwin, Freeman, & Theodorakopoulos (2022, 2023).

The FPEM indicators follow the "horses for courses" philosophy: different
indicators are appropriate for different types of supply chain shocks.

Key indicators:
- FPEM: Full "look-through" exposure (using Leontief inverse LF)
- FPEMfv: "Face value" direct exposure (using (I+A)F)
- FPEMhe: "Hidden exposure" (FPEM - FPEMfv)

These indicators answer: "What share of my inputs come from foreign country X?"
accounting for different levels of supply chain depth.

References
----------
Baldwin, Freeman, & Theodorakopoulos (2022): "Horses for Courses"
    Section 6: "Measuring exposure to foreign suppliers"
Baldwin, Freeman, & Theodorakopoulos (2023): "Hidden Exposure"
    Core empirical application to US-China supply chains
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from functools import reduce

from utils.logging_config import setup_logger
from utils.helpers import df_melt, safe_division

logger = setup_logger(__name__)


class ExposureCalculator:
    """
    Calculator for supply chain exposure indicators.

    This class implements the FPEM indicator suite for measuring foreign
    supply chain exposure at different levels of granularity.

    Attributes
    ----------
    L : np.ndarray
        Leontief inverse matrix.
    F : np.ndarray
        Final demand vector.
    A : np.ndarray
        Technical coefficients matrix.
    csnames : pd.Index
        Country-sector identifiers.

    Examples
    --------
    >>> calc = ExposureCalculator(L, F, A, csnames)
    >>> fpem = calc.compute_fpem()
    >>> fpemfv = calc.compute_fpem_face_value()
    >>> fpemhe = calc.compute_hidden_exposure(fpem, fpemfv)
    """

    def __init__(
        self,
        L: np.ndarray,
        F: np.ndarray,
        A: np.ndarray,
        csnames: pd.Index
    ):
        """
        Initialize exposure calculator.

        Parameters
        ----------
        L : np.ndarray
            Leontief inverse matrix (from LeontiefAnalyzer).
        F : np.ndarray
            Final demand vector.
        A : np.ndarray
            Technical coefficients matrix.
        csnames : pd.Index
            Country-sector identifiers (e.g., ['usa_c10t12', 'chn_c20', ...]).

        Notes
        -----
        The exposure indicators are computed relative to total intermediate
        sourcing (TI), defined as the sum of all intermediate inputs needed
        to satisfy final demand, accounting for full supply chains.
        """
        self.L = L
        self.F = F
        self.A = A
        self.csnames = csnames

        logger.info("ExposureCalculator initialized")

    def compute_total_intermediate_sourcing(self, LF: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute total intermediate sourcing (TI).

        TI represents the total value of intermediate inputs (from all sources,
        domestic and foreign) required to satisfy final demand, accounting for
        full supply chain linkages.

        Parameters
        ----------
        LF : np.ndarray, optional
            Precomputed LF matrix. If None, will compute from L and F.

        Returns
        -------
        np.ndarray
            Total intermediate sourcing vector (one value per buying sector).

        Notes
        -----
        Mathematical definition:
            TI_j = Σ_i (LF)_ij

        This is the denominator used to normalize FPEM indicators, representing
        the total "exposure" that will be decomposed into bilateral components.

        The column sums of LF approximately equal gross output X.
        """
        if LF is None:
            LF = self.L * self.F[np.newaxis, :]

        # Sum over selling sectors (rows)
        TI = LF.sum(axis=0)

        logger.debug(f"Total intermediate sourcing (TI) range: [{TI.min():.0f}, {TI.max():.0f}]")

        return TI

    def compute_fpem(
        self,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute FPEM (Foreign Production Exposure Measure) - look-through version.

        FPEM measures the share of intermediate inputs sourced from each
        foreign country-sector on a "look-through" basis, meaning it accounts
        for the full supply chain (suppliers of suppliers, etc.).

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize by total intermediate sourcing, by default True.

        Returns
        -------
        np.ndarray
            FPEM matrix. Element FPEM[i,j] represents the share of sector j's
            intermediate inputs sourced from sector i (look-through basis).

        Notes
        -----
        Mathematical formulation:
            FPEM = LF / TI

        where:
        - LF: Total requirements matrix (Leontief inverse × final demand)
        - TI: Total intermediate sourcing (column sums of LF)

        Interpretation:
        If FPEM['chn_c20', 'usa_c29'] = 0.05, this means 5% of all
        intermediate inputs used by the US vehicles sector (usa_c29) come
        from China's chemicals sector (chn_c20), accounting for full supply
        chains.

        This is the "horses for courses" indicator appropriate for:
        - Gross production shocks (e.g., factory shutdowns)
        - Comprehensive supply chain mapping
        - Strategic sourcing decisions

        References
        ----------
        Baldwin et al. (2022, 2023): Core FPEM indicator definition
        """
        # Compute LF
        LF = self.L * self.F[np.newaxis, :]

        if normalize:
            # Normalize by total intermediate sourcing
            TI = self.compute_total_intermediate_sourcing(LF)
            FPEM = safe_division(LF, TI[np.newaxis, :], fill_value=0.0)
        else:
            FPEM = LF

        logger.debug("Computed FPEM (look-through exposure)")
        if normalize:
            logger.debug(f"  Column sums (should be ≈1.0): [{FPEM.sum(axis=0).min():.3f}, "
                        f"{FPEM.sum(axis=0).max():.3f}]")

        return FPEM

    def compute_fpem_face_value(
        self,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute FPEMfv (Face Value exposure) - direct bilateral only.

        FPEMfv measures the share of intermediate inputs sourced directly
        from each country-sector, without accounting for indirect supply
        chain linkages. This is the "level-2 answer" or "tier-1 suppliers only."

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize by total intermediate sourcing, by default True.

        Returns
        -------
        np.ndarray
            FPEMfv matrix (face value exposure).

        Notes
        -----
        Mathematical formulation:
            FPEMfv = (I + A)F / TI

        where:
        - I: Identity matrix (captures direct final demand)
        - A: Technical coefficients (captures tier-1 suppliers)
        - F: Final demand vector

        The (I+A)F term represents:
        - IF: Direct sales to final demand
        - AF: Direct intermediate purchases (no recursion)

        This is appropriate for:
        - Direct trade disruptions (e.g., port closures, tariffs)
        - Border-crossing events
        - Immediate bilateral exposure

        Contrast with FPEM (look-through): FPEMfv ignores that Chinese parts
        might come via Mexico, or that Mexican parts contain Chinese components.

        References
        ----------
        Baldwin et al. (2023), Box I.B: "Face value versus look through measures"
        """
        n = self.A.shape[0]
        I = np.eye(n)

        # Compute (I + A)F
        I_plus_A_F = (I + self.A) * self.F[np.newaxis, :]

        if normalize:
            # Normalize by total intermediate sourcing (using LF for consistency)
            LF = self.L * self.F[np.newaxis, :]
            TI = self.compute_total_intermediate_sourcing(LF)
            FPEMfv = safe_division(I_plus_A_F, TI[np.newaxis, :], fill_value=0.0)
        else:
            FPEMfv = I_plus_A_F

        logger.debug("Computed FPEMfv (face value exposure)")

        return FPEMfv

    def compute_hidden_exposure(
        self,
        FPEM: Optional[np.ndarray] = None,
        FPEMfv: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute FPEMhe (Hidden Exposure) = FPEM - FPEMfv.

        Hidden exposure represents the additional supply chain exposure that
        exists beyond direct bilateral trade relationships. It captures the
        exposure to a foreign supplier that is "hidden" in the supply chains
        of direct suppliers.

        Parameters
        ----------
        FPEM : np.ndarray, optional
            Look-through exposure. If None, will be computed.
        FPEMfv : np.ndarray, optional
            Face value exposure. If None, will be computed.

        Returns
        -------
        np.ndarray
            FPEMhe matrix (hidden exposure).

        Notes
        -----
        Mathematical definition:
            FPEMhe = FPEM - FPEMfv = LF / TI - (I+A)F / TI

        Interpretation:
        - Hidden exposure is always non-negative (look-through ≥ face value)
        - Large hidden exposure indicates complex supply chains
        - Hidden exposure to China is often 4x the face value for US sectors

        Example:
        If USA imports car parts from Mexico, and Mexico uses Chinese steel:
        - FPEMfv captures direct imports from Mexico
        - FPEM captures both Mexico AND the Chinese content in Mexican parts
        - FPEMhe = exposure to China hidden in Mexican supplies

        This indicator is appropriate for:
        - Comprehensive sanctions analysis
        - Strategic resilience planning
        - Understanding true dependencies

        References
        ----------
        Baldwin et al. (2023), Section II.C: "Hidden Exposure: Look Through
        Versus Face Value Measures"
        """
        if FPEM is None:
            FPEM = self.compute_fpem()

        if FPEMfv is None:
            FPEMfv = self.compute_fpem_face_value()

        FPEMhe = FPEM - FPEMfv

        logger.debug("Computed FPEMhe (hidden exposure)")
        logger.debug(f"  Hidden exposure range: [{FPEMhe.min():.4f}, {FPEMhe.max():.4f}]")

        # Sanity check: hidden exposure should be non-negative
        if (FPEMhe < -1e-6).any():
            n_negative = (FPEMhe < -1e-6).sum()
            logger.warning(f"Found {n_negative} negative hidden exposure values (likely numerical error)")

        return FPEMhe

    def compute_all_indicators(self) -> Dict[str, np.ndarray]:
        """
        Compute all three FPEM indicators in one call.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'FPEM': Look-through exposure
            - 'FPEMfv': Face value exposure
            - 'FPEMhe': Hidden exposure
            - 'LF': Total requirements matrix
            - 'IplusAF': Face value matrix (unnormalized)

        Examples
        --------
        >>> calc = ExposureCalculator(L, F, A, csnames)
        >>> indicators = calc.compute_all_indicators()
        >>> fpem_df = pd.DataFrame(indicators['FPEM'], index=csnames, columns=csnames)
        """
        # Compute LF
        LF = self.L * self.F[np.newaxis, :]

        # Compute (I+A)F
        n = self.A.shape[0]
        I = np.eye(n)
        IplusAF = (I + self.A) * self.F[np.newaxis, :]

        # Compute TI for normalization
        TI = self.compute_total_intermediate_sourcing(LF)

        # Compute normalized indicators
        FPEM = safe_division(LF, TI[np.newaxis, :], fill_value=0.0)
        FPEMfv = safe_division(IplusAF, TI[np.newaxis, :], fill_value=0.0)
        FPEMhe = FPEM - FPEMfv

        logger.info("Computed all exposure indicators")

        return {
            'FPEM': FPEM,
            'FPEMfv': FPEMfv,
            'FPEMhe': FPEMhe,
            'LF': LF,
            'IplusAF': IplusAF,
        }

    def create_bilateral_exposure_dataframe(
        self,
        indicators: Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Create long-format DataFrame with bilateral exposure indicators.

        This converts the square exposure matrices into a long-format table
        with explicit 'selling' and 'sourcing' country-sector pairs, suitable
        for filtering, aggregation, and visualization.

        Parameters
        ----------
        indicators : Dict[str, np.ndarray], optional
            Dictionary of indicator matrices. If None, will compute all.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ['selling', 'sourcing', 'FPEM', 'FPEMfv', 'FPEMhe', 'LF', 'IplusAF']

        Notes
        -----
        This format is essential for country-level aggregation and filtering.
        For example, to get USA's exposure to all Chinese sectors:
            df[df['selling'].str.startswith('chn_') & df['sourcing'].str.startswith('usa_')]

        Examples
        --------
        >>> calc = ExposureCalculator(L, F, A, csnames)
        >>> df = calc.create_bilateral_exposure_dataframe()
        >>> usa_from_china = df[df['selling'].str.startswith('chn_') &
        ...                      df['sourcing'].str.startswith('usa_')]
        """
        if indicators is None:
            indicators = self.compute_all_indicators()

        # Convert each indicator to DataFrame and melt
        dfs = []
        for name, matrix in indicators.items():
            df_matrix = pd.DataFrame(matrix, index=self.csnames, columns=self.csnames)
            df_melted = df_melt(df_matrix, indicator_name=name)
            dfs.append(df_melted)

        # Merge all indicators
        df_combined = reduce(
            lambda left, right: pd.merge(left, right, on=['selling', 'sourcing'], how='outer'),
            dfs
        )

        # Extract country and sector from combined identifiers
        df_combined['selling_country'] = df_combined['selling'].str.split('_', n=1).str[0]
        df_combined['sourcing_country'] = df_combined['sourcing'].str.split('_', n=1).str[0]
        df_combined['selling_industry'] = df_combined['selling'].str.split('_', n=1).str[1]
        df_combined['sourcing_industry'] = df_combined['sourcing'].str.split('_', n=1).str[1]

        logger.info(f"Created bilateral exposure DataFrame with {len(df_combined)} rows")

        return df_combined

    def filter_by_country(
        self,
        df: pd.DataFrame,
        sourcing_country: Optional[str] = None,
        selling_country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter bilateral exposure DataFrame by country.

        Parameters
        ----------
        df : pd.DataFrame
            Bilateral exposure DataFrame (from create_bilateral_exposure_dataframe).
        sourcing_country : str, optional
            Filter sourcing (buying) country, by default None.
        selling_country : str, optional
            Filter selling (supplying) country, by default None.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.

        Examples
        --------
        >>> # Get USA's exposure to China
        >>> usa_china = calc.filter_by_country(df, sourcing_country='usa', selling_country='chn')
        """
        df_filtered = df.copy()

        if sourcing_country:
            df_filtered = df_filtered[df_filtered['sourcing_country'] == sourcing_country]

        if selling_country:
            df_filtered = df_filtered[df_filtered['selling_country'] == selling_country]

        logger.debug(f"Filtered to {len(df_filtered)} rows")

        return df_filtered

    def aggregate_by_country(
        self,
        df: pd.DataFrame,
        indicators: List[str] = ['FPEM', 'FPEMfv', 'FPEMhe']
    ) -> pd.DataFrame:
        """
        Aggregate bilateral exposure from sector-level to country-level.

        Parameters
        ----------
        df : pd.DataFrame
            Bilateral exposure DataFrame.
        indicators : List[str], optional
            Indicators to aggregate, by default ['FPEM', 'FPEMfv', 'FPEMhe'].

        Returns
        -------
        pd.DataFrame
            Country-level aggregated exposure.

        Examples
        --------
        >>> df_country = calc.aggregate_by_country(df)
        >>> usa_china_total = df_country[
        ...     (df_country['sourcing_country'] == 'usa') &
        ...     (df_country['selling_country'] == 'chn')
        ... ]
        """
        df_agg = df.groupby(['selling_country', 'sourcing_country'])[indicators].sum().reset_index()

        logger.debug(f"Aggregated to {len(df_agg)} country pairs")

        return df_agg
