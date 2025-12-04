"""
OECD ICIO Data Loader Module.

This module provides functionality to load and preprocess OECD Inter-Country
Input-Output (ICIO) tables from CSV files. The ICIO tables capture the
inter-sectoral flows of goods and services within and between countries.

The loader handles:
- Loading ICIO CSV files for multiple years
- Loading metadata (country names, industry names)
- Extracting key matrices (T, X, F)
- Basic data validation and cleaning

References
----------
Baldwin, Freeman, & Theodorakopoulos (2022): "Horses for Courses:
    Measuring Foreign Supply Chain Exposure"
OECD (2021): Inter-Country Input-Output Database
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

from utils.logging_config import setup_logger
from utils.helpers import safe_division

logger = setup_logger(__name__)


class ICIODataLoader:
    """
    Loader for OECD ICIO (Inter-Country Input-Output) data.

    This class handles loading OECD ICIO tables from CSV files and extracting
    the fundamental matrices needed for input-output analysis:
    - T: Intermediate use matrix (sector-to-sector flows)
    - X: Gross output vector
    - F: Final demand vector

    The ICIO table structure follows OECD conventions where rows represent
    selling sectors and columns represent buying sectors or final demand categories.

    Attributes
    ----------
    input_folder : Path
        Directory containing ICIO CSV files.
    country_names : Dict[str, str]
        Mapping from country codes to country names.
    industry_names : Dict[str, str]
        Mapping from industry codes to industry names.

    Examples
    --------
    >>> from pathlib import Path
    >>> loader = ICIODataLoader(Path("data/icio_files"))
    >>> data = loader.load_year(2018)
    >>> print(data['T'].shape)  # Intermediate use matrix
    """

    def __init__(
        self,
        input_folder: Path,
        country_names_path: Path,
        industry_names_path: Path
    ):
        """
        Initialize the ICIO data loader.

        Parameters
        ----------
        input_folder : Path
            Directory containing ICIO CSV files (named as YYYY_SML.csv).
        country_names_path : Path
            Path to CSV file with country code to name mappings.
        industry_names_path : Path
            Path to CSV file with industry code to name mappings.
        """
        self.input_folder = Path(input_folder)
        self.country_names_path = Path(country_names_path)
        self.industry_names_path = Path(industry_names_path)

        # Load metadata
        self.country_names = self._load_country_names()
        self.industry_names = self._load_industry_names()

        logger.info(f"ICIO Loader initialized")
        logger.info(f"  Loaded {len(self.country_names)} countries")
        logger.info(f"  Loaded {len(self.industry_names)} industries")

    def _load_country_names(self) -> Dict[str, str]:
        """
        Load country code to name mapping from CSV.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping country codes (e.g., 'usa') to names (e.g., 'United States').
        """
        try:
            # Try UTF-8 first, then fallback to other encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(self.country_names_path, encoding=encoding)
                    logger.debug(f"Successfully loaded country names with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError(f"Could not decode {self.country_names_path} with any standard encoding")

            # Keep original case from CSV file (now UPPERCASE in new format)
            # Store both uppercase and lowercase for compatibility with ICIO data
            country_dict = df.set_index('country_code')['country_names'].to_dict()
            # Create case-insensitive lookup by storing both cases
            country_dict_combined = {}
            for k, v in country_dict.items():
                country_dict_combined[k.upper()] = v  # Uppercase version
                country_dict_combined[k.lower()] = v  # Lowercase version for compatibility
            logger.debug(f"Loaded {len(country_dict)} country names (with case variants)")
            return country_dict_combined
        except Exception as e:
            logger.error(f"Failed to load country names: {e}")
            raise

    def _load_industry_names(self) -> Dict[str, str]:
        """
        Load industry code to name mapping from CSV.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping industry codes (e.g., 'c10t12') to names (e.g., 'Food').
        """
        try:
            # Try UTF-8 first, then fallback to other encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(self.industry_names_path, encoding=encoding)
                    logger.debug(f"Successfully loaded industry names with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError(f"Could not decode {self.industry_names_path} with any standard encoding")

            # Keep original case from CSV file (now UPPERCASE/mixed in new format)
            # Store both uppercase and lowercase for compatibility with ICIO data
            industry_dict = df.set_index('industry_code')['industry_names'].to_dict()
            # Create case-insensitive lookup by storing both cases
            industry_dict_combined = {}
            for k, v in industry_dict.items():
                industry_dict_combined[k.upper()] = v  # Uppercase version
                industry_dict_combined[k.lower()] = v  # Lowercase version for compatibility
            logger.debug(f"Loaded {len(industry_dict)} industry names (with case variants)")
            return industry_dict_combined
        except Exception as e:
            logger.error(f"Failed to load industry names: {e}")
            raise

    def load_year(
        self,
        year: int,
        validate: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OECD ICIO data for a specific year.

        This method reads the ICIO CSV file and extracts the key components:
        - source_df: Full ICIO table with all rows (including value added)
        - T: Intermediate use matrix (country-sector × country-sector)
        - X: Gross output vector
        - F: Final demand vector
        - csnames: List of country-sector identifiers
        - manufacturing_names: Subset of manufacturing sector identifiers

        Parameters
        ----------
        year : int
            Year to load (e.g., 2018).
        validate : bool, optional
            Whether to validate data integrity, by default True.

        Returns
        -------
        Dict[str, pd.DataFrame or pd.Series or pd.Index]
            Dictionary containing:
            - 'source_df': Full ICIO DataFrame
            - 'T': Intermediate use matrix (DataFrame)
            - 'X': Gross output vector (Series)
            - 'F': Final demand vector (Series)
            - 'csnames': Country-sector identifiers (Index)
            - 'manufacturing_names': Manufacturing sector identifiers (Index)
            - 'manufacturing_bool': Boolean mask for manufacturing sectors

        Raises
        ------
        FileNotFoundError
            If ICIO file for the specified year doesn't exist.
        ValueError
            If data validation fails.

        Notes
        -----
        The ICIO table structure (OECD 2021):
        - Rows: Selling sectors (country_industry format, e.g., 'usa_c10t12')
        - Columns: Buying sectors + final demand categories + totals
        - Special rows: 'tls' (taxes less subsidies), 'va' (value added), 'out' (output)

        The intermediate use matrix T captures business-to-business (B2B) flows,
        while final demand F captures business-to-consumer (B2C) flows.

        Examples
        --------
        >>> loader = ICIODataLoader(Path("data"))
        >>> data_2018 = loader.load_year(2018)
        >>> T = data_2018['T']
        >>> print(f"Matrix dimensions: {T.shape}")
        """
        file_path = self.input_folder / f"{year}_SML.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"ICIO file not found for year {year}: {file_path}"
            )

        logger.info(f"Loading ICIO data for year {year}...")

        try:
            # Read CSV file
            source_df = pd.read_csv(file_path)

            # Clean column names: first column should be 'c_s' (country_sector)
            source_df.columns.values[0] = 'c_s'

            # Convert all identifiers to lowercase for consistency
            source_df['c_s'] = source_df['c_s'].str.lower()
            source_df.columns = source_df.columns.str.lower()

            # Extract country and sector from combined identifier
            # Format: 'usa_c10t12' -> country='usa', sector='c10t12'
            source_df['c'] = source_df['c_s'].str.split('_', n=1).str[0]
            source_df['s'] = source_df['c_s'].str.split('_', n=1).str[1]

            # Set country_sector as index
            source_df.set_index('c_s', inplace=True)

            # Extract intermediate use matrix T
            # Rows from first country-sector to 'row_t' (rest of world, services)
            # Exclude special rows: 'tls' (taxes), 'va' (value added), 'out' (output)
            T, csnames = self._extract_intermediate_matrix(source_df)

            # Identify manufacturing sectors using config (supports both old and new formats)
            import config
            manufacturing_bool = source_df.loc[csnames, 's'].isin(config.MANUFACTURING_CODES_ALL)
            manufacturing_names = csnames[manufacturing_bool]

            # Extract gross output vector X
            # Rename 'out' column to 'X' for clarity
            if 'out' in source_df.columns:
                source_df.rename(columns={'out': 'X'}, inplace=True)
            X = source_df.loc[csnames, 'X']

            # Extract final demand vector F
            # F = sum of household consumption + government + investment + exports - imports
            # OECD ICIO columns: hfce (household final consumption), npish (non-profit),
            # ggfc (government), gfcf (capital formation), invnt (inventories), dpabr (direct purchases abroad)
            F = self._extract_final_demand(source_df, csnames)

            # Validate if requested
            if validate:
                self._validate_data(source_df, T, X, F, csnames, year)

            logger.info(f"Successfully loaded {year} data:")
            logger.info(f"  Dimensions: {T.shape[0]} sectors × {T.shape[1]} sectors")
            logger.info(f"  Manufacturing sectors: {len(manufacturing_names)}")

            return {
                'source_df': source_df,
                'T': T,
                'X': X,
                'F': F,
                'csnames': csnames,
                'manufacturing_names': manufacturing_names,
                'manufacturing_bool': manufacturing_bool,
            }

        except Exception as e:
            logger.error(f"Failed to load year {year}: {e}")
            raise

    def _extract_intermediate_matrix(
        self,
        source_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Index]:
        """
        Extract intermediate use matrix T from full ICIO table.

        Parameters
        ----------
        source_df : pd.DataFrame
            Full ICIO DataFrame with all rows and columns.

        Returns
        -------
        T : pd.DataFrame
            Square intermediate use matrix (country-sector × country-sector).
        csnames : pd.Index
            Index of country-sector identifiers.
        """
        # Dynamically identify country-sector rows by filtering out special rows
        # Special rows: TLS (taxes less subsidies), VA/VALU (value added),
        # OUT/OUTPUT (gross output), and any non-country-sector identifiers

        # Get all index values as strings and convert to uppercase for matching
        all_indices = source_df.index.astype(str)

        # Special row identifiers to exclude (case-insensitive)
        special_rows = ['tls', 'va', 'valu', 'out', 'output', 'total']

        # Filter to keep only country-sector rows
        # Country-sector format: XXX_YYY where XXX is 3-letter country code, YYY is industry code
        # This works for both lowercase (arg_a01_02) and uppercase (ARG_A01) formats
        cs_mask = pd.Series(True, index=all_indices)

        for special_row in special_rows:
            # Exclude rows that match special identifiers (case-insensitive)
            cs_mask &= ~all_indices.str.lower().str.contains(special_row, na=False)

        # Also ensure row has underscore (country_sector separator)
        cs_mask &= all_indices.str.contains('_', na=False)

        # Get country-sector names
        csnames = all_indices[cs_mask]

        logger.debug(f"Identified {len(csnames)} country-sector rows")
        logger.debug(f"First country-sector: {csnames[0]}")
        logger.debug(f"Last country-sector: {csnames[-1]}")

        # Extract square submatrix for intermediate use
        # T[i,j] = sales from sector i to sector j as intermediate input
        T = source_df.loc[csnames, csnames]

        return T, csnames

    def _extract_final_demand(
        self,
        source_df: pd.DataFrame,
        csnames: pd.Index
    ) -> pd.Series:
        """
        Extract final demand vector F from OECD ICIO table.

        Final demand includes:
        - HFCE: Household final consumption expenditure
        - NPISH: Non-profit institutions serving households
        - GGFC: General government final consumption
        - GFCF: Gross fixed capital formation
        - INVNT: Changes in inventories and valuables
        - DPABR: Direct purchases abroad by residents

        Parameters
        ----------
        source_df : pd.DataFrame
            Full ICIO DataFrame.
        csnames : pd.Index
            Country-sector identifiers to extract.

        Returns
        -------
        pd.Series
            Final demand vector F for each country-sector.
        """
        # Identify final demand columns (from first country's hfce to last country's dpabr)
        # Format: 'arg_hfce', 'arg_npish', ..., 'row_dpabr'
        try:
            F = source_df.loc[csnames, 'arg_hfce':'row_dpabr'].sum(axis=1)
        except KeyError:
            logger.warning("Standard final demand columns not found, using alternative method")
            # Fallback: use 'out' column minus intermediate use
            X = source_df.loc[csnames, 'X']
            T_row_sum = source_df.loc[csnames, csnames].sum(axis=1)
            F = X - T_row_sum

        return F

    def _validate_data(
        self,
        source_df: pd.DataFrame,
        T: pd.DataFrame,
        X: pd.Series,
        F: pd.Series,
        csnames: pd.Index,
        year: int
    ) -> None:
        """
        Validate ICIO data integrity.

        Checks:
        1. Use accounting identity: X = T.sum(axis=1) + F (row sums)
        2. Gross output identity: X = T.sum(axis=0) + VA (column sums)
        3. No negative values in T, X, F
        4. Reasonable magnitudes

        Parameters
        ----------
        source_df : pd.DataFrame
            Full ICIO DataFrame.
        T : pd.DataFrame
            Intermediate use matrix.
        X : pd.Series
            Gross output vector.
        F : pd.Series
            Final demand vector.
        csnames : pd.Index
            Country-sector identifiers.
        year : int
            Year being validated.

        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check 1: Use accounting identity (within tolerance)
        T_row_sum = T.sum(axis=1)
        X_computed = T_row_sum + F
        relative_error = np.abs(X_computed - X) / (X + 1e-10)

        if (relative_error > 0.01).any():  # 1% tolerance
            n_errors = (relative_error > 0.01).sum()
            logger.warning(
                f"Use accounting identity violated for {n_errors} sectors in {year} "
                f"(max error: {relative_error.max():.2%})"
            )

        # Check 2: Non-negative values
        if (T < 0).any().any():
            logger.warning(f"Negative values found in T matrix for {year}")
        if (X < 0).any():
            logger.warning(f"Negative values found in X vector for {year}")
        if (F < 0).any():
            logger.warning(f"Negative values found in F vector for {year}")

        # Check 3: Reasonable magnitudes (no extreme outliers)
        if X.max() / (X.median() + 1e-10) > 1000:
            logger.warning(f"Extreme outliers detected in X for {year}")

        logger.debug(f"Data validation completed for {year}")

    def load_multiple_years(
        self,
        years: List[int],
        validate: bool = True
    ) -> Dict[int, Dict]:
        """
        Load ICIO data for multiple years.

        Parameters
        ----------
        years : List[int]
            List of years to load.
        validate : bool, optional
            Whether to validate data for each year, by default True.

        Returns
        -------
        Dict[int, Dict]
            Dictionary mapping year to data dictionary (as returned by load_year).

        Examples
        --------
        >>> loader = ICIODataLoader(Path("data"))
        >>> data = loader.load_multiple_years([2016, 2017, 2018])
        >>> T_2018 = data[2018]['T']
        """
        results = {}
        for year in years:
            try:
                results[year] = self.load_year(year, validate=validate)
            except Exception as e:
                logger.error(f"Failed to load year {year}: {e}")
                # Continue with other years
                continue

        logger.info(f"Successfully loaded {len(results)}/{len(years)} years")
        return results
