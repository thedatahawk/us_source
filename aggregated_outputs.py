"""
Aggregated Output Generator Module.

This module creates smaller, aggregated output files similar to the original
input_output.py script. Instead of saving full 16M row bilateral DataFrames,
it creates focused outputs:
- USA sourcing data (all industries)
- World manufacturing sourcing data
- Aggregated indicators by country

These files are much smaller and more practical for analysis.

Author: Refactored OECD ICIO Analysis Package
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class AggregatedOutputGenerator:
    """
    Generate aggregated output files for practical analysis.

    This class creates smaller, focused output files instead of the full
    16M row bilateral exposure DataFrames. It replicates the key outputs
    from the original input_output.py script:

    1. USA sourcing data (all industries, all countries)
    2. World manufacturing sourcing data (manufacturing only, all countries)
    3. Country-level aggregated indicators

    These files aggregate data to country or country-industry level,
    making them much more manageable for analysis and visualization.

    Attributes
    ----------
    country_names : Dict[str, str]
        Mapping from country codes to country names
    industry_names : Dict[str, str]
        Mapping from industry codes to industry names
    country_list : List[str]
        List of country codes (excluding 'row' and 'foreign')

    Examples
    --------
    >>> from pathlib import Path
    >>> generator = AggregatedOutputGenerator(
    ...     country_names={'usa': 'United States', 'chn': 'China'},
    ...     industry_names={'c10t12': 'Food & beverages'},
    ...     country_list=['usa', 'chn']
    ... )
    >>> df_bilateral = pd.read_csv('bilateral_exposure_2018.csv')
    >>> usa_sourcing = generator.create_usa_sourcing_data([df_bilateral])
    """

    def __init__(
        self,
        country_names: Dict[str, str],
        industry_names: Dict[str, str],
        country_list: List[str]
    ):
        """
        Initialize aggregated output generator.

        Parameters
        ----------
        country_names : Dict[str, str]
            Mapping from country codes to country names
        industry_names : Dict[str, str]
            Mapping from industry codes to industry names
        country_list : List[str]
            List of country codes (excluding 'row' and 'foreign')
        """
        self.country_names = country_names
        self.industry_names = industry_names
        self.country_list = country_list

        logger.info("AggregatedOutputGenerator initialized")

    def create_usa_sourcing_data(
        self,
        df_list: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create USA sourcing data (USA as sourcing country, all industries).

        This replicates the usa_sourcing.pkl output from the original code.
        It filters for USA as the sourcing country and aggregates selling
        countries into domestic/foreign/row categories.

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of bilateral exposure DataFrames (one per year)

        Returns
        -------
        pd.DataFrame
            USA sourcing data with columns:
            - year
            - sourcing_country (always 'United States')
            - sourcing_industry
            - sourcing_industry_name
            - selling_country
            - selling_industry
            - selling_industry_name
            - FPEM (as percentage)
            - FPEMfv (as percentage)
            - FPEMhe (as percentage)

        Notes
        -----
        This aggregates selling countries into:
        - Specific countries in the sample
        - 'row' (Rest of World) for countries not in sample
        - 'foreign' (all non-USA countries combined)
        """
        logger.info("Creating USA sourcing data...")

        # Combine all years
        df_combined = pd.concat(df_list, ignore_index=True)

        # Filter for USA as sourcing country
        usa_df = df_combined[
            df_combined['sourcing'].str.lower().str.startswith('usa_')
        ].copy()

        if len(usa_df) == 0:
            logger.warning("No USA sourcing data found")
            return pd.DataFrame()

        # Extract country and industry codes
        usa_df['sourcing_country'] = usa_df['sourcing'].str.split('_', n=1).str[0].str.lower()
        usa_df['sourcing_industry'] = usa_df['sourcing'].str.split('_', n=1).str[1].str.lower()
        usa_df['selling_country'] = usa_df['selling'].str.split('_', n=1).str[0].str.lower()
        usa_df['selling_industry'] = usa_df['selling'].str.split('_', n=1).str[1].str.lower()

        # Aggregate selling countries to 'row' for non-sample countries
        usa_df['tosum'] = 'row'
        usa_df.loc[usa_df['selling_country'].isin(self.country_list), 'tosum'] = 'sample'
        usa_df.loc[usa_df['tosum'] == 'row', 'selling_country'] = 'row'
        usa_df = usa_df.drop(columns=['tosum'])

        # Create FOREIGN aggregate (all non-USA countries)
        usa_foreign = usa_df.copy()
        usa_foreign.loc[usa_foreign['selling_country'] != 'usa', 'selling_country'] = 'foreign'
        usa_foreign = usa_foreign[usa_foreign['selling_country'] == 'foreign']
        usa_foreign = usa_foreign.groupby([
            'year', 'sourcing_country', 'sourcing_industry',
            'selling_country', 'selling_industry'
        ])[['FPEM', 'FPEMfv', 'FPEMhe']].sum().reset_index()

        # Combine original data with FOREIGN aggregate
        usa_sourcing = pd.concat([usa_df, usa_foreign], axis=0)

        # Add industry names
        usa_sourcing['sourcing_industry_name'] = usa_sourcing['sourcing_industry'].map(self.industry_names)
        usa_sourcing['selling_industry_name'] = usa_sourcing['selling_industry'].map(self.industry_names)

        # Add country names
        usa_sourcing['selling_country'] = usa_sourcing['selling_country'].map(self.country_names)
        usa_sourcing['sourcing_country'] = usa_sourcing['sourcing_country'].map(self.country_names)

        # Convert to percentages
        for var in ['FPEM', 'FPEMfv', 'FPEMhe']:
            usa_sourcing[var] = 100 * usa_sourcing[var]

        # Select final columns
        usa_sourcing = usa_sourcing[[
            'year', 'sourcing_country', 'sourcing_industry', 'sourcing_industry_name',
            'selling_country', 'selling_industry', 'selling_industry_name',
            'FPEM', 'FPEMfv', 'FPEMhe'
        ]]

        logger.info(f"Created USA sourcing data with {len(usa_sourcing):,} rows")
        return usa_sourcing

    def create_world_manufacturing_data_chunk(
        self,
        df: pd.DataFrame,
        manufacturing_codes: List[str],
        skip_average: bool = False
    ) -> pd.DataFrame:
        """
        Create world manufacturing data for a single chunk (no averaging).

        This is used for chunked processing. The averaging step ('s10t33')
        should be done separately after all chunks are combined.
        """
        # Extract country and industry codes
        df['selling_country'] = df['selling'].str.split('_', n=1).str[0].str.lower()
        df['sourcing_country'] = df['sourcing'].str.split('_', n=1).str[0].str.lower()
        df['selling_industry'] = df['selling'].str.split('_', n=1).str[1].str.lower()
        df['sourcing_industry'] = df['sourcing'].str.split('_', n=1).str[1].str.lower()

        # Filter for manufacturing sourcing industries
        manu_df = df[df['sourcing_industry'].isin(manufacturing_codes)].copy()

        if len(manu_df) == 0:
            return pd.DataFrame()

        # Aggregate by selling_country × sourcing (manufacturing industry)
        indicators_manu = manu_df.groupby([
            'year', 'selling_country', 'sourcing'
        ])[['FPEM', 'FPEMfv', 'FPEMhe']].sum().reset_index()

        # Extract sourcing country and industry from 'sourcing' column
        indicators_manu['sourcing_country'] = indicators_manu['sourcing'].str.split('_', n=1).str[0].str.lower()
        indicators_manu['sourcing_industry'] = indicators_manu['sourcing'].str.split('_', n=1).str[1].str.lower()
        indicators_manu = indicators_manu.drop(columns=['sourcing'])

        # Aggregate selling countries to 'row' for non-sample countries
        indicators_manu['tosum'] = 'row'
        indicators_manu.loc[indicators_manu['selling_country'].isin(self.country_list), 'tosum'] = 'sample'
        indicators_manu.loc[indicators_manu['tosum'] == 'row', 'selling_country'] = 'row'
        indicators_manu = indicators_manu.drop(columns=['tosum'])

        # Group by year, sourcing_country, sourcing_industry, selling_country
        world_heat_grouped = indicators_manu.groupby([
            'year', 'sourcing_country', 'sourcing_industry', 'selling_country'
        ]).sum().reset_index()

        # Create FOREIGN aggregate for each sourcing country
        world_foreign = pd.DataFrame()
        for country in world_heat_grouped['sourcing_country'].unique():
            temp_heat = world_heat_grouped[world_heat_grouped['sourcing_country'] == country].copy()
            temp_heat.loc[temp_heat['selling_country'] != country, 'selling_country'] = 'foreign'
            temp_heat = temp_heat.groupby([
                'year', 'sourcing_country', 'sourcing_industry', 'selling_country'
            ]).sum().reset_index()
            temp_heat = temp_heat[temp_heat['selling_country'] == 'foreign']
            world_foreign = pd.concat([world_foreign, temp_heat], axis=0)

        # Combine original data with FOREIGN aggregate
        world_heat = pd.concat([world_heat_grouped, world_foreign], axis=0)

        # Add industry names
        world_heat['sourcing_industry_name'] = world_heat['sourcing_industry'].map(self.industry_names)

        # Add country names
        world_heat['selling_country'] = world_heat['selling_country'].map(self.country_names)
        world_heat['sourcing_country'] = world_heat['sourcing_country'].map(self.country_names)

        # Convert to percentages
        for var in ['FPEM', 'FPEMfv', 'FPEMhe']:
            world_heat[var] = 100 * world_heat[var]

        return world_heat

    def create_world_manufacturing_data(
        self,
        df_list: List[pd.DataFrame],
        manufacturing_codes: List[str]
    ) -> pd.DataFrame:
        """
        Create world manufacturing sourcing data (manufacturing sectors only).

        This replicates the world_sourcing.pkl output from the original code.
        It filters for manufacturing sectors and creates aggregated indicators
        by selling country and sourcing manufacturing industry.

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of bilateral exposure DataFrames (one per year)
        manufacturing_codes : List[str]
            List of manufacturing industry codes

        Returns
        -------
        pd.DataFrame
            World manufacturing sourcing data with columns:
            - year
            - sourcing_country
            - sourcing_industry
            - sourcing_industry_name
            - selling_country
            - FPEM (as percentage)
            - FPEMfv (as percentage)
            - FPEMhe (as percentage)

        Notes
        -----
        This creates aggregates at the selling_country × sourcing_industry level,
        summing across all selling industries. It includes:
        - Specific manufacturing industries
        - 's10t33' (manufacturing average)
        - 'row' and 'foreign' selling country aggregates
        """
        logger.info("Creating world manufacturing sourcing data...")

        # Combine all years
        df_combined = pd.concat(df_list, ignore_index=True)

        # Extract country and industry codes
        df_combined['selling_country'] = df_combined['selling'].str.split('_', n=1).str[0].str.lower()
        df_combined['sourcing_country'] = df_combined['sourcing'].str.split('_', n=1).str[0].str.lower()
        df_combined['selling_industry'] = df_combined['selling'].str.split('_', n=1).str[1].str.lower()
        df_combined['sourcing_industry'] = df_combined['sourcing'].str.split('_', n=1).str[1].str.lower()

        # Filter for manufacturing sourcing industries
        manu_df = df_combined[
            df_combined['sourcing_industry'].isin(manufacturing_codes)
        ].copy()

        if len(manu_df) == 0:
            logger.warning("No manufacturing data found")
            return pd.DataFrame()

        # Aggregate by selling_country × sourcing (manufacturing industry)
        # This sums across all selling industries for each sourcing industry
        indicators_manu = manu_df.groupby([
            'year', 'selling_country', 'sourcing'
        ])[['FPEM', 'FPEMfv', 'FPEMhe']].sum().reset_index()

        # Extract sourcing country and industry from 'sourcing' column
        indicators_manu['sourcing_country'] = indicators_manu['sourcing'].str.split('_', n=1).str[0].str.lower()
        indicators_manu['sourcing_industry'] = indicators_manu['sourcing'].str.split('_', n=1).str[1].str.lower()
        indicators_manu = indicators_manu.drop(columns=['sourcing'])

        # Aggregate selling countries to 'row' for non-sample countries
        indicators_manu['tosum'] = 'row'
        indicators_manu.loc[indicators_manu['selling_country'].isin(self.country_list), 'tosum'] = 'sample'
        indicators_manu.loc[indicators_manu['tosum'] == 'row', 'selling_country'] = 'row'
        indicators_manu = indicators_manu.drop(columns=['tosum'])

        # Group by year, sourcing_country, sourcing_industry, selling_country
        world_heat_grouped = indicators_manu.groupby([
            'year', 'sourcing_country', 'sourcing_industry', 'selling_country'
        ]).sum().reset_index()

        # Create FOREIGN aggregate for each sourcing country
        world_foreign = pd.DataFrame()
        for country in world_heat_grouped['sourcing_country'].unique():
            temp_heat = world_heat_grouped[world_heat_grouped['sourcing_country'] == country].copy()
            temp_heat.loc[temp_heat['selling_country'] != country, 'selling_country'] = 'foreign'
            temp_heat = temp_heat.groupby([
                'year', 'sourcing_country', 'sourcing_industry', 'selling_country'
            ]).sum().reset_index()
            temp_heat = temp_heat[temp_heat['selling_country'] == 'foreign']
            world_foreign = pd.concat([world_foreign, temp_heat], axis=0)

        # Combine original data with FOREIGN aggregate
        world_heat = pd.concat([world_heat_grouped, world_foreign], axis=0)

        # Create manufacturing average (s10t33)
        average_world_heat = world_heat.groupby([
            'year', 'sourcing_country', 'selling_country'
        ])[['FPEM', 'FPEMfv', 'FPEMhe']].mean().reset_index()
        average_world_heat['sourcing_industry'] = 's10t33'
        average_world_heat['sourcing_industry_name'] = 'Manuf. avg.'

        # Combine specific industries with average
        world_sourcing = pd.concat([world_heat, average_world_heat], axis=0)

        # Add industry names (for specific industries, not the average)
        world_sourcing.loc[
            world_sourcing['sourcing_industry'] != 's10t33',
            'sourcing_industry_name'
        ] = world_sourcing['sourcing_industry'].map(self.industry_names)

        # Add country names
        world_sourcing['selling_country'] = world_sourcing['selling_country'].map(self.country_names)
        world_sourcing['sourcing_country'] = world_sourcing['sourcing_country'].map(self.country_names)

        # Convert to percentages
        for var in ['FPEM', 'FPEMfv', 'FPEMhe']:
            world_sourcing[var] = 100 * world_sourcing[var]

        logger.info(f"Created world manufacturing sourcing data with {len(world_sourcing):,} rows")
        return world_sourcing

    def save_aggregated_outputs(
        self,
        df_list: List[pd.DataFrame],
        manufacturing_codes: List[str],
        output_folder: Path
    ) -> Dict[str, Path]:
        """
        Create and save all aggregated output files.

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            List of bilateral exposure DataFrames (one per year)
        manufacturing_codes : List[str]
            List of manufacturing industry codes
        output_folder : Path
            Directory to save output files

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping output name to file path
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Create USA sourcing data
        logger.info("=" * 80)
        logger.info("CREATING AGGREGATED OUTPUT FILES")
        logger.info("=" * 80)

        usa_sourcing = self.create_usa_sourcing_data(df_list)
        if len(usa_sourcing) > 0:
            output_path = output_folder / "usa_sourcing.csv"
            usa_sourcing.to_csv(output_path, index=False)
            logger.info(f"Saved USA sourcing data: {output_path}")
            saved_files['usa_sourcing'] = output_path

        # Create world manufacturing data
        world_sourcing = self.create_world_manufacturing_data(df_list, manufacturing_codes)
        if len(world_sourcing) > 0:
            output_path = output_folder / "world_sourcing.csv"
            world_sourcing.to_csv(output_path, index=False)
            logger.info(f"Saved world manufacturing sourcing data: {output_path}")
            saved_files['world_sourcing'] = output_path

        logger.info("=" * 80)
        logger.info(f"CREATED {len(saved_files)} AGGREGATED OUTPUT FILES")
        logger.info("=" * 80)

        return saved_files

    def save_aggregated_outputs_incremental(
        self,
        years: List[int],
        manufacturing_codes: List[str],
        output_folder: Path
    ) -> Dict[str, Path]:
        """
        Create and save aggregated output files incrementally (year-by-year).

        This method reads individual year CSV files one at a time, processes them,
        and appends results directly to output files. This is truly memory-efficient
        as it never holds more than one year in memory.

        Parameters
        ----------
        years : List[int]
            List of years to process
        manufacturing_codes : List[str]
            List of manufacturing industry codes
        output_folder : Path
            Directory containing bilateral CSV files and where output files will be saved

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping output name to file path
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        logger.info("=" * 80)
        logger.info("CREATING AGGREGATED OUTPUT FILES (INCREMENTAL APPEND)")
        logger.info("=" * 80)

        # Process USA sourcing data incrementally with append
        logger.info("Processing USA sourcing data year-by-year...")
        usa_output_path = output_folder / "usa_sourcing.csv"

        for i, year in enumerate(years):
            csv_path = output_folder / f"bilateral_exposure_{year}.csv"
            if not csv_path.exists():
                logger.warning(f"Skipping year {year}: file not found at {csv_path}")
                continue

            logger.info(f"  Processing year {year}...")

            # Read CSV in chunks to reduce memory usage
            chunksize = 1000000  # 1M rows at a time
            first_chunk = True
            total_rows = 0

            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                usa_chunk = self.create_usa_sourcing_data([chunk])

                if len(usa_chunk) > 0:
                    # Write with header only for first chunk of first year
                    if i == 0 and first_chunk:
                        usa_chunk.to_csv(usa_output_path, index=False, mode='w')
                        first_chunk = False
                    else:
                        usa_chunk.to_csv(usa_output_path, index=False, mode='a', header=False)
                    total_rows += len(usa_chunk)

                del chunk, usa_chunk  # Free memory immediately

            if total_rows > 0:
                logger.info(f"    Appended {total_rows:,} rows to usa_sourcing.csv")

        if usa_output_path.exists():
            saved_files['usa_sourcing'] = usa_output_path
            logger.info(f"Saved USA sourcing data: {usa_output_path}")

        # Process world manufacturing data incrementally with append
        logger.info("Processing world manufacturing sourcing data year-by-year...")
        world_output_path = output_folder / "world_sourcing.csv"

        for i, year in enumerate(years):
            csv_path = output_folder / f"bilateral_exposure_{year}.csv"
            if not csv_path.exists():
                logger.warning(f"Skipping year {year}: file not found at {csv_path}")
                continue

            logger.info(f"  Processing year {year}...")

            # Read CSV in chunks to reduce memory usage
            chunksize = 1000000  # 1M rows at a time
            first_chunk = True
            total_rows = 0

            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                world_chunk = self.create_world_manufacturing_data_chunk(chunk, manufacturing_codes)

                if len(world_chunk) > 0:
                    # Write with header only for first chunk of first year
                    if i == 0 and first_chunk:
                        world_chunk.to_csv(world_output_path, index=False, mode='w')
                        first_chunk = False
                    else:
                        world_chunk.to_csv(world_output_path, index=False, mode='a', header=False)
                    total_rows += len(world_chunk)

                del chunk, world_chunk  # Free memory immediately

            if total_rows > 0:
                logger.info(f"    Appended {total_rows:,} rows to world_sourcing.csv")

        if world_output_path.exists():
            # Add manufacturing average ('s10t33') by post-processing the file
            logger.info("Computing manufacturing average (s10t33)...")
            logger.info("  Reading world_sourcing.csv in chunks...")

            # Read in chunks and compute average
            avg_chunks = []
            for chunk in pd.read_csv(world_output_path, chunksize=100000):
                avg_chunk = chunk.groupby([
                    'year', 'sourcing_country', 'selling_country'
                ])[['FPEM', 'FPEMfv', 'FPEMhe']].mean().reset_index()
                avg_chunks.append(avg_chunk)

            # Combine average chunks and aggregate
            if avg_chunks:
                avg_combined = pd.concat(avg_chunks, ignore_index=True)
                avg_final = avg_combined.groupby([
                    'year', 'sourcing_country', 'selling_country'
                ])[['FPEM', 'FPEMfv', 'FPEMhe']].mean().reset_index()
                avg_final['sourcing_industry'] = 's10t33'
                avg_final['sourcing_industry_name'] = 'Manuf. avg.'

                # Reorder columns to match the file structure
                avg_final = avg_final[[
                    'year', 'sourcing_country', 'sourcing_industry', 'selling_country',
                    'FPEM', 'FPEMfv', 'FPEMhe', 'sourcing_industry_name'
                ]]

                # Append to file
                avg_final.to_csv(world_output_path, index=False, mode='a', header=False)
                logger.info(f"  Appended {len(avg_final):,} manufacturing average rows")

            saved_files['world_sourcing'] = world_output_path
            logger.info(f"Saved world manufacturing sourcing data: {world_output_path}")

        logger.info("=" * 80)
        logger.info(f"CREATED {len(saved_files)} AGGREGATED OUTPUT FILES")
        logger.info("=" * 80)

        return saved_files
