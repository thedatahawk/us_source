"""
Main execution script for OECD ICIO Analysis Package.

This script provides a command-line interface for running input-output
analysis on OECD ICIO data. It orchestrates data loading, Leontief calculations,
value-added decomposition, and exposure indicator computation.

Usage:
    python main.py --years 2016 2017 2018 --countries usa chn --save-results

Author: Refactored following Baldwin, Freeman, & Theodorakopoulos (2022, 2023)
Date: 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# Import configuration
import config

# Import modules
from io_data import ICIODataLoader
from analysis import LeontiefAnalyzer, ValueAddedDecomposition, ExposureCalculator
from analysis.horses import HorsesClassificationEngine
from analysis.aggregated_outputs import AggregatedOutputGenerator
from utils.logging_config import setup_logger
from utils.helpers import generate_sector_groups

logger = setup_logger(__name__, level=config.LOG_LEVEL)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='OECD ICIO Input-Output Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single year
    python main.py --years 2018

    # Process range of years
    python main.py --years 2016 2017 2018

    # Filter specific sourcing countries
    python main.py --years 2018 --sourcing-countries usa

    # Save intermediate results
    python main.py --years 2018 --save-results

    # Manufacturing sectors only
    python main.py --years 2018 --manufacturing-only
        """
    )

    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[config.YEAR_END],
        help=f'Years to process (default: {config.YEAR_END})'
    )

    parser.add_argument(
        '--sourcing-countries',
        type=str,
        nargs='*',
        default=None,
        help='Filter to specific sourcing countries (e.g., usa chn). Default: all countries.'
    )

    parser.add_argument(
        '--selling-countries',
        type=str,
        nargs='*',
        default=None,
        help='Filter to specific selling countries. Default: all countries.'
    )

    parser.add_argument(
        '--manufacturing-only',
        action='store_true',
        help='Analyze manufacturing sectors only'
    )

    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save full bilateral exposure results (large files, 16M rows per year)'
    )

    parser.add_argument(
        '--aggregated-outputs',
        action='store_true',
        help='Create smaller aggregated output files (usa_sourcing.csv, world_sourcing.csv) - RECOMMENDED'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['pickle', 'csv', 'both'],
        default='csv',
        help='Output file format (default: csv)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate data integrity (default: True)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def process_year(
    year: int,
    loader: ICIODataLoader,
    args: argparse.Namespace
) -> Optional[pd.DataFrame]:
    """
    Process a single year of ICIO data.

    Parameters
    ----------
    year : int
        Year to process.
    loader : ICIODataLoader
        Data loader instance.
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    pd.DataFrame or None
        Combined results DataFrame, or None if processing failed.
    """
    logger.info(f"=" * 80)
    logger.info(f"PROCESSING YEAR {year}")
    logger.info(f"=" * 80)

    try:
        # Step 1: Load data
        logger.info("Step 1: Loading ICIO data...")
        data = loader.load_year(year, validate=args.validate)

        T = data['T']
        X = data['X']
        F = data['F']
        csnames = data['csnames']
        manufacturing_bool = data['manufacturing_bool']

        # Filter to manufacturing if requested
        if args.manufacturing_only:
            logger.info("Filtering to manufacturing sectors only...")
            T = T.loc[manufacturing_bool, manufacturing_bool]
            X = X[manufacturing_bool]
            F = F[manufacturing_bool]
            csnames = csnames[manufacturing_bool]

        logger.info(f"Loaded data with {len(csnames)} sectors")

        # Step 2: Leontief analysis
        logger.info("Step 2: Computing Leontief inverse...")
        leontief = LeontiefAnalyzer(T, X, F, csnames, validate=args.validate)
        A = leontief.compute_technical_coefficients()
        L = leontief.compute_leontief_inverse()

        logger.info(f"Leontief inverse computed successfully")
        logger.info(f"  Used pseudo-inverse: {leontief.L_used_pseudo}")

        # Step 3: Exposure indicators
        logger.info("Step 3: Computing exposure indicators...")
        exposure_calc = ExposureCalculator(L, F.to_numpy(), A, csnames)
        indicators = exposure_calc.compute_all_indicators()

        logger.info("Computed exposure indicators:")
        for name, matrix in indicators.items():
            logger.info(f"  {name}: shape {matrix.shape}")

        # Step 4: Create bilateral exposure DataFrame
        logger.info("Step 4: Creating bilateral exposure DataFrame...")
        df_bilateral = exposure_calc.create_bilateral_exposure_dataframe(indicators)

        # Add year column
        df_bilateral['year'] = year

        logger.info(f"Created bilateral DataFrame with {len(df_bilateral):,} rows")

        # Step 5: Filter by countries if requested
        if args.sourcing_countries or args.selling_countries:
            logger.info("Step 5: Filtering by countries...")
            if args.sourcing_countries:
                df_bilateral = df_bilateral[
                    df_bilateral['sourcing_country'].isin(args.sourcing_countries)
                ]
            if args.selling_countries:
                df_bilateral = df_bilateral[
                    df_bilateral['selling_country'].isin(args.selling_countries)
                ]
            logger.info(f"Filtered to {len(df_bilateral):,} rows")

        # Step 6: Save results if requested
        if args.save_results:
            logger.info("Step 6: Saving results...")
            save_results(year, df_bilateral, indicators, csnames, args)

        logger.info(f"Successfully processed year {year}")
        return df_bilateral

    except Exception as e:
        logger.error(f"Failed to process year {year}: {e}", exc_info=True)
        return None


def save_results(
    year: int,
    df_bilateral: pd.DataFrame,
    indicators: dict,
    csnames: pd.Index,
    args: argparse.Namespace
) -> None:
    """
    Save analysis results to disk.

    Parameters
    ----------
    year : int
        Year being saved.
    df_bilateral : pd.DataFrame
        Bilateral exposure DataFrame.
    indicators : dict
        Dictionary of indicator matrices.
    csnames : pd.Index
        Country-sector identifiers.
    args : argparse.Namespace
        Command-line arguments.
    """
    output_folder = config.OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save bilateral DataFrame
    if args.output_format in ['pickle', 'both']:
        pickle_path = output_folder / f"bilateral_exposure_{year}.pkl"
        df_bilateral.to_pickle(pickle_path)
        logger.info(f"Saved bilateral DataFrame to {pickle_path}")

    if args.output_format in ['csv', 'both']:
        csv_path = output_folder / f"bilateral_exposure_{year}.csv"
        df_bilateral.to_csv(csv_path, index=False)
        logger.info(f"Saved bilateral DataFrame to {csv_path}")

    # Save indicator matrices if requested
    if config.SAVE_INTERMEDIATE_RESULTS:
        aux_folder = config.AUXILIARY_OUTPUT_FOLDER
        aux_folder.mkdir(parents=True, exist_ok=True)

        for name, matrix in indicators.items():
            df = pd.DataFrame(matrix, index=csnames, columns=csnames)
            pickle_path = aux_folder / f"{name}_{year}.pkl"
            df.to_pickle(pickle_path)
            logger.debug(f"Saved {name} matrix to {pickle_path}")


def combine_multi_year_results(results: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine results from multiple years into single DataFrame.

    Parameters
    ----------
    results : List[pd.DataFrame]
        List of yearly result DataFrames.

    Returns
    -------
    pd.DataFrame
        Combined multi-year DataFrame.
    """
    df_combined = pd.concat(results, axis=0, ignore_index=True)
    logger.info(f"Combined {len(results)} years into DataFrame with {len(df_combined):,} rows")
    return df_combined


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics for processed data.

    Parameters
    ----------
    df : pd.DataFrame
        Bilateral exposure DataFrame.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    # Years covered
    years = sorted(df['year'].unique())
    logger.info(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")

    # Countries
    sourcing_countries = df['sourcing_country'].nunique()
    selling_countries = df['selling_country'].nunique()
    logger.info(f"Sourcing countries: {sourcing_countries}")
    logger.info(f"Selling countries: {selling_countries}")

    # Exposure indicators
    logger.info("")
    logger.info("Exposure Indicators (all years, all countries):")
    for indicator in ['FPEM', 'FPEMfv', 'FPEMhe']:
        if indicator in df.columns:
            mean_val = df[indicator].mean()
            median_val = df[indicator].median()
            max_val = df[indicator].max()
            logger.info(f"  {indicator}: mean={mean_val:.4f}, median={median_val:.4f}, max={max_val:.4f}")

    # Top exposures (largest FPEM values)
    logger.info("")
    logger.info("Top 10 Bilateral Exposures (by FPEM):")
    top10 = df.nlargest(10, 'FPEM')[['selling', 'sourcing', 'FPEM', 'year']]
    for idx, row in top10.iterrows():
        logger.info(f"  {row['selling']} -> {row['sourcing']}: {row['FPEM']:.4f} ({row['year']})")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Set verbose logging if requested
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    # Validate configuration
    logger.info("Validating configuration...")
    if not config.validate_config():
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)

    logger.info("Configuration validated successfully")
    logger.info(f"Input folder: {config.INPUT_FOLDER}")
    logger.info(f"Output folder: {config.OUTPUT_FOLDER}")

    # Initialize data loader
    logger.info("Initializing data loader...")
    loader = ICIODataLoader(
        input_folder=config.INPUT_FOLDER,
        country_names_path=config.COUNTRY_NAMES_PATH,
        industry_names_path=config.INDUSTRY_NAMES_PATH
    )

    # Process each year
    results = []
    for year in args.years:
        result = process_year(year, loader, args)
        if result is not None:
            results.append(result)

    if not results:
        logger.error("No years processed successfully. Exiting.")
        sys.exit(1)

    # Create aggregated outputs if requested (memory-efficient incremental processing)
    if args.aggregated_outputs:
        logger.info("")
        logger.info("Creating aggregated output files (incremental processing)...")

        # Prepare country and industry mappings
        country_names_dict = loader.country_names
        industry_names_dict = loader.industry_names

        # Get list of countries (excluding row and foreign) from first year
        all_countries = set()
        df_sample = results[0]
        for cs in df_sample['selling'].str.split('_', n=1).str[0].str.lower().unique():
            if cs not in ['row', 'foreign']:
                all_countries.add(cs)
        for cs in df_sample['sourcing'].str.split('_', n=1).str[0].str.lower().unique():
            if cs not in ['row', 'foreign']:
                all_countries.add(cs)
        country_list = list(all_countries)

        # Initialize aggregated output generator
        agg_generator = AggregatedOutputGenerator(
            country_names=country_names_dict,
            industry_names=industry_names_dict,
            country_list=country_list
        )

        # Convert manufacturing codes to lowercase for matching
        manufacturing_codes_lower = [code.lower() for code in config.MANUFACTURING_CODES]

        # Create aggregated outputs incrementally (reads CSV files year-by-year)
        saved_files = agg_generator.save_aggregated_outputs_incremental(
            years=args.years,
            manufacturing_codes=manufacturing_codes_lower,
            output_folder=config.OUTPUT_FOLDER
        )

    # Print summary statistics (only combine in memory if user has few years)
    if len(results) <= 5:
        logger.info("")
        logger.info("Combining results for summary statistics...")
        if len(results) > 1:
            df_combined = combine_multi_year_results(results)
        else:
            df_combined = results[0]
        print_summary_statistics(df_combined)
    else:
        logger.info("")
        logger.info("Skipping combined summary statistics (too many years - would exceed memory)")
        logger.info("Summary statistics for last year only:")
        print_summary_statistics(results[-1])

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed {len(results)} year(s) successfully")
    if len(results) > 0:
        logger.info(f"Observations per year: ~{len(results[0]):,}")

    if args.save_results:
        logger.info(f"Full bilateral results saved to: {config.OUTPUT_FOLDER}")

    if args.aggregated_outputs:
        logger.info(f"Aggregated outputs saved to: {config.OUTPUT_FOLDER}")
        logger.info(f"  - usa_sourcing.csv: USA as sourcing country (all industries)")
        logger.info(f"  - world_sourcing.csv: Manufacturing sectors (all countries)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
