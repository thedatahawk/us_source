"""
Configuration module for OECD ICIO Analysis Package.

This module centralizes all configuration settings, avoiding hard-coded paths
and magic numbers throughout the codebase. Users should modify this file
to set paths and parameters appropriate to their environment.

Author: Refactored from original by Baldwin, Freeman, Theodorakopoulos methodology
Date: 2025
"""

import os
from pathlib import Path
from typing import List, Dict

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directory - users should modify this to point to their data location
BASE_FOLDER = Path(r"C:\Users\willi\OneDrive\datahawk\python_projects\input_output")

# Input/Output folders
INPUT_FOLDER = BASE_FOLDER / "icio_files"
OUTPUT_FOLDER = BASE_FOLDER / "output"
AUXILIARY_OUTPUT_FOLDER = BASE_FOLDER / "output_auxiliary"

# Metadata files
COUNTRY_NAMES_PATH = BASE_FOLDER / "country_names.csv"
INDUSTRY_NAMES_PATH = BASE_FOLDER / "ind_names.csv"

# Research papers folder
RESEARCH_PAPERS_FOLDER = BASE_FOLDER / "research_papers"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Year range for analysis
YEAR_START = 1995
YEAR_END = 2022

# Manufacturing industry codes (OECD ICIO classification - NEW FORMAT)
# Updated to match new industry code format (UPPERCASE, different granularity)
# Note: Codes stored in both uppercase and lowercase for compatibility
MANUFACTURING_CODES = [
    "C10T12",   # Food & beverages
    "C13T15",   # Textiles & apparel
    "C16",      # Wood products
    "C17_18",   # Paper & printing
    "C19",      # Petroleum products
    "C20",      # Chemicals
    "C21",      # Pharmaceuticals
    "C22",      # Rubber & plastics
    "C23",      # Non-metallic minerals
    "C24A",     # Basic iron & steel (NEW: split from C24)
    "C24B",     # Non-ferrous metals (NEW: split from C24)
    "C25",      # Fabricated metals
    "C26",      # Electronics
    "C27",      # Electrical equipment
    "C28",      # Machinery
    "C29",      # Motor vehicles
    "C301",     # Shipbuilding (NEW: split from C30)
    "C302T309", # Other transport equip. (NEW: split from C30)
    "C31T33",   # Furniture & other manuf.
]

# Also store lowercase versions for compatibility with ICIO data files
MANUFACTURING_CODES_LOWER = [code.lower() for code in MANUFACTURING_CODES]

# Combined list for matching (checks both cases)
MANUFACTURING_CODES_ALL = MANUFACTURING_CODES + MANUFACTURING_CODES_LOWER

# Sector groupings (for aggregation analysis) - UPDATED FOR NEW FORMAT
# Maps individual industry codes to broader sector groups
SECTOR_GROUPS: Dict[str, List[str]] = {
    "01t03": ["A01", "A02", "A03", "a01", "a02", "a03"],  # Agriculture, forestry, fishing
    "05t09": ["B05", "B06", "B07", "B08", "B09", "b05", "b06", "b07", "b08", "b09"],  # Mining and quarrying
    "10t33": MANUFACTURING_CODES_ALL,  # All manufacturing (both cases)
    "35t43": ["D", "E", "F", "d", "e", "f"],  # Utilities, water, construction
    "45t98": [  # Services (both uppercase and lowercase)
        "G", "H49", "H50", "H51", "H52", "H53", "I", "J58T60", "J61", "J62_63",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "g", "h49", "h50", "h51", "h52", "h53", "i", "j58t60", "j61", "j62_63",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"
    ],
}

# =============================================================================
# EXPOSURE INDICATORS TO COMPUTE
# =============================================================================

# List of exposure variables to calculate
# FPEM: Foreign Production Exposure Measure (look-through, using Leontief inverse)
# FPEMfv: Face Value exposure (direct bilateral only, using I+A)
# FPEMhe: Hidden Exposure (difference between FPEM and FPEMfv)
EXPOSURE_VARIABLES = ['FPEM', 'FPEMfv', 'FPEMhe']

# =============================================================================
# COMPUTATIONAL SETTINGS
# =============================================================================

# Numerical tolerance for matrix operations
NUMERICAL_TOLERANCE = 1e-10

# Maximum condition number for matrix inversion (detect near-singular matrices)
MAX_CONDITION_NUMBER = 1e15

# Whether to use pseudo-inverse for singular matrices
USE_PSEUDO_INVERSE_FALLBACK = True

# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

# Save intermediate results (pickle files)
SAVE_INTERMEDIATE_RESULTS = False

# Create time series files (can be very large)
CREATE_TIME_SERIES_FILES = False

# Analysis modes
CREATE_VECTORS = False  # Create vector summary statistics
CREATE_ALL_INDUSTRY = True  # Analyze all industries
CREATE_MANUFACTURING_ONLY = False  # Analyze manufacturing industries only

# =============================================================================
# LOGGING
# =============================================================================

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config() -> bool:
    """
    Validate that all required paths exist and configuration is sensible.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise.
    """
    errors = []

    # Check if required input folders exist
    if not INPUT_FOLDER.exists():
        errors.append(f"Input folder not found: {INPUT_FOLDER}")

    if not COUNTRY_NAMES_PATH.exists():
        errors.append(f"Country names file not found: {COUNTRY_NAMES_PATH}")

    if not INDUSTRY_NAMES_PATH.exists():
        errors.append(f"Industry names file not found: {INDUSTRY_NAMES_PATH}")

    # Check year range
    if YEAR_START > YEAR_END:
        errors.append(f"Invalid year range: {YEAR_START} > {YEAR_END}")

    if YEAR_START < 1995:
        errors.append(f"Year start {YEAR_START} predates OECD ICIO data (1995)")

    # Create output folders if they don't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if SAVE_INTERMEDIATE_RESULTS:
        AUXILIARY_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Report errors
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    if validate_config():
        print("Configuration valid!")
        print(f"  Input folder: {INPUT_FOLDER}")
        print(f"  Output folder: {OUTPUT_FOLDER}")
        print(f"  Year range: {YEAR_START}-{YEAR_END}")
        print(f"  Manufacturing sectors: {len(MANUFACTURING_CODES)}")
    else:
        print("Configuration invalid. Please fix errors above.")
