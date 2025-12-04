"""
Horses-for-Courses (H4C) Classification Framework.

This module implements a modular, extensible classification system for
categorizing industries by production characteristics (labor-intensive,
capital-intensive, skill-intensive, etc.).

The "horses for courses" philosophy recognizes that:
1. Different supply chain indicators are appropriate for different shocks
2. Different industries have different vulnerabilities
3. Policy responses should be tailored to industry characteristics

This framework provides infrastructure for classifying industries and
mapping them to appropriate analytical approaches.

References
----------
Baldwin, Freeman, & Theodorakopoulos (2022, 2023): Core H4C philosophy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class IndustryCharacteristic(Enum):
    """Enumeration of industry production characteristics."""
    LABOR_INTENSIVE = "labor_intensive"
    CAPITAL_INTENSIVE = "capital_intensive"
    SKILL_INTENSIVE = "skill_intensive"
    RESOURCE_INTENSIVE = "resource_intensive"
    SCALE_INTENSIVE = "scale_intensive"
    TECH_INTENSIVE = "technology_intensive"


@dataclass
class IndustryClassification:
    """
    Classification of an industry by multiple characteristics.

    Attributes
    ----------
    industry_code : str
        OECD ICIO industry code (e.g., 'c10t12').
    industry_name : str
        Human-readable industry name.
    characteristics : Set[IndustryCharacteristic]
        Set of characteristics applying to this industry.
    priority_characteristic : IndustryCharacteristic
        Primary/dominant characteristic.
    notes : str
        Additional notes or justification.
    """
    industry_code: str
    industry_name: str
    characteristics: Set[IndustryCharacteristic]
    priority_characteristic: IndustryCharacteristic
    notes: str = ""


class HorsesClassificationEngine:
    """
    Modular classification engine for industry categorization.

    This engine allows users to define classification rules and apply them
    to OECD ICIO industries. The design is extensible: users can add new
    characteristic types and classification rules.

    Examples
    --------
    >>> engine = HorsesClassificationEngine()
    >>> engine.load_default_us_classifications()
    >>> classification = engine.get_classification('c10t12')
    >>> print(classification.primary_characteristic)
    """

    def __init__(self):
        """Initialize the classification engine."""
        self.classifications: Dict[str, IndustryClassification] = {}
        logger.info("HorsesClassificationEngine initialized")

    def add_classification(
        self,
        industry_code: str,
        industry_name: str,
        characteristics: List[str],
        priority: str,
        notes: str = ""
    ) -> None:
        """
        Add or update industry classification.

        Parameters
        ----------
        industry_code : str
            OECD industry code.
        industry_name : str
            Industry name.
        characteristics : List[str]
            List of characteristic names (from IndustryCharacteristic enum).
        priority : str
            Primary characteristic (must be in characteristics list).
        notes : str, optional
            Additional notes.
        """
        # Convert strings to enum members
        char_set = {IndustryCharacteristic(c) for c in characteristics}
        priority_char = IndustryCharacteristic(priority)

        if priority_char not in char_set:
            raise ValueError(f"Priority {priority} must be in characteristics list")

        classification = IndustryClassification(
            industry_code=industry_code,
            industry_name=industry_name,
            characteristics=char_set,
            priority_characteristic=priority_char,
            notes=notes
        )

        self.classifications[industry_code] = classification
        logger.debug(f"Added classification for {industry_code}: {industry_name}")

    def load_default_us_classifications(self) -> None:
        """
        Load default classifications for US-focused analysis.

        These classifications are based on standard industry characteristics
        and can be customized by users for their specific needs.

        Notes
        -----
        Classifications are informed by:
        - OECD STAN indicators
        - US Bureau of Labor Statistics data
        - Industry structure literature

        This is a starting point; users should validate and adjust based
        on their specific research questions.
        """
        # Food products - Labor and scale intensive
        self.add_classification(
            "c10t12", "Food Products",
            ["labor_intensive", "scale_intensive"],
            "labor_intensive",
            "Food processing involves significant labor for handling and packaging"
        )

        # Textiles and apparel - Highly labor intensive
        self.add_classification(
            "c13t15", "Textiles & Apparel",
            ["labor_intensive"],
            "labor_intensive",
            "Classic labor-intensive sector, often offshored to low-wage countries"
        )

        # Chemicals - Capital and scale intensive
        self.add_classification(
            "c20", "Chemicals",
            ["capital_intensive", "scale_intensive", "technology_intensive"],
            "capital_intensive",
            "Requires large capital investments in plants and equipment"
        )

        # Pharmaceuticals - Skill and technology intensive
        self.add_classification(
            "c21", "Pharmaceuticals",
            ["skill_intensive", "technology_intensive", "capital_intensive"],
            "skill_intensive",
            "R&D intensive, requires highly skilled workforce"
        )

        # Basic metals - Capital and scale intensive
        self.add_classification(
            "c24", "Basic Metals",
            ["capital_intensive", "scale_intensive", "resource_intensive"],
            "capital_intensive",
            "Steel, aluminum production requires massive capital investment"
        )

        # Electronics - Technology and skill intensive
        self.add_classification(
            "c26", "Electronics",
            ["technology_intensive", "skill_intensive", "capital_intensive"],
            "technology_intensive",
            "Semiconductors, computers require cutting-edge technology"
        )

        # Motor vehicles - Capital and scale intensive
        self.add_classification(
            "c29", "Motor Vehicles",
            ["capital_intensive", "scale_intensive", "technology_intensive"],
            "scale_intensive",
            "Auto manufacturing has huge scale economies and capital requirements"
        )

        # Add more as needed...
        logger.info(f"Loaded {len(self.classifications)} default classifications")

    def get_classification(self, industry_code: str) -> Optional[IndustryClassification]:
        """
        Retrieve classification for an industry.

        Parameters
        ----------
        industry_code : str
            OECD industry code.

        Returns
        -------
        IndustryClassification or None
            Classification if available, None otherwise.
        """
        return self.classifications.get(industry_code)

    def filter_by_characteristic(
        self,
        characteristic: str
    ) -> List[IndustryClassification]:
        """
        Find all industries with a given characteristic.

        Parameters
        ----------
        characteristic : str
            Characteristic name (from IndustryCharacteristic enum).

        Returns
        -------
        List[IndustryClassification]
            List of industries with this characteristic.

        Examples
        --------
        >>> engine = HorsesClassificationEngine()
        >>> engine.load_default_us_classifications()
        >>> labor_intensive = engine.filter_by_characteristic('labor_intensive')
        >>> print([c.industry_name for c in labor_intensive])
        """
        char = IndustryCharacteristic(characteristic)
        results = [
            c for c in self.classifications.values()
            if char in c.characteristics
        ]
        logger.debug(f"Found {len(results)} industries with characteristic '{characteristic}'")
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export classifications to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: industry_code, industry_name,
            priority_characteristic, all_characteristics, notes.
        """
        records = []
        for code, classification in self.classifications.items():
            records.append({
                'industry_code': code,
                'industry_name': classification.industry_name,
                'priority_characteristic': classification.priority_characteristic.value,
                'all_characteristics': ', '.join(
                    [c.value for c in classification.characteristics]
                ),
                'notes': classification.notes
            })

        df = pd.DataFrame(records)
        logger.debug(f"Exported {len(df)} classifications to DataFrame")
        return df

    def recommend_indicator(
        self,
        industry_code: str,
        shock_type: str
    ) -> str:
        """
        Recommend appropriate exposure indicator for industry and shock type.

        This implements the "horses for courses" principle: different indicators
        for different combinations of industry characteristics and shock types.

        Parameters
        ----------
        industry_code : str
            OECD industry code.
        shock_type : str
            Type of shock: 'production', 'trade', 'value_added', 'technology'.

        Returns
        -------
        str
            Recommended indicator name.

        Notes
        -----
        Recommendation logic (can be customized):
        - Production shocks → FPEM (look-through)
        - Trade shocks → FPEMfv (face value)
        - Value-added shocks → VA-based indicators
        - For labor-intensive sectors → emphasize employment metrics
        - For capital-intensive → emphasize capital flow metrics

        Examples
        --------
        >>> engine.recommend_indicator('c29', 'production')
        'FPEM (look-through exposure for production shock)'
        """
        classification = self.get_classification(industry_code)

        if classification is None:
            logger.warning(f"No classification for {industry_code}, using default")
            return "FPEM (default)"

        # Simple recommendation logic (can be made more sophisticated)
        if shock_type == 'production':
            return "FPEM (look-through exposure for gross production shocks)"
        elif shock_type == 'trade':
            return "FPEMfv (face value for direct trade disruptions)"
        elif shock_type == 'value_added':
            return "VA decomposition (for value-added shocks)"
        elif shock_type == 'technology':
            if IndustryCharacteristic.TECH_INTENSIVE in classification.characteristics:
                return "Technology flow indicators (for tech-intensive sectors)"
            else:
                return "FPEM (standard look-through)"
        else:
            return "FPEM (default)"


# Example usage and extension point
def create_custom_classification_from_data(
    data_path: str,
    engine: HorsesClassificationEngine
) -> None:
    """
    Load custom classifications from external data file.

    This function demonstrates how users can extend the classification
    system with their own data sources (e.g., from proprietary industry
    analysis, updated OECD indicators, etc.).

    Parameters
    ----------
    data_path : str
        Path to CSV file with columns: industry_code, industry_name,
        characteristics (comma-separated), priority, notes.
    engine : HorsesClassificationEngine
        Engine to populate with classifications.

    Examples
    --------
    >>> engine = HorsesClassificationEngine()
    >>> create_custom_classification_from_data('my_classifications.csv', engine)
    """
    df = pd.read_csv(data_path)

    for _, row in df.iterrows():
        characteristics = [c.strip() for c in row['characteristics'].split(',')]
        engine.add_classification(
            industry_code=row['industry_code'],
            industry_name=row['industry_name'],
            characteristics=characteristics,
            priority=row['priority'],
            notes=row.get('notes', '')
        )

    logger.info(f"Loaded custom classifications from {data_path}")
