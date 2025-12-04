"""
U.S. Manufacturing Exposure Dashboard

Interactive Dash application for visualizing U.S. production exposure by country
and industry using OECD ICIO data. Creates treemap visualizations showing exposure
metrics across different industries and countries.

Based on Baldwin, Freeman & Theodorakopoulos (2022, 2023):
- Hidden Exposure: Measuring US Supply Chain Reliance (NBER w31820)
- Horses for Courses: Measuring Foreign Supply Chain Exposure (NBER w30525)

Usage:
    python us_source_app.py

Author: Refactored for modular OECD ICIO Analysis Package
Date: 2025
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import config
from utils.logging_config import setup_logger

logger = setup_logger(__name__, level=config.LOG_LEVEL)

# -----------------------
# Configuration
# -----------------------
DATA_PATH = config.OUTPUT_FOLDER / "usa_sourcing.csv"
DEFAULT_YEAR = "2022"
DEFAULT_INDUSTRY = "Motor vehicles"
DEFAULT_METRIC = "FPEM"

# -----------------------
# Color Mappings
# -----------------------
INDUSTRY_COLOR_MAPPING = {
    # Agriculture & Natural Resources
    "Agriculture": "#1f77b4",
    "Forestry": "#2ca02c",
    "Fishing": "#ff7f0e",

    # Mining & Extraction
    "Coal mining": "#d62728",
    "Oil & gas": "#9467bd",
    "Metal ore mining": "#8c564b",
    "Other mining": "#e377c2",
    "Mining support": "#7f7f7f",

    # Manufacturing
    "Food & beverages": "#bcbd22",
    "Textiles & apparel": "#17becf",
    "Wood products": "#aec7e8",
    "Paper & printing": "#ffbb78",
    "Petroleum products": "#98df8a",
    "Chemicals": "#ff9896",
    "Pharmaceuticals": "#c5b0d5",
    "Rubber & plastics": "#c49c94",
    "Non-metallic minerals": "#f7b6d2",
    "Basic iron & steel": "#c7c7c7",
    "Non-ferrous metals": "#dbdb8d",
    "Fabricated metals": "#9edae5",
    "Electronics": "#393b79",
    "Electrical equipment": "#637939",
    "Machinery": "#8c6d31",
    "Motor vehicles": "#843c39",
    "Shipbuilding": "#7b4173",
    "Other transport equip.": "#3182bd",
    "Furniture & other manuf.": "#e6550d",

    # Utilities & Construction
    "Electricity & utilities": "#31a354",
    "Water & waste": "#756bb1",
    "Construction": "#636363",

    # Services
    "Retail & vehicle repair": "#fdae6b",
    "Land transport": "#9c9ede",
    "Water transport": "#fdd0a2",
    "Air transport": "#31a1c4",
    "Warehousing & support": "#74c476",
    "Postal & courier": "#ffeda0",
    "Accommodation & food": "#feb24c",
    "Media & publishing": "#ff5f5f",
    "Telecom": "#66c2a5",
    "IT services": "#fc8d62",
    "Finance & insurance": "#8da0cb",
    "Real estate": "#b3b3b3",
    "Professional services": "#1b9e77",
    "Admin & support": "#d95f02",
    "Public administration": "#7570b3",
    "Education": "#e7298a",
    "Health & social work": "#66a61e",
    "Arts & recreation": "#e6ab02",
    "Other services": "#a6761d",
    "Household activities": "#666666"
}

COUNTRY_COLOR_MAPPING = {
    # Major Economies (distinct colors)
    "United States": "#1f77b4",
    "China, People's Republic of": "#e78ac3",
    "Germany": "#7570b3",
    "Japan": "#80b1d3",
    "United Kingdom": "#b2df8a",
    "France": "#ff7f0e",
    "India": "#6a3d9a",
    "Canada": "#66c2a5",
    "Italy": "#bebada",
    "Korea": "#fccde5",
    "Rest of the World": "#393b79",

    # Europe
    "Austria": "#4daf4a",
    "Belgium": "#984ea3",
    "Bulgaria": "#ffff33",
    "Croatia": "#e31a1c",
    "Cyprus": "#1b9e77",
    "Czechia": "#d95f02",
    "Denmark": "#e7298a",
    "Estonia": "#a6761d",
    "Finland": "#666666",
    "Greece": "#33a02c",
    "Hungary": "#fdbf6f",
    "Iceland": "#b15928",
    "Ireland": "#ffff99",
    "Lithuania": "#bc80bd",
    "Luxembourg": "#ccebc5",
    "Malta": "#4575b4",
    "Netherlands": "#fc8d59",
    "Norway": "#d73027",
    "Poland": "#2c7bb6",
    "Portugal": "#d7191c",
    "Romania": "#f46d43",
    "Slovak Republic": "#d8b365",
    "Slovenia": "#5e3c99",
    "Spain": "#e6ab02",
    "Sweden": "#80cdc1",
    "Switzerland": "#fc8d62",
    "Belarus": "#a65628",
    "Ukraine": "#d9ef8b",
    "Russian Federation": "#637939",
    "Türkiye": "#8073ac",

    # Asia
    "Bangladesh": "#ff7f00",
    "Brunei Darussalam": "#999999",
    "Cambodia": "#b3de69",
    "Hong Kong, China": "#fb9a99",
    "Indonesia": "#cab2d6",
    "Israel": "#8dd3c7",
    "Jordan": "#fb8072",
    "Kazakhstan": "#fdb462",
    "Lao (People's Democratic Republic)": "#d9d9d9",
    "Malaysia": "#e0f3f8",
    "Myanmar": "#91bfdb",
    "Pakistan": "#74add1",
    "Philippines": "#abd9e9",
    "Singapore": "#fee08b",
    "Chinese Taipei": "#e7d4e8",
    "Thailand": "#f4a582",
    "Viet Nam": "#e7a1a1",
    "United Arab Emirates": "#8dd3c7",
    "Saudi Arabia": "#8c6d31",

    # Americas
    "Argentina": "#e41a1c",
    "Brazil": "#f781bf",
    "Chile": "#8da0cb",
    "Colombia": "#e5c494",
    "Costa Rica": "#b3b3b3",
    "Mexico": "#a1dab4",
    "Peru": "#fdae61",

    # Africa
    "Angola": "#fee090",
    "Cameroon": "#ffd92f",
    "Côte d'Ivoire": "#a6d854",
    "C�te d'Ivoire": "#a6d854",
    "Democratic Republic of the Congo": "#f781bf",
    "Egypt": "#66a61e",
    "Morocco": "#1a9641",
    "Nigeria": "#fee090",
    "São Tomé and Príncipe": "#b3de69",
    "Sa� Tom� and Pr�ncipe": "#b3de69",
    "Senegal": "#3288bd",
    "South Africa": "#a1d99b",
    "Tunisia": "#b2abd2",

    # Oceania
    "Australia": "#377eb8",
    "New Zealand": "#313695",

    # Special
    "Foreign": "#cccccc"
}

VALUE_OPTIONS = [
    {'label': 'Foreign Production Exposure: Look Through', 'value': 'FPEM'},
    {'label': 'Foreign Production Exposure: Face Value', 'value': 'FPEMfv'},
    {'label': 'Foreign Production Exposure: Hidden', 'value': 'FPEMhe'}
]

# -----------------------
# Data Loading
# -----------------------
def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess USA sourcing data.

    Parameters
    ----------
    data_path : Path
        Path to usa_sourcing.csv file.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with year as string and filtered for valid data.
    """
    logger.info(f"Loading data from {data_path}...")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run: python main.py --years <years> --aggregated-outputs"
        )

    df = pd.read_csv(data_path, dtype={"year": str})

    # Filter out rows where all exposure metrics are zero
    df_mask = ((df['FPEM'] == 0) & (df['FPEMfv'] == 0) & (df['FPEMhe'] == 0))
    df = df.loc[~df_mask]

    # Filter out 'Foreign' selling country (keep individual countries)
    df = df.loc[df['selling_country'] != 'Foreign']

    logger.info(f"Loaded {len(df):,} rows of data")
    logger.info(f"Years: {sorted(df['year'].unique())}")
    logger.info(f"Industries: {df['sourcing_industry_name'].nunique()}")
    logger.info(f"Countries: {df['selling_country'].nunique()}")

    return df

# -----------------------
# Helper Functions
# -----------------------
def build_treemap_data(
    df: pd.DataFrame,
    metric: str,
    ordering: str,
    industry_map: dict,
    country_map: dict
) -> tuple:
    """
    Build hierarchical data structure for treemap visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame for the treemap.
    metric : str
        Exposure metric to visualize (FPEM, FPEMfv, FPEMhe).
    ordering : str
        Hierarchy ordering: 'industry-country' or 'country-industry'.
    industry_map : dict
        Industry to color mapping.
    country_map : dict
        Country to color mapping.

    Returns
    -------
    tuple
        (ids, labels, parents, values, colors) for Plotly treemap.
    """
    if ordering == 'industry-country':
        parent_col = 'selling_industry_name'
        child_col = 'selling_country'
        parent_color_map = industry_map
        child_color_map = country_map
    else:  # 'country-industry'
        parent_col = 'selling_country'
        child_col = 'selling_industry_name'
        parent_color_map = country_map
        child_color_map = industry_map

    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    # Build two-level hierarchy: parent nodes at top level, children below
    for parent_val, parent_group in df.groupby(parent_col):
        parent_id = str(parent_val)
        parent_value = parent_group[metric].sum()
        parent_color = parent_color_map.get(parent_val, "#CCCCCC")

        ids.append(parent_id)
        labels.append(parent_val)
        parents.append("")
        values.append(parent_value)
        colors.append(parent_color)

        for child_val, child_group in parent_group.groupby(child_col):
            child_id = f"{parent_id}-{child_val}"
            child_value = child_group[metric].sum()
            child_color = child_color_map.get(child_val, "#DDDDDD")

            ids.append(child_id)
            labels.append(child_val)
            parents.append(parent_id)
            values.append(child_value)
            colors.append(child_color)

    return ids, labels, parents, values, colors

def build_treemap_figure(
    ids: list,
    labels: list,
    parents: list,
    values: list,
    colors: list,
    title: str
) -> go.Figure:
    """
    Create Plotly treemap figure.

    Parameters
    ----------
    ids : list
        Node identifiers.
    labels : list
        Node labels.
    parents : list
        Parent node identifiers.
    values : list
        Node values.
    colors : list
        Node colors.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly treemap figure.
    """
    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues="total",
        textinfo="label+value+percent parent"
    ))
    fig.update_layout(title=title)
    return fig

# -----------------------
# Dash App Setup
# -----------------------
def create_app(df: pd.DataFrame) -> dash.Dash:
    """
    Create and configure Dash application.

    Parameters
    ----------
    df : pd.DataFrame
        USA sourcing data.

    Returns
    -------
    dash.Dash
        Configured Dash app instance.
    """
    app = dash.Dash(__name__)
    app.title = "U.S. Manufacturing Exposure"

    # Get unique values for dropdowns
    sourcing_industries = sorted(df['sourcing_industry_name'].unique())
    years = sorted(df['year'].unique())

    app.layout = html.Div([
        html.H1("U.S. Manufacturing Exposure by Country and Industry"),

        html.Div([
            html.P([
                "This data tool is based on the work of Richard Baldwin, Rebecca Freeman & Angelos Theodorakopoulos as described in ",
                html.A(
                    "Hidden Exposure: Measuring US Supply Chain Reliance",
                    href="https://www.nber.org/papers/w31820",
                    target="_blank", rel="noopener noreferrer"
                ),
                " and ",
                html.A(
                    "Horses for Courses: Measuring Foreign Supply Chain Exposure",
                    href="https://www.nber.org/papers/w30525",
                    target="_blank", rel="noopener noreferrer"
                ),
                "."
            ]),
            html.P([
                "I am not affiliated in any way with the authors; any errors in the data tool are my own. Source data are from ",
                html.A(
                    "OECD Trade in Value-Added.",
                    href="https://www.oecd.org/en/topics/sub-issues/trade-in-value-added.html",
                    target="_blank", rel="noopener noreferrer"
                ),
                "."
            ]),
            html.P([
                "Note: the authors used manufacturing-to-manufacturing exposure, but this tool covers all goods and services. "
                "For manufacturing-to-manufacturing exposure by country, see the World Source dashboard."
            ]),
            html.P("This tool displays three estimates of U.S. production exposure in goods and services:"),
            html.Ul([
                html.Li("Face Value: Direct imports from each country"),
                html.Li("Look Through: Total foreign value-added (direct + indirect)"),
                html.Li("Hidden: Indirect exposure through third countries")
            ])
        ]),

        html.Div([
            html.Label("Select Sourcing Industry:"),
            dcc.Dropdown(
                id='sourcing-industry',
                options=[{'label': j, 'value': j} for j in sourcing_industries],
                value=DEFAULT_INDUSTRY if DEFAULT_INDUSTRY in sourcing_industries else sourcing_industries[0],
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        html.Div([
            html.Label("Select Exposure Metric:"),
            dcc.Dropdown(
                id='value-metric',
                options=VALUE_OPTIONS,
                value=DEFAULT_METRIC,
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        html.Div([
            html.Label("Select Year:"),
            dcc.Dropdown(
                id='year-metric',
                options=[{'label': y, 'value': y} for y in years],
                value=DEFAULT_YEAR if DEFAULT_YEAR in years else years[-1],
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        html.Div([
            html.Label("Select Treemap Path Order:"),
            dcc.Dropdown(
                id='path-order',
                options=[
                    {'label': 'Industry → Country', 'value': 'industry-country'},
                    {'label': 'Country → Industry', 'value': 'country-industry'}
                ],
                value='industry-country',
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        dcc.Graph(
            id='treemap-chart',
            style={'height': '800px', 'width': '100%'}
        ),

        html.Div(id='summary-text', style={'whiteSpace': 'pre-line', 'margin-top': '20px'})
    ])

    @app.callback(
        [Output('treemap-chart', 'figure'),
         Output('summary-text', 'children')],
        [Input('sourcing-industry', 'value'),
         Input('value-metric', 'value'),
         Input('year-metric', 'value'),
         Input('path-order', 'value')]
    )
    def update_graph(selected_sourcing_industry, selected_metric, selected_year, selected_path_order):
        """Update treemap and summary text based on user selections."""
        # Filter data
        filtered_df = df.loc[
            (df['sourcing_industry_name'] == selected_sourcing_industry) &
            (df['year'] == selected_year)
        ]

        if filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data available")
            return fig, "No data available for selected filters."

        # Filter out rows where the selected metric is zero or very small
        filtered_df = filtered_df[filtered_df[selected_metric] > 0.0001].copy()

        if filtered_df.empty or filtered_df[selected_metric].sum() < 0.001:
            fig = go.Figure()
            fig.update_layout(title_text=f"No significant {selected_metric} data for {selected_sourcing_industry} in {selected_year}")
            return fig, f"All {selected_metric} values are near zero or below threshold (0.0001%) for this selection."

        # Build treemap
        ids, labels, parents, values, colors = build_treemap_data(
            filtered_df, selected_metric, selected_path_order,
            INDUSTRY_COLOR_MAPPING, COUNTRY_COLOR_MAPPING
        )

        metric_labels = {
            'FPEM': 'Look Through',
            'FPEMfv': 'Face Value',
            'FPEMhe': 'Hidden'
        }
        title = f"{selected_sourcing_industry} - {metric_labels.get(selected_metric, selected_metric)} Exposure ({selected_year})"
        fig = build_treemap_figure(ids, labels, parents, values, colors, title)

        # Generate summary text
        total_exposure = filtered_df[selected_metric].sum()
        top_countries = filtered_df.groupby('selling_country')[selected_metric].sum().nlargest(5)

        summary = f"Total {selected_metric} exposure: {total_exposure:.2f}%\n\n"
        summary += f"Top 5 selling countries:\n"
        for i, (country, value) in enumerate(top_countries.items(), 1):
            summary += f"  {i}. {country}: {value:.2f}%\n"

        return fig, summary

    return app

# -----------------------
# Main Execution
# -----------------------
def main():
    """Main execution function."""
    try:
        df = load_data(DATA_PATH)
        app = create_app(df)

        logger.info("Starting U.S. Source Dashboard...")
        logger.info("Access the dashboard at: http://127.0.0.1:8050")

        app.run(debug=True)

    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
