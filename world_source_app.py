"""
World Manufacturing Exposure Dashboard

Interactive Dash application for visualizing manufacturing production exposure
by country and industry using OECD ICIO data. Creates line charts showing
exposure trends over time.

Based on Baldwin, Freeman & Theodorakopoulos (2022, 2023):
- Hidden Exposure: Measuring US Supply Chain Reliance (NBER w31820)
- Horses for Courses: Measuring Foreign Supply Chain Exposure (NBER w30525)

Usage:
    python world_source_app.py

Author: Refactored for modular OECD ICIO Analysis Package
Date: 2025
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import config
from utils.logging_config import setup_logger

logger = setup_logger(__name__, level=config.LOG_LEVEL)

# -----------------------
# Configuration
# -----------------------
DATA_PATH = config.OUTPUT_FOLDER / "world_sourcing.csv"
DEFAULT_SOURCING_COUNTRY = "United States"
DEFAULT_SELLING_COUNTRY = "United States"
DEFAULT_INDUSTRY = "Manuf. avg."
DEFAULT_METRIC = "FPEM"

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
    Load and preprocess world manufacturing sourcing data.

    Parameters
    ----------
    data_path : Path
        Path to world_sourcing.csv file.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with year as string.
    """
    logger.info(f"Loading data from {data_path}...")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run: python main.py --years <years> --aggregated-outputs"
        )

    # Only load columns we need for the dashboard
    columns_to_load = [
        'sourcing_country', 'selling_country', 'sourcing_industry_name',
        'year', 'FPEM', 'FPEMfv', 'FPEMhe'
    ]

    logger.info("Reading CSV file (this may take a minute)...")

    df = pd.read_csv(
        data_path,
        usecols=columns_to_load,
        low_memory=False
    )

    # Drop rows with missing critical data
    df = df.dropna(subset=['sourcing_country', 'selling_country', 'sourcing_industry_name'])

    # Filter out rows with "Manuf. avg." if they have invalid data
    # (temporary workaround until data is regenerated)
    manuf_avg = df[df['sourcing_industry_name'] == 'Manuf. avg.']
    if len(manuf_avg) > 0:
        # Check if selling_country contains numeric values (data corruption)
        try:
            pd.to_numeric(manuf_avg['selling_country'].iloc[0])
            logger.warning("Removing corrupted 'Manuf. avg.' data - please regenerate aggregated outputs")
            df = df[df['sourcing_industry_name'] != 'Manuf. avg.']
        except (ValueError, TypeError):
            # Data is fine, keep it
            pass

    # Convert to optimized dtypes to reduce memory usage
    df['sourcing_country'] = df['sourcing_country'].astype('category')
    df['selling_country'] = df['selling_country'].astype('category')
    df['sourcing_industry_name'] = df['sourcing_industry_name'].astype('category')
    df['year'] = df['year'].astype(str).astype('category')

    # Convert numeric columns, handling any non-numeric values
    for col in ['FPEM', 'FPEMfv', 'FPEMhe']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

    logger.info(f"Loaded {len(df):,} rows of data")
    logger.info(f"Years: {sorted(df['year'].unique())}")
    logger.info(f"Industries: {df['sourcing_industry_name'].nunique()}")
    logger.info(f"Sourcing countries: {df['sourcing_country'].nunique()}")
    logger.info(f"Selling countries: {df['selling_country'].nunique()}")

    return df

# -----------------------
# Dash App Setup
# -----------------------
def create_app(df: pd.DataFrame) -> dash.Dash:
    """
    Create and configure Dash application.

    Parameters
    ----------
    df : pd.DataFrame
        World manufacturing sourcing data.

    Returns
    -------
    dash.Dash
        Configured Dash app instance.
    """
    app = dash.Dash(__name__)
    app.title = "World Manufacturing Exposure"

    # Get unique values for dropdowns
    sourcing_countries = sorted(df['sourcing_country'].unique())
    selling_countries = sorted(df['selling_country'].unique())
    industries = sorted(df['sourcing_industry_name'].unique())

    app.layout = html.Div([
        html.H1("Manufacturing Exposure by Country and Industry"),

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
            html.P("This tool displays three estimates of production exposure in manufacturing:"),
            html.Ul([
                html.Li("Face Value: Direct imports from each country"),
                html.Li("Look Through: Total foreign value-added (direct + indirect)"),
                html.Li("Hidden: Indirect exposure through third countries")
            ])
        ]),

        html.Div([
            html.Label("Select Sourcing Country:"),
            dcc.Dropdown(
                id='sourcing-country',
                options=[{'label': s, 'value': s} for s in sourcing_countries],
                value=DEFAULT_SOURCING_COUNTRY if DEFAULT_SOURCING_COUNTRY in sourcing_countries else sourcing_countries[0],
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        html.Div([
            html.Label("Select Selling Country:"),
            dcc.Dropdown(
                id='selling-country',
                options=[{'label': c, 'value': c} for c in selling_countries],
                value=DEFAULT_SELLING_COUNTRY if DEFAULT_SELLING_COUNTRY in selling_countries else selling_countries[0],
                clearable=False,
                style={'width': '400px'}
            )
        ], style={'display': 'inline-block'}),

        html.Br(),
        html.Br(),

        html.Div([
            html.Label("Select Industry:"),
            dcc.Dropdown(
                id='industry',
                options=[{'label': i, 'value': i} for i in industries],
                value=DEFAULT_INDUSTRY if DEFAULT_INDUSTRY in industries else industries[0],
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
        ], style={'display': 'inline-block'}),

        dcc.Loading(
            id="loading",
            type="default",
            children=[
                dcc.Graph(
                    id='line-chart',
                    style={'height': '600px', 'width': '100%'}
                )
            ]
        ),

        html.Div(id='summary-text', style={'whiteSpace': 'pre-line', 'margin-top': '20px'})
    ])

    @app.callback(
        [Output('line-chart', 'figure'),
         Output('summary-text', 'children')],
        [Input('sourcing-country', 'value'),
         Input('selling-country', 'value'),
         Input('industry', 'value'),
         Input('value-metric', 'value')]
    )
    def update_graph(selected_sourcing_country, selected_selling_country, selected_industry, selected_metric):
        """Update line chart and summary text based on user selections."""
        # Filter data for the selected combination using query for better performance
        try:
            filtered_df = df.query(
                'sourcing_country == @selected_sourcing_country & '
                'selling_country == @selected_selling_country & '
                'sourcing_industry_name == @selected_industry'
            ).copy()
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            fig = px.line(title="Error filtering data")
            return fig, f"Error filtering data: {e}"

        if filtered_df.empty:
            fig = px.line(title="No data available for selected filters")
            return fig, "No data available for selected filters."

        # Create line chart
        metric_labels = {
            'FPEM': 'Look Through',
            'FPEMfv': 'Face Value',
            'FPEMhe': 'Hidden'
        }

        title = (
            f"{selected_sourcing_country} {metric_labels.get(selected_metric, selected_metric)} "
            f"exposure from {selected_selling_country} in {selected_industry}"
        )

        fig = px.line(
            filtered_df,
            x='year',
            y=selected_metric,
            title=title,
            labels={
                'year': 'Year',
                selected_metric: f'{metric_labels.get(selected_metric, selected_metric)} (%)'
            }
        )

        fig.update_traces(mode='lines+markers', line=dict(width=4), marker=dict(size=15))
        fig.update_xaxes(type='category')

        # Generate summary text for domestic exposure
        try:
            domestic_df = df.query(
                'sourcing_country == @selected_sourcing_country & '
                'selling_country == @selected_sourcing_country & '
                'sourcing_industry_name == @selected_industry'
            )
        except Exception:
            domestic_df = pd.DataFrame()

        if not domestic_df.empty:
            sorted_df = domestic_df.sort_values(by='year', ascending=True)
            first_year = sorted_df.iloc[0]['year']
            first_value = round(sorted_df.iloc[0][selected_metric], 2)
            latest_year = sorted_df.iloc[-1]['year']
            latest_value = round(sorted_df.iloc[-1][selected_metric], 2)
            change_value = round(latest_value - first_value, 2)

            summary = (
                f"For {selected_sourcing_country} in {selected_industry} using {metric_labels.get(selected_metric, selected_metric)}:\n"
                f"  - {first_year}: {first_value}%\n"
                f"  - {latest_year}: {latest_value}%\n"
                f"  - Percentage point change: {change_value:+.2f}%"
            )
        else:
            summary = "No domestic exposure data available for comparison."

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

        logger.info("Starting World Source Dashboard...")
        logger.info("Access the dashboard at: http://127.0.0.1:8050")

        app.run(debug=True)

    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
