import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# -----------------------
# Data Loading and Filtering
# -----------------------
link_to_df = r"https://raw.githubusercontent.com/thedatahawk/datasets/refs/heads/main/us_sourcing.csv"
df = pd.read_csv(link_to_df, dtype={"year": str})

# Filter out rows where all three exposure metrics are zero and where selling_country is 'Foreign'
df_mask = ((df['FPEM'] == 0) & (df['FPEMfv'] == 0) & (df['FPEMhe'] == 0))
df = df.loc[~df_mask]
df = df.loc[df['selling_country'] != 'Foreign']

# Get unique dropdown values
selling_countries = df['selling_country'].unique()
selling_industries = df['selling_industry_name'].unique()
sourcing_industries = df['sourcing_industry_name'].unique()
years = df['year'].unique()

value_options = [
    {'label': 'Foreign Production Exposure: Look Through', 'value': 'FPEM'},
    {'label': 'Foreign Production Exposure: Face Value', 'value': 'FPEMfv'},
    {'label': 'Foreign Production Exposure: Hidden', 'value': 'FPEMhe'}
]

# -----------------------
# Color Mappings
# -----------------------
industry_color_mapping = {
    "Agri & Forestry": "#1f77b4",
    "Fishing & Aqua": "#ff7f0e",
    "Energy Mining": "#2ca02c",
    "Non-Energy Mining": "#d62728",
    "Mining Support": "#9467bd",
    "Food": "#8c564b",
    "Clothes": "#e377c2",
    "Wood": "#7f7f7f",
    "Paper": "#bcbd22",
    "Refin. Petrol": "#17becf",
    "Chemicals": "#aec7e8",
    "Pharma": "#ffbb78",
    "Plastics": "#98df8a",
    "Non-Metal Gds": "#ff9896",
    "Basic Metals": "#c5b0d5",
    "Fab Metal Gds": "#c49c94",
    "Electronics": "#f7b6d2",
    "Elec. Eq.": "#c7c7c7",
    "Machinery": "#dbdb8d",
    "Vehicles": "#9edae5",
    "Transp. Eq.": "#393b79",
    "Manuf.": "#637939",
    "Utilities": "#8c6d31",
    "Water & Waste": "#843c39",
    "Construction": "#7b4173",
    "Trade & Repair": "#3182bd",
    "Land Transport": "#e6550d",
    "Water Transport": "#31a354",
    "Air Transport": "#756bb1",
    "Warehousing": "#636363",
    "Postal & Courier": "#fdae6b",
    "Accom. & Food": "#9c9ede",
    "Media": "#fdd0a2",
    "Telecoms": "#31a1c4",
    "IT Services": "#74c476",
    "Finance & Insurance": "#ffeda0",
    "Real Estate": "#feb24c",
    "Prof. Services": "#ff5f5f",
    "Admin Services": "#66c2a5",
    "Public Admin": "#fc8d62",
    "Education": "#8da0cb",
    "Health & Social": "#fdd0a2",
    "Arts & Rec.": "#fdd0a2",
    "Other Services": "#33a02c",
    "Household Activities": "#ff7f7f"
}

country_color_mapping = {
    "Argentina": "#e41a1c",
    "Australia": "#377eb8",
    "Austria": "#4daf4a",
    "Belgium": "#984ea3",
    "Bangladesh": "#ff7f00",
    "Bulgaria": "#ffff33",
    "Belarus": "#a65628",
    "Brazil": "#f781bf",
    "Brunei Darussalam": "#999999",
    "Canada": "#66c2a5",
    "Switzerland": "#fc8d62",
    "Chile": "#8da0cb",
    "China": "#e78ac3",
    "Côte d'Ivoire": "#a6d854",
    "Cameroon": "#ffd92f",
    "Colombia": "#e5c494",
    "Costa Rica": "#b3b3b3",
    "Cyprus": "#1b9e77",
    "Czechia": "#d95f02",
    "Germany": "#7570b3",
    "Denmark": "#e7298a",
    "Egypt": "#66a61e",
    "Spain": "#e6ab02",
    "Estonia": "#a6761d",
    "Finland": "#666666",
    "France": "#1f78b4",
    "United Kingdom": "#b2df8a",
    "Greece": "#33a02c",
    "Hong Kong, China": "#fb9a99",
    "Croatia": "#e31a1c",
    "Hungary": "#fdbf6f",
    "Indonesia": "#cab2d6",
    "India": "#6a3d9a",
    "Ireland": "#ffff99",
    "Iceland": "#b15928",
    "Israel": "#8dd3c7",
    "Italy": "#bebada",
    "Jordan": "#fb8072",
    "Japan": "#80b1d3",
    "Kazakhstan": "#fdb462",
    "Cambodia": "#b3de69",
    "Korea": "#fccde5",
    "Lao (People's Democratic Republic)": "#d9d9d9",
    "Lithuania": "#bc80bd",
    "Luxembourg": "#ccebc5",
    "Latvia": "#ffed6f",
    "Morocco": "#1a9641",
    "Mexico": "#a1dab4",
    "Malta": "#4575b4",
    "Myanmar": "#91bfdb",
    "Malaysia": "#e0f3f8",
    "Nigeria": "#fee090",
    "Netherlands": "#fc8d59",
    "Norway": "#d73027",
    "New Zealand": "#313695",
    "Pakistan": "#74add1",
    "Peru": "#fdae61",
    "Philippines": "#abd9e9",
    "Poland": "#2c7bb6",
    "Portugal": "#d7191c",
    "Romania": "#f46d43",
    "Rest of World": "#393b79",
    "Russian Federation": "#637939",
    "Saudi Arabia": "#8c6d31",
    "Senegal": "#3288bd",
    "Singapore": "#fee08b",
    "Slovakia": "#d8b365",
    "Slovenia": "#5e3c99",
    "Sweden": "#80cdc1",
    "Thailand": "#f4a582",
    "Tunisia": "#b2abd2",
    "Türkiye": "#8073ac",
    "Chinese Taipei": "#e7d4e8",
    "Ukraine": "#d9ef8b",
    "United States": "#1f77b4",  # Highlighted blue
    "Viet Nam": "#e7a1a1",
    "South Africa": "#a1d99b"
}

# -----------------------
# Helper Functions
# -----------------------
def build_treemap_data(df, metric, ordering, industry_map, country_map):
    """
    Build lists for ids, labels, parents, values, and colors for a two-level treemap.
    No artificial root node is added so that each parent node (with no parent) appears at top level.
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

    # Build tree: each parent becomes a top-level node (parent=""), and its children are added.
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

def build_treemap_figure(ids, labels, parents, values, colors, title):
    """
    Build a Plotly Graph Objects treemap figure using provided data lists.
    """
    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues="total",  # Ensure parent's value equals sum of children
        textinfo="label+value+percent parent"
    ))
    fig.update_layout(title=title)
    return fig

# -----------------------
# Dash App Setup
# -----------------------
app = dash.Dash(__name__)
server = app.server

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
            "Note: the authors used manufacturing-to-manufacturing exposure, but this tool covers all goods and services. For manufacturing-to-manufacturing exposure by country, see ",
            html.A(
                "World Source",
                href="https://world-source.onrender.com/",
                target="_blank", rel="noopener noreferrer"
            ),
            "."
        ]),
        html.P("This tool displays three estimates of U.S. production exposure in goods and services:"),
        html.Ul([
            html.Li("Face Value"),
            html.Li("Look Through"),
            html.Li("Hidden")
        ])
    ]),




    html.Div([
        html.Label("Select Sourcing Industry:"),
        dcc.Dropdown(
            id='sourcing-industry',
            options=[{'label': j, 'value': j} for j in sourcing_industries],
            value='Vehicles',
            clearable=False,
            style={'width': '400px'}
        )
    ], style={'display': 'inline-block', 'margin-right': '20px'}),
    
    html.Div([
        html.Label("Select Exposure Metric:"),
        dcc.Dropdown(
            id='value-metric',
            options=value_options,
            value='FPEM',
            clearable=False,
            style={'width': '400px'}
        )
    ], style={'display': 'inline-block', 'margin-right': '20px'}),
    
    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-metric',
            options=[{'label': y, 'value': y} for y in years],
            value='2020',
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
    
    html.Div(id='latest-value-text', style={'whiteSpace': 'pre-line'})
])

@app.callback(
    [Output('treemap-chart', 'figure'),
     Output('latest-value-text', 'children')],
    [Input('sourcing-industry', 'value'),
     Input('value-metric', 'value'),
     Input('year-metric', 'value'),
     Input('path-order', 'value')]
)
def update_graph(selected_sourcing_industry, selected_metric, selected_year, selected_path_order):
    # Filter data based on selections
    filtered_df = df.loc[
        (df['sourcing_industry_name'] == selected_sourcing_industry) &
        (df['year'] == selected_year)
    ]
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title_text="No data available")
        return fig, "No data available"
    
    # Build treemap data
    ids, labels, parents, values, colors = build_treemap_data(
        filtered_df, selected_metric, selected_path_order,
        industry_color_mapping, country_color_mapping
    )
    
    title = f"{selected_sourcing_industry} exposure ({selected_metric}) for the year {selected_year}"
    fig = build_treemap_figure(ids, labels, parents, values, colors, title)
    
    # Retrieve a latest value (for example, from the last row)
    latest_value = filtered_df[selected_metric].iloc[-1]
    latest_text = f"Latest value for {selected_metric}: {latest_value}"
    
    return fig, latest_text

if __name__ == '__main__':
    app.run(debug=True)
