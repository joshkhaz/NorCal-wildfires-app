import ast
import dash
import dash_bootstrap_components as dbc
import datetime
import gzip
import json
import math
import numpy as np
import os
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import sklearn.metrics
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_shap_components import ForcePlot



DEFAULT_CONFIDENCE = 41
MAP_CENTER = (39.926535, -121.629275)
curdir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(
    os.path.join(curdir, 'modeling-2_historic_and_predictions.csv.gz'),
    usecols=['latitude', 'longitude', 'date', 'type', 'Target', 'Pred'],
    parse_dates=['date'],
    dtype={'type': 'category'},
)
df_fids = pd.read_csv(
    os.path.join(curdir, 'modeling-2_fire_ids.csv.gz'),
    usecols=[
        'wildfire_id', 'date', 'lat_min', 'lat_max', 'long_min', 'long_max',
        'county'
    ],
    parse_dates=['date'],
)
df_fids['county'] = df_fids['county'].apply(ast.literal_eval)
df_fids = df_fids.sort_values(['date'])
with open('modeling-2_featureNames.txt') as f:
    featureNames = f.read().splitlines()
featureNames = {i:featureNames[i] for i in range(len(featureNames))}

square_side_miles = 1
miles_per_degree = 24901.461 / 360 # circumference of earth at equator divided by 360
square_side_degrees = square_side_miles / miles_per_degree
square_radius_deg = square_side_degrees / 2 - 0.000085

# Sort the fire coordinates so that it forms a box
def sort_fire(fire):
    start = fire[0]
    def sort(pt):
        return math.atan2(start[0] - pt[0], start[1] - pt[1])
    fire.sort(key=sort)

    return fire

def get_data(date, conf_thresh):
    df_curr = df[(df['date'] == date) & (df['type'] == 'Historic')]
    df_pred = df[
        (df['date'] == date) &
        (df['type'] == 'Prediction') &
        (df['Pred'] >= conf_thresh)
    ]
    lats_curr, lons_curr = [], []
    for row in df_curr.itertuples():
        lats_curr.append(row.latitude)
        lons_curr.append(row.longitude)
    lats_pred, lons_pred, conf_pred, target_pred, shap_pred = [], [], [], [], []
    for row in df_pred.itertuples():
        lat, lon, conf, target, shap = row.latitude, row.longitude, row.Pred, row.Target, '{}_{}_{}'.format(row.date.strftime('%Y-%m-%d'), round(row.latitude,3), round(row.longitude,3))
        lats_pred.append(lat + square_radius_deg)
        lons_pred.append(lon + square_radius_deg)
        lats_pred.append(lat + square_radius_deg)
        lons_pred.append(lon - square_radius_deg)
        lats_pred.append(lat - square_radius_deg)
        lons_pred.append(lon - square_radius_deg)
        lats_pred.append(lat - square_radius_deg)
        lons_pred.append(lon + square_radius_deg)
        lats_pred.append(lat + square_radius_deg)
        lons_pred.append(lon + square_radius_deg)
        conf_pred.extend([conf] * 5)
        shap_pred.extend([shap] * 5)
        target_pred.extend([target] * 5)
        lats_pred.append(None)
        lons_pred.append(None)
        conf_pred.append(None)
        shap_pred.append(None)
        target_pred.append(None)
    return (lats_curr, lons_curr), (lats_pred, lons_pred, conf_pred, target_pred, shap_pred)

def get_hist_data(start_date, end_date):
    df_curr = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['type'] == 'Historic')
    ]

    lats_curr, lons_curr = [], []
    for row in df_curr.itertuples():
        lats_curr.append(row.latitude)
        lons_curr.append(row.longitude)

    return (lats_curr, lons_curr), ()

def get_eval_data(start_date, end_date, confidence):
    df_eval = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['type'] == 'Prediction')
    ]
    y_pred = df_eval['Pred'].apply(lambda x: 1.0 if x >= confidence else 0.0)

    return df_eval['Target'], y_pred

def get_fire_ids(start_date, end_date):
    df_fids_curr = df_fids[
        (df_fids['date'] >= start_date) &
        (df_fids['date'] <= end_date)
    ].copy()
    df_fids_curr['size'] = (
        (df_fids_curr['lat_max'] - df_fids_curr['lat_min']) *
        (df_fids_curr['long_max'] - df_fids_curr['long_min'])
    )
    df_fids_curr = df_fids_curr.sort_values(['size'], ascending=False)

    ids = {}
    for row in df_fids_curr.itertuples():
        ids[row.wildfire_id] = '/'.join(row.county) if row.county else 'Unknown'

    return ids

def get_fire_data(fire_id, start_date, end_date):
    data = df_fids[
        (df_fids['wildfire_id'] == fire_id) &
        (df_fids['date'] >= start_date) &
        (df_fids['date'] <= end_date)
    ]
    if len(data) == 1:
        return data.to_dict('records')[-1]
    return {
        'lat_min': data['lat_min'].min(),
        'lat_max': data['lat_max'].max(),
        'long_min': data['long_min'].min(),
        'long_max': data['long_max'].max(),
    }

# Use plotly to make scatter plot for dataset description.
def get_fire_map(curr, pred, fire_data):
    if fire_data:
        center = (
            (fire_data['lat_min'] + fire_data['lat_max']) / 2,
            (fire_data['long_min'] + fire_data['long_max']) / 2,
        )
        box_height = fire_data['lat_max'] - fire_data['lat_min']
        box_width = fire_data['long_max'] - fire_data['long_min']
        box_area = box_height * box_width
        zoom = np.interp(
            x=box_area,
            xp=[0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
            fp=[20, 15,    14,     13,     12,     7,      5]
        )
        zoom -= 1
    else:
        center = MAP_CENTER
        zoom = 6

    fig = go.Figure()
    if pred:
        fig.add_scattermapbox(
            mode="lines",
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            lat=pred[0],
            lon=pred[1],
            marker={'color': 'blue'},
            hovertemplate=[
                '(%{lat}°, %{lon}°)' + '<br>Confidence: {:.2f}%<br>Actual: {}<br>Click to view SHAP force plot<extra>Fire Prediction</extra>'.format(
                    f * 100, "Fire" if t == 1 else "No Fire"
                )
                if (f!=None) & (t!=None) else None for f,t in zip(pred[2], pred[3])
            ],
            customdata=pred[4],
            name='Fire Prediction (Based on confidence threshold)',
        )

    fig.add_scattermapbox(
        lat=curr[0],
        lon=curr[1],
        marker={'color': 'red'},
        name='Fire Burning',
    )

    fig.update_layout(
        mapbox={
            'style': "stamen-terrain",
            'center': {'lon': center[1], 'lat': center[0]},
            'zoom': zoom,
        },
        showlegend=True,
        legend={
            'yanchor': 'top',
            'y': 0.99,
            'xanchor': 'left',
            'x': 0.01
        },
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    )

    return fig

def get_force_plot(key):

    date = key.split('_')[0]
    try:
        with gzip.open('shap/shap_dict_{}.json.gz'.format(date)) as f:
            shap_dict_full = json.load(f)
    except FileNotFoundError:
        return html.Div(
            'Sorry, SHAP force plot is not available for that date.'
        )
    shap = shap_dict_full[key]

    return ForcePlot(
        baseValue=shap['baseValue'],
        features=shap['features'],
        featureNames=featureNames,
    )

def get_layout():
    curr, pred = get_data('2021-07-01', DEFAULT_CONFIDENCE / 100)
    return html.Div([
        html.Div([
            html.Div(html.B('Quick Start')),
            html.Div('This tool provides actual fire and predicted fire data for northern California. While fires may be detected/displayed in Nevada or Oregon predictions are not available outside northern California.'),
            html.Br(),
            html.Div('To use the tool:'),
            html.Ol([
                html.Li('Select your date.'),
                html.Li('Select your desired confidence threshold.'),
                html.Li('Select the fire you wish to zoom in on, potentially impacted counties are indicated OR explore the map by zooming in or out. Viewing predictions will require you to zoom.'),
            ]),
            html.Div([
                html.B('Date'), ' (Available Data: Jan. 2021-Jan. 2022)',
            ]),
            html.Div([
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=datetime.date(2021, 7, 1),
                ),
            ]),
            html.Br(),
            html.Div(html.B('Confidence')),
            dcc.Slider(
                min=0, max=100, value=DEFAULT_CONFIDENCE, id='conf-slider'
            ),
            html.Br(),
            html.Div(html.B('Fire # & Potentially Impacted Counties')),
            html.Div([
                dcc.Dropdown(
                    id='fire-dropdown',
                ),
            ]),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            dcc.Graph(
                id='firemap',
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                },
                figure=get_fire_map(curr, pred, None),
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            html.Div(id='force-plot-1'),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            html.Div(id='force-plot-2'),
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'column'})

def get_history_layout():
    return html.Div([
        html.Div([
            html.Div(html.B('Quick Start')),
            html.Div('This tool provides actual fire data for northern California.'),
            html.Br(),
            html.Div('To use the tool:'),
            html.Ol([
                html.Li('Select your date range.'),
                html.Li('Select the fire you wish to zoom in on, potentially impacted counties are indicated OR explore the map by zooming in or out. Viewing fires will require you to zoom.'),
            ]),
            html.Div([
                html.B('Date'), ' (Available Data: Jan. 2014-Jan. 2022)',
            ]),
            html.Div([
                dcc.DatePickerRange(id='hist-date-picker'),
            ]),
            html.Br(),
            html.Div(html.B('Fire # & Potentially Impacted Counties')),
            html.Div([
                dcc.Dropdown(id='hist-fire-dropdown'),
            ]),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            dcc.Graph(
                id='hist-firemap',
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                },
                figure=get_fire_map(([], []), None, None),
            ),
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'column'})

def get_eval_layout():
    return html.Div([
        html.Div([
            html.Div(html.B('Date')),
            html.Div([
                dcc.DatePickerRange(
                    id='eval-date-picker-range',
                ),
            ]),
            html.Br(),
            html.Div(html.B('Confidence')),
            dcc.Slider(
                min=0, max=100, value=DEFAULT_CONFIDENCE, id='eval-conf-slider'
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th('Recall'),
                        html.Th('Precision'),
                        html.Th('F1 Score'),
                    ]),
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(id='recall'),
                        html.Td(id='precision'),
                        html.Td(id='f1-score'),
                    ]),
                ]),
            ], id='eval-table', style={'display': 'none'}),
        ], style={'padding': 10, 'flex': 1}),
        html.Div([
            dcc.Graph(
                config={'displayModeBar': False},
                id='confusion-matrix',
                style={'display': 'none'},
            ),
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'column'})

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
]
dash_app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
)
dash_app.config['suppress_callback_exceptions'] = True
app = dash_app.server

@dash_app.callback(
    Output('fire-dropdown', 'options'),
    Output('fire-dropdown', 'placeholder'),
    Output('fire-dropdown', 'disabled'),
    Output('fire-dropdown', 'value'),
    Output('firemap', 'figure'),
    Input('date-picker', 'date'),
    Input('conf-slider', 'value'),
    Input('fire-dropdown', 'value'),
)
def input_changed(date, conf, fire):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    fire_value = dash.no_update
    if trigger == 'date-picker':
        fire = None
        fire_value = None

    fire_data = None
    curr, pred = get_data(str(date), conf / 100)
    fids = get_fire_ids(date, date)
    if fire:
        fire_data = get_fire_data(fire, date, date)
    options = [
        {'label': '{} ({})'.format(fid, county), 'value': fid}
        for fid, county in fids.items()
    ]
    placeholder = 'Please select a fire...'
    if not options:
        placeholder = 'No fire identifications available for selected date'
    disabled = False if options else True
    return (
        options,
        placeholder,
        disabled,
        fire_value,
        get_fire_map(curr, pred, fire_data),
    )

@dash_app.callback(
    Output('hist-fire-dropdown', 'options'),
    Output('hist-fire-dropdown', 'placeholder'),
    Output('hist-fire-dropdown', 'disabled'),
    Output('hist-fire-dropdown', 'value'),
    Output('hist-firemap', 'figure'),
    Input('hist-date-picker', 'start_date'),
    Input('hist-date-picker', 'end_date'),
    Input('hist-fire-dropdown', 'value'),
)
def hist_input_changed(start_date, end_date, fire):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    fire_value = dash.no_update
    if trigger == 'hist-date-picker':
        fire = None
        fire_value = None
    if start_date is None or end_date is None:
        raise dash.exceptions.PreventUpdate

    fire_data = None
    curr, pred = get_hist_data(str(start_date), str(end_date))
    fids = get_fire_ids(start_date, end_date)
    if fire:
        fire_data = get_fire_data(fire, start_date, end_date)
    options = [
        {'label': '{} ({})'.format(fid, county), 'value': fid}
        for fid, county in fids.items()
    ]
    placeholder = 'Please select a fire...'
    if not options:
        placeholder = 'No fire identifications available for selected date'
    disabled = False if options else True
    return (
        options,
        placeholder,
        disabled,
        fire_value,
        get_fire_map(curr, pred, fire_data),
    )

@dash_app.callback(
    Output('force-plot-1', 'children'),
    Output('force-plot-2', 'children'),
    Output('clicks', 'data'),
    Input('firemap', 'clickData'),
    State('clicks', 'data'),
    prevent_initial_call=True,
)
def map_clicked(clickData, clicks):

    if clickData is None:
        raise dash.exceptions.PreventUpdate

    try:
        key = clickData['points'][0]['customdata']
    except KeyError:
        raise dash.exceptions.PreventUpdate

    clicks += 1

    if clicks % 2 != 0:
        return get_force_plot(key), None, clicks
    else:
        return None, get_force_plot(key), clicks

@dash_app.callback(
    Output('eval-table', 'style'),
    Output('confusion-matrix', 'style'),
    Output('recall', 'children'),
    Output('precision', 'children'),
    Output('f1-score', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('eval-date-picker-range', 'start_date'),
    Input('eval-date-picker-range', 'end_date'),
    Input('eval-conf-slider', 'value'),
)
def eval_inputs_changed(start_date, end_date, confidence):
    if not start_date or not end_date or not confidence:
        return {'display': 'none'}, {'display': 'none'}, '', '', '', go.Figure()

    y_true, y_pred = get_eval_data(start_date, end_date, confidence / 100)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    cm_fig = ff.create_annotated_heatmap(
        x=['No fire', 'Fire'],
        y=['No fire', 'Fire'],
        z=confusion_matrix,
    )
    cm_fig.update_layout(
        title_text='<b>Confusion Matrix</b>',
        xaxis={'title': 'Prediction'},
        yaxis={'title': 'Truth'},
    )
    cm_fig['data'][0]['showscale'] = True

    return (
        {'display': 'initial'},
        {'display': 'initial'},
        '{:.3f}'.format(recall),
        '{:.3f}'.format(precision),
        '{:.3f}'.format(f1_score),
        cm_fig,
    )

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("User Guide", href="/userguide", active="exact"),
                dbc.NavLink("Predictions", href="/predictions", active="exact"),
                dbc.NavLink("History", href="/history", active="exact"),
                dbc.NavLink("Evaluation", href="/evaluation", active="exact"),
                dbc.NavLink(
                    "Modeling Details", href="/modelingdetails", active="exact"
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
dash_app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Store(id = 'clicks',
              data = 0)])

@dash_app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")],
)
def render_page_content(pathname):
    if pathname == "/":
        with open('home.md') as f:
            return html.Div(dcc.Markdown(f.read()))
    elif pathname == "/userguide":
        with open('user_guide.md') as f:
            return html.Div(dcc.Markdown(f.read()))
    elif pathname == "/predictions":
        return get_layout()
    elif pathname == "/history":
        return get_history_layout()
    elif pathname == "/evaluation":
        return get_eval_layout()
    elif pathname == "/modelingdetails":
        with open('modeling_details.md') as f:
            return html.Div(dcc.Markdown(f.read()))
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
if __name__ == '__main__':
    dash_app.run_server(debug=True)
