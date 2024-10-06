# Import packages
from LinearRegression import Normal
from dash import Dash, html, Output, Input, dcc
import dash_bootstrap_components as dbc

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, use_pages=True,  external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.config.suppress_callback_exceptions = True

from pages.home import *

# Navigation Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Predict V1", href="/predict")),
        dbc.NavItem(dbc.NavLink("Predict V2", href="/predictvone")),
        dbc.NavItem(dbc.NavLink("Predict V3", href="/predictvthree"))
    ],
    brand="ML2024: A3 Car Price Prediction",
    brand_href="/",
    color="primary",
    dark=True,
)


app.layout = html.Div([
    navbar,
    dash.page_container
])

# Run the app
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(host='0.0.0.0', port='80',debug=True)
    from utils import load_mlflow
    load_mlflow(stage="Production")
    app.run(debug=True)