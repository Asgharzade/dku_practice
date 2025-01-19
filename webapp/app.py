import dash
from dash import dcc, html
import dash_auth

# Define username and password pairs
VALID_USERNAME_PASSWORD_PAIRS = {
    'dataiku': 'dataiku',
}

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
    dcc.Dropdown(
        id='example-dropdown',
        options=[
            {'label': 'Option 1', 'value': '1'},
            {'label': 'Option 2', 'value': '2'},
            {'label': 'Option 3', 'value': '3'}
        ],
        value='1'  # default value
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)