from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import requests
import pandas as pd
from dash import dash_table
import json

app = Dash(__name__)

response_data_client = requests.get('http://localhost:5000/data')
response_pred_client = requests.get('http://localhost:5000/prediction')

data = json.loads(response_data_client.text)
pred = response_pred_client.json()
print(response_data_client.text)
print(response_pred_client.text)

# options = [{'label': k, 'value': k} for k in data.keys()]
# df = pd.DataFrame.from_dict(data, orient='index', columns=list(data.keys()))

df = pd.DataFrame(data)

table = html.Div(
    [
        html.H3('Mon dataframe'),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records')
        ),
    ]
)


# dropdown = dcc.Dropdown(
#     options=options,  # options à partir des clés du dictionnaire
#     value=list(data.keys())[0]  # valeur par défaut est la première clé du dictionnaire
# )

app.layout = html.Div(children=[
    html.H1('Dashboard Client'),
    html.Label('Enter Customer ID'),
    dcc.Input(id='ID_ClIENT', type='number', value=1),
    html.Button('Submit', id='submit-button', n_clicks=0),
    # dropdown,
    html.Label(f'La prédiction {str(pred)}'),
    table
    # html.Div(id='output-div')
    
])

# print("la prédiction")
# print(pred)


if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    # app.layout = html.Div([
#     html.H1(children='Ma fonction mon dash', style={'textAlign':'center'}),
#     dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
#     dcc.Graph(id='graph-content')
  
# ])

# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )

# def update_graph(value):
#     dff = df[df.country==value]
#     return px.line(dff, x='year', y='pop')