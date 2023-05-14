from flask import Flask
from src.pipeline.predict_pipeline import DataClient
import json

"""
# Essai
# client n° : 100065 100092 100117 100150 100171 100232 100253 100280 100331 / 100038  100005 100444 100091

"""


application= Flask(__name__)
app= application

data_clients = DataClient()


@app.route('/data/<int:id_client>', methods=['GET'])
def results_json(id_client):
    df_client = data_clients.get_data_as_df(ID_client= int(id_client))
    data_json = df_client.to_json(orient="records")
    return data_json

@app.route('/prediction/<int:id_client>', methods=['GET'])
def prediction(id_client):
    df_client = data_clients.get_data_as_df(ID_client= int(id_client))
    pred = data_clients.predict_function(df_client)
    return json.dumps(pred, indent=4)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
