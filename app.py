from flask import Flask, request, render_template
import numpy as numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData

application= Flask(__name__)

app= application

data_client = CustomData()

## Route for a home page


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # print("étape1")
        SK_ID_CURR = int(request.form.get("client_id"))
        # print("étape2")
        df= data_client.get_data_as_data_frame(SK_ID_CURR)
        # print("étape3")
        pred = data_client.predict_function(df)
        return render_template('home.html', results= pred[0], ID =SK_ID_CURR)

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
# Essai
# client n° : 100065 100092 100117 100150 100171 100232 100253 100280 100331 / 100038  100005 100444 100091