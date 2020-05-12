# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:18:47 2020

@author: degananda.reddy
"""

from flask import Flask,request, url_for, redirect, render_template, jsonify

import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio', 'Male_2',
       'Normal_2', 'Normal_3', 'gluc_2', 'gluc_3', 'smoke_1', 'alco_1']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)