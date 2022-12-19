#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
with open('lr.pickle', 'rb') as lr:
    model=pickle.load(lr)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0], 2)
    
    # arguments will be called on the webpage
    return render_template('index.html', prediction_text=f'House price should be {output}')

if __name__ == "__main__":
    app.run(port=5000, debug=True)