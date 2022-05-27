# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:13:02 2022

@author: user
"""

import pickle
from flask import Flask, request
import numpy as np

app = Flask(__name__)

classifier_model = pickle.load(open("flower.pkl","rb"))

# http://localhost:5000/api_predict

@app.route("/api_predict", methods=["GET","POST"])
def api_predict():
    if request.method == "GET":
        return "please send a post request"
    elif request.method == "POST":
        
        data = request.get_json()
    
        sepal_length = data['sepal_length']
        sepal_width =  data['sepal_width']
        petal_length = data['petal_length']
        petal_width =  data['petal_width']
    
        input1 = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    
        prediction = classifier_model.predict(input1)
        return str(prediction)
 
if __name__ == "__main__" :   
    app.run()