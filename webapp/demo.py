#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# author: JL

from flask import Flask, render_template, request
import numpy as np
import pickle
import lightgbm

app = Flask(__name__)

# 載入資料, 用lgbm_model預測, 回傳分數
def predict(sepal_length, sepal_width, petal_length, petal_width):
    with open('../model/iris.pkl', 'rb') as f:
        model = pickle.load(f)
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])    
    Y = model.predict(X)
    cate_map = {0: 'setosa',
                1: 'versicolor',
                2: 'virginica'}    
    
    return cate_map[Y[0]]

@app.route("/", methods=['GET', 'POST'])

def post_score():
	# 抓資料, 用request.values[] 取值 
    if request.method == 'POST':
        sepal_length = float(request.values['sepal_length']) 
        sepal_width = float(request.values['sepal_width'])
        petal_length = float(request.values['petal_length'])
        petal_width = float(request.values['petal_width'])
        score = predict(sepal_length, sepal_width, petal_length, petal_width)
        return render_template('score.html', final_score=score)
    return render_template('score.html')

if __name__ == "__main__":
    app.debug = True
    app.run()



