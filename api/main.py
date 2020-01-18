from flask import Flask, jsonify, request
import pickle
import numpy as np
import lightgbm

app = Flask(__name__)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    with open('../model/iris.pkl', 'rb') as f:
        model = pickle.load(f)
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])    
    Y = model.predict(X)
    cate_map = {0: 'setosa',
                1: 'versicolor',
                2: 'virginica'}    
    
    return cate_map[Y[0]]


@app.route('/score', methods=['POST'])
def post_score():
    try:
	    param = request.get_json()
	    sepal_length = float(param['sepal_length']) 
	    sepal_width = float(param['sepal_width'])
	    petal_length = float(param['petal_length'])
	    petal_width = float(param['petal_width'])
	    score = predict(sepal_length, sepal_width, petal_length, petal_width)
	    return jsonify(
	        {'message': score}
	    ) 
    except Exception as e:
    	return jsonify(
        	{'message': str(e)}
        ) 
if __name__ == '__main__':
    app.run()

