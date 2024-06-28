from flask import Flask, render_template
from flask_restful import Api
import logging
from sklearn import linear_model
from xgboost import XGBRegressor
from flask_cors import CORS
from resources.PredictPrice import Predict
from resources.ObtainCloser import Closer

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
api = Api(app)
CORS(app)


api.add_resource(Predict, '/predict-price/')
api.add_resource(Closer, '/closer-diamond/')

@app.route('/predict')
def predict_page():
    return render_template('predict_price.html')

@app.route('/closer')
def closer_page():
    return render_template('find_closer.html')

if __name__ == '__main__':
    app.run(debug=True)