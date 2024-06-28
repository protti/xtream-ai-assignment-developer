from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import pickle, logging, pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from resources.PredictPrice import Predict
from resources.ObtainCloser import Closer

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


class Observability(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(100), nullable=False)
    operation = db.Column(db.String(100), nullable=False)
    request = db.Column(db.String(100), nullable=True)
    response = db.Column(db.String(100), nullable=True)
	
    def __repr__(self):
        return f"Observe(type = {type}, operation = {operation}, request = {request}, response = {response})"

# Create all tables (this should be done before running the app)
if not db.engine.dialect.has_table(db.engine, 'observability'):
    db.create_all()


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