from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import pickle, logging, pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from resources.PredictPrice import Predict
from resources.ObtainCloser import Closer
from resources.Observability import db, Observability
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db.init_app(app)

def init_db():
    with app.app_context():
        if not db.engine.dialect.has_table(db.engine, 'observability'):
            db.create_all()


init_db()



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