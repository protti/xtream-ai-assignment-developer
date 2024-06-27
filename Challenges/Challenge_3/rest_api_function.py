from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import pickle, logging, pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
api = Api(app)
CORS(app)

class Predict(Resource):
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                logging.error(f"Model: {self.model}")
        except FileNotFoundError:
            logging.error(f"Model file not found at path: {model_path}")
            self.model = None
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            self.model = None

    def adapt_data_to_model(self, data, model):
        logging.debug(f"Model Name: {model.__class__.__name__.lower()}")
        if "linearmodel" in model.__class__.__name__.lower():
            data = pd.get_dummies(data, columns=['cut', 'color', 'clarity'], drop_first=False)
            expected_columns = [
                'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 
                'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 
                'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 
                'clarity_VVS1', 'clarity_VVS2'
            ]
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = False
            for col in expected_columns:
                data[col] = data[col].astype('category').cat.codes
        elif "xgboost" in model.__class__.__name__.lower():
            for col in ['cut', 'color', 'clarity']:
                data[col] = data[col].astype('category').cat.codes
            logging.debug(f"Data: {data}")
        return data

    def put(self):
        data = request.get_json()
        required_fields = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "path"]
        missing_fields = [field for field in required_fields if data.get(field) is None]
        if missing_fields:
            return {'message': f'Missing fields: {", ".join(missing_fields)}'}, 400

        data = {k: (float(v) if k in ["carat", "depth", "table", "x", "y", "z"] else str(v)) for k, v in data.items()}
        self.load_model(data.get("path"))
        data_df = pd.DataFrame([data]).drop(columns=["path"])
        data_original = self.adapt_data_to_model(data_df, self.model)
        data_keep = data_original[self.model.features_adopted]
        predicted_value = float(self.model.predict(data_keep)[0])
        logging.debug(f"Predict: {predicted_value}")
        return {"price": predicted_value}

class Closer(Resource):
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                logging.error(f"Model: {self.model}")
        except FileNotFoundError:
            logging.error(f"Model file not found at path: {model_path}")
            self.model = None
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            self.model = None

    def get_similar_samples(self, sample: pd.Series, n: int) -> pd.DataFrame:
        logging.debug(f"Model: {self.model.training_data}")
        filtered_data = self.model.training_data[
            (self.model.training_data['cut'] == sample['cut'].values[0]) &
            (self.model.training_data['color'] == sample['color'].values[0]) &
            (self.model.training_data['clarity'] == sample['clarity'].values[0])
        ]
        logging.debug(f"Filtered Data: {filtered_data}")
        filtered_data['carat_diff'] = (filtered_data['carat'] - sample['carat'].values[0]).abs()
        closest_sample = filtered_data.nsmallest(n, 'carat_diff').drop(columns=['carat_diff'])
        logging.debug(f"Closest Sample: {closest_sample}")
        return closest_sample

    def put(self):
        data = request.get_json()
        required_fields = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "path", "n_neighbors"]
        missing_fields = [field for field in required_fields if data.get(field) is None]
        if missing_fields:
            return {'message': f'Missing fields: {", ".join(missing_fields)}'}, 400

        data = {k: (float(v) if k in ["carat", "depth", "table", "x", "y", "z"] else str(v)) for k, v in data.items()}
        data['n_neighbors'] = int(data['n_neighbors'])
        self.load_model(data.get("path"))
        data_df = pd.DataFrame([data])
        similar_samples = self.get_similar_samples(data_df, data.get("n_neighbors"))
        logging.debug(f"Similar Samples: {similar_samples}")
        return {"closer": similar_samples.to_dict(orient='records')}

api.add_resource(Predict, '/predict_value/')
api.add_resource(Closer, '/closer/')

@app.route('/predict')
def predict_page():
    return render_template('predict_price.html')

@app.route('/closer')
def closer_page():
    return render_template('find_closer.html')

if __name__ == '__main__':
    app.run(debug=True)