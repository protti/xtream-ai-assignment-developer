from flask_restful import Resource
from flask import request
import pickle, logging, pandas as pd


class Closer(Resource):
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"Model file not found at path: {model_path}")
            self.model = None
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            self.model = None

    def get_similar_samples(self, sample: pd.Series, n: int) -> pd.DataFrame:

        filtered_data = self.model.training_data[
            (self.model.training_data['cut'] == sample['cut'].values[0]) &
            (self.model.training_data['color'] == sample['color'].values[0]) &
            (self.model.training_data['clarity'] == sample['clarity'].values[0])
        ]
        filtered_data['carat_diff'] = (filtered_data['carat'] - sample['carat'].values[0]).abs()
        closest_sample = filtered_data.nsmallest(n, 'carat_diff').drop(columns=['carat_diff'])
        
        return closest_sample

    def post(self):
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
        return {"closer": similar_samples.to_dict(orient='records')}