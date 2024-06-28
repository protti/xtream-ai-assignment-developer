from flask_restful import Resource
from flask import request
import pickle, logging, pandas as pd
from .Observability import db, Observability

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

    def post(self):
        data = request.get_json()
        required_fields = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "path"]
        missing_fields = [field for field in required_fields if data.get(field) is None]
        if missing_fields:
            response = {'message': f'Missing fields: {", ".join(missing_fields)}'}
            db.session.add(Observability(type='PredictPrice', operation='POST', request=str(data), response=str(response)))
            db.session.commit()
            return response, 400

        data = {k: (float(v) if k in ["carat", "depth", "table", "x", "y", "z"] else str(v)) for k, v in data.items()}
        self.load_model(data.get("path"))
        data_df = pd.DataFrame([data]).drop(columns=["path"])
        data_original = self.adapt_data_to_model(data_df, self.model)
        data_keep = data_original[self.model.features_adopted]
        predicted_value = float(self.model.predict(data_keep)[0])
        response = {"price": predicted_value}
        logging.debug(f"Predict: {predicted_value}")
        
        db.session.add(Observability(type='PredictPrice', operation='POST', request=str(data), response=str(response)))
        db.session.commit()
        
        return response
