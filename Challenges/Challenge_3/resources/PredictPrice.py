from flask_restful import Resource
from flask import request
import pickle, logging, pandas as pd

class Predict(Resource):
    def __init__(self):
        # Initialize the Predict resource with a model attribute set to None
        self.model = None

    def load_model(self, model_path):
        """
        Load a machine learning model from a specified file path.
        
        :param model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"Model file not found at path: {model_path}")
            self.model = None
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")
            self.model = None

    def post(self):
        """
        Handle POST requests to predict the price based on the provided data.
        
        :return: A JSON response containing the predicted price or an error message
        """
        data = request.get_json()
        required_fields = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "path"]
        # Check for missing fields in the request data
        missing_fields = [field for field in required_fields if data.get(field) is None]
        if missing_fields:
            return {'message': f'Missing fields: {", ".join(missing_fields)}'}, 400

        # Convert data types as necessary
        data = {k: (float(v) if k in ["carat", "depth", "table", "x", "y", "z"] else str(v)) for k, v in data.items()}
        # Load the model from the specified path
        self.load_model(data.get("path"))
        data_df = pd.DataFrame([data]).drop(columns=["path"])
        predicted_value = float(self.model.predict(data_df)[0])
        return {"price": predicted_value}