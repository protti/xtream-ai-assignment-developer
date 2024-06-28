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

    def adapt_data_to_model(self, data, model):
        """
        Adapt the input data to match the model's expected format.
        
        :param data: Input data as a pandas DataFrame
        :param model: The machine learning model
        :return: Adapted data as a pandas DataFrame
        """
        if "linearmodel" in model.__class__.__name__.lower():
            # Convert categorical variables to dummy/indicator variables
            data = pd.get_dummies(data, columns=['cut', 'color', 'clarity'], drop_first=False)
            expected_columns = [
                'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 
                'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 
                'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 
                'clarity_VVS1', 'clarity_VVS2'
            ]
            # Ensure all expected columns are present in the data
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = False
            # Convert categorical columns to numerical codes
            for col in expected_columns:
                data[col] = data[col].astype('category').cat.codes
        elif "xgboost" in model.__class__.__name__.lower():
            # Convert categorical variables to numerical codes
            for col in ['cut', 'color', 'clarity']:
                data[col] = data[col].astype('category').cat.codes

        return data

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
        # Adapt the data to match the model's expected format
        data_original = self.adapt_data_to_model(data_df, self.model)
        data_keep = data_original[self.model.features_adopted]
        # Predict the price using the model
        predicted_value = float(self.model.predict(data_keep)[0])
        return {"price": predicted_value}