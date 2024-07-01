from flask_restful import Resource
from flask import request
import pickle, logging, pandas as pd
from .Observability import db, Observability
from datetime import datetime

class Closer(Resource):
    def __init__(self):
        # Initialize the Closer resource with a model attribute set to None
        self.model = None

    def load_model(self, model_path):
        """
        Load a machine learning model from a specified file path.
        
        :param model_path: Path to the model file
        """
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
        """
        Get the n most similar samples from the model's training data based on the provided sample.
        
        :param sample: A pandas Series representing the sample to compare
        :param n: Number of similar samples to retrieve
        :return: A DataFrame containing the n most similar samples
        """

        # Filter the training data based on the sample's attributes
        filtered_data = self.model.training_data[
            self.model.training_data.filter(like='cut').eq(sample.filter(like='cut').values[0]).all(axis=1) &
            self.model.training_data.filter(like='color').eq(sample.filter(like='color').values[0]).all(axis=1) &
            self.model.training_data.filter(like='clarity').eq(sample.filter(like='clarity').values[0]).all(axis=1)
        ]
        # Calculate the absolute difference in carat and find the closest samples
        filtered_data['carat_diff'] = (filtered_data['carat'] - sample['carat'].values[0]).abs()
        closest_sample = filtered_data.nsmallest(n, 'carat_diff').drop(columns=['carat_diff'])
        
        return closest_sample
        

    def post(self):
        """
        Handle POST requests to find the closest samples based on the provided data.
        
        :return: A JSON response containing the closest samples or an error message
        """
        data = request.get_json()
        required_fields = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "path", "n_neighbors"]
        # Check for missing fields in the request data
        missing_fields = [field for field in required_fields if data.get(field) is None]
        if missing_fields:
            response = {'message': f'Missing fields: {", ".join(missing_fields)}'}
            # Log the missing fields in the Observability table
            db.session.add(Observability(method='POST', timestamp=datetime.now(), model=data.get("path"), type_request='CloserDiamond', request=str(data), response=str(response)))
            db.session.commit()
            return response, 400

        # Convert data types as necessary
        data = {k: (float(v) if k in ["carat", "depth", "table", "x", "y", "z"] else str(v)) for k, v in data.items()}
        data['n_neighbors'] = int(data['n_neighbors'])
        # Load the model from the specified path
        self.load_model(data.get("path"))
        data_df = pd.DataFrame([data])
        data_df = self.model.preprocessing(data_df)
        # Get similar samples from the model's training data
        similar_samples = self.get_similar_samples(data_df, data.get("n_neighbors"))
        response = {"closer": similar_samples.to_dict(orient='records')}
        
        
        # Log the request and response in the Observability table
        db.session.add(Observability(method='POST', timestamp=datetime.now(), model=data.get("path"), type_request='CloserDiamond', request=str(data), response=str(response)))
        db.session.commit()
        
        return response