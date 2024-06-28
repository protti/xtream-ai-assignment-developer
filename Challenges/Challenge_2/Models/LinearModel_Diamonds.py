import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import time
import os
from .BaseModel_Diamonds import BaseModel

class LinearModelDiamonds(BaseModel):
    """
    This class implements a linear model for predicting diamond prices.

    Attributes:
    model (sklearn.linear_model): The linear model to be used for predictions.
    """
    def __init__(self, model: sk.linear_model):
        """
        Initializes the LinearModelDiamonds class with a given linear model.

        Parameters:
        model (sklearn.linear_model): The linear model to be used for predictions.
        """
        self.model = model
        self.features_adopted = []

    def preprocessing(self, diamonds_processed: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the diamonds DataFrame by converting categorical columns to dummy variables.

        Parameters:
        diamonds_processed (pd.DataFrame): The input DataFrame containing diamond data.

        Returns:
        pd.DataFrame: The processed DataFrame with categorical columns converted to dummy variables.
        """
        diamonds_processed = pd.get_dummies(diamonds_processed, columns=['cut', 'color', 'clarity'], drop_first=True)
        return diamonds_processed


    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame=None, y_test: pd.Series=None) -> sk.linear_model:
        """
        Fits the linear model to the training data.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.

        Returns:
        LinearModelDiamonds: The fitted model.
        """
        self.model.fit(x_train, y_train)
        self.features_adopted = x_train.columns
        return self

    
    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
        """
        Fits the model to the training data and makes predictions on the test data.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.
        x_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data target.

        Returns:
        np.ndarray: The predicted values for the test data.
        """
        self.model.fit(x_train, y_train)
        self.features_adopted = x_train.columns
        y_pred = self.model.predict(x_test)
        return y_pred

    
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on the test data using the fitted model.

        Parameters:
        x_test (pd.DataFrame): The test data features.

        Returns:
        np.ndarray: The predicted values for the test data.
        """
        y_pred = self.model.predict(x_test)
        return y_pred

    
    def score(self, y_test: pd.Series, y_pred: np.ndarray) -> tuple:
        """
        Calculates the R2 score and Mean Absolute Error (MAE) for the predictions.

        Parameters:
        y_test (pd.Series): The true target values.
        y_pred (np.ndarray): The predicted values.

        Returns:
        tuple: The R2 score and MAE.
        """
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return r2, mae
    
    def save_model(self, path: str) -> str:
        """
        Saves the trained model to a pickle file.

        Parameters:
        path (str): The directory path where the model will be saved.

        Returns:
        str: The filename of the saved model.
        """
        time.sleep(1)
        filename = f"{self.get_type_model()}_{int(time.time())}.pkl"

        full_path = os.path.join(path, filename)


        # Save the model in a pickle file    
        with open(full_path, 'wb') as file:
            pickle.dump(self, file)

        return filename
    
    def get_type_model(self) -> str:
        """
        Returns the type of the linear model.

        Returns:
        str: The type of the linear model.
        """
        return f"LinearModel_Diamonds_{type(self.model).__name__}"