from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import optuna
import numpy as np
from .BaseModel_Diamonds import BaseModel
import pickle
import os
import time

"""
This class implements the XGBoost method for predicting diamond prices.
"""
class XGBoostDiamonds(BaseModel):
    def __init__(self, optimized_params: bool = False, n_trials: int = 20, random_state: int = 42, enable_categorical: bool = True):
        self.model = None
        self.optimized_params = optimized_params
        self.features_adopted = []
        self.n_trials = n_trials
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        
    def preprocessing(self, diamonds: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the diamonds DataFrame by converting categorical columns to ordered categorical types.

        Parameters:
        diamonds (pd.DataFrame): The input DataFrame containing diamond data.

        Returns:
        pd.DataFrame: The processed DataFrame with categorical columns converted.
        """
        diamonds_processed_xgb = diamonds.copy()
        diamonds_processed_xgb['cut'] = pd.Categorical(diamonds_processed_xgb['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
        diamonds_processed_xgb['color'] = pd.Categorical(diamonds_processed_xgb['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
        diamonds_processed_xgb['clarity'] = pd.Categorical(diamonds_processed_xgb['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
        return diamonds_processed_xgb
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame=None, y_test: pd.Series=None) -> XGBRegressor:
            
        """
        Fits the XGBoost model to the training data.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.
        enable_categorical (bool): Whether to enable categorical data handling. Default is True.
        random_state (int): The random seed for reproducibility. Default is 42.

        Returns:
        XGBRegressor: The fitted XGBoost model.
        """
    
        self.features_adopted = x_train.columns
        if self.optimized_params:
            assert x_test is not None and y_test is not None, "x_test and y_test must be provided if optimized_params is True"
            self.model = XGBRegressor(**self.optimize_hyperparameters(x_train, y_train, x_test, y_test, self.n_trials), enable_categorical=self.enable_categorical, random_state=self.random_state)
        else:
            self.model = XGBRegressor(enable_categorical=self.enable_categorical, random_state=self.random_state)
        
        self.model.fit(x_train, y_train)
        self.best_params = self.model.get_xgb_params()
        return self.model

    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series=None) -> np.ndarray:
        """
        Fits the model to the training data and makes predictions on the test data.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.
        x_test (pd.DataFrame): The test data features.

        Returns:
        np.ndarray: The predicted values for the test data.
        """
        self.features_adopted = x_train.columns
        if self.optimized_params:
            assert x_test is not None and y_test is not None, "x_test and y_test must be provided if optimized_params is True"
            self.fit(x_train, y_train, x_test, y_test)
        else:
            self.fit(x_train, y_train)
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
        return self.model.predict(x_test)

    def score(self, y_test: pd.Series, y_pred: np.ndarray) -> tuple:
        """
        Calculates the R2 score and Mean Absolute Error (MAE) for the predictions.

        Parameters:
        y_test (pd.Series): The true target values.
        y_pred (np.ndarray): The predicted values.

        Returns:
        tuple: The R2 score and MAE.
        """
        return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)


    def objective(self, x_train, y_train, x_test, y_test, trial: optuna.trial.Trial) -> float:
        """
        Defines the objective function for hyperparameter optimization using Optuna.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.
        x_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data target.
        trial (optuna.trial.Trial): The Optuna trial object.

        Returns:
        float: The Mean Absolute Error (MAE) of the model predictions.
        """
        # Define hyperparameters to tune
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }

        # Train the model
        model = XGBRegressor(**param)
        model.fit(x_train, y_train)
        # Make predictions
        preds = model.predict(x_test)
        # Calculate MAE
        mae = mean_absolute_error(y_test, preds)
        return mae
    
    def optimize_hyperparameters(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, n_trials: int = 20) -> dict:
        """
        Optimizes the hyperparameters of the XGBoost model using Optuna.

        Parameters:
        x_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data target.
        x_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data target.
        n_trials (int): The number of trials for optimization. Default is 20.

        Returns:
        dict: The best hyperparameters found during optimization.
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(x_train, y_train, x_test, y_test, trial), n_trials=n_trials)
        self.best_params = study.best_params
        self.optimized_params = True
        return study.best_params
    
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
        Returns the type of the XGBoost model.

        Returns:
        str: The type of the XGBoost model.
        """
        if self.optimized_params:
            return "XGBoost_Diamonds_XGBRegressor_optimized"
        else:
            return "XGBoost_Diamonds_XGBRegressor"
    
