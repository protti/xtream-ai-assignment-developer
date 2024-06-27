from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self.model = None


    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> object:
        pass

    @abstractmethod
    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, y_test: pd.Series, y_pred: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def save_model(self, path: str) -> str:
        pass
    
    @abstractmethod
    def get_type_model(self) -> str:
        pass