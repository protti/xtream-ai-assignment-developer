from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

class XGBoostDiamonds:
    def __init__(self, diamonds: pd.DataFrame, target_column: str):
        self.diamonds = diamonds
        self.target_column = target_column
        self.model = None

    def preprocessing(self):
        diamonds_processed_xgb = self.diamonds.copy()
        diamonds_processed_xgb['cut'] = pd.Categorical(diamonds_processed_xgb['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
        diamonds_processed_xgb['color'] = pd.Categorical(diamonds_processed_xgb['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
        diamonds_processed_xgb['clarity'] = pd.Categorical(diamonds_processed_xgb['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
        return diamonds_processed_xgb
    
    def fit(self, x_train, y_train):
        self.model = XGBRegressor(enable_categorical=True, random_state=42)
        self.model.fit(x_train, y_train)
        return self.model

    def fit_predict(self, x_train, y_train, x_test):
        self.fit(x_train,y_train)
        y_pred = self.model.predict(x_test)
        return y_pred
    
    def predict(self, x_test):
        return self.model.predict(x_test)

    def score(self, y_test, y_pred):
        return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    