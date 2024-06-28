import util as ut
import features_analysis as fa
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from LinearModel_Diamonds import LinearModelDiamonds
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
import argparse
import json

def get_model(model_name):
    models = {
        "LinearRegression": LinearRegression(),
        "LogisticRegression": LogisticRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    return models.get(model_name, Lasso())  # Default to Lasso if model_name is not found

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_model(config):
    model_name = config.get("model", "Lasso")
    path = config.get("path", "data/diamonds.csv")
    test_size = config.get("test_size", 0.2)

    diamonds = ut.load_preprocess_dataset(path)
    LinearModel = LinearModelDiamonds(get_model(model_name))
    diamonds = LinearModel.preprocessing(diamonds)

    x = diamonds.drop(columns='price')
    y = diamonds.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    pred = LinearModel.fit_predict(x_train, y_train, x_test)
    
    results = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

    diamonds = ut.load_preprocess_dataset(path)
    pred = LinearModel.predict(diamonds.iloc[0:1])
    print("pred: ", pred)
    print("MAE: ", results["MAE"])
    print("R2: ", results["R2"])
    # ut.save_results(results, LinearModel, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a linear model on the diamonds dataset.")
    parser.add_argument("config", type=str, help="Path to the config file (config.json)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_model(config)