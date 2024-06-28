import util as ut
import features_analysis as fa
import Models.XGBoost_Diamonds as xgb
from Models.LinearModel_Diamonds import LinearModelDiamonds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import json
import argparse

def main(config_path):
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load the dataset
    path = config["data_path"]
    diamonds = ut.load_preprocess_dataset(path)

    # Feature evaluation
    diamonds = fa.features_evaluation(diamonds)

    model_to_use = config["model_to_use"]
    
    if model_to_use == "LinearModel":
        linear_model_type = config["linear_model_type"]
        if linear_model_type == "Lasso":
            model = LinearModelDiamonds(Lasso())
        elif linear_model_type == "LinearRegression":
            model = LinearModelDiamonds(LinearRegression())
        else:
            raise ValueError(f"Unknown linear model type: {linear_model_type}")
    
    elif model_to_use == "XGBModel":
        model = xgb.XGBoostDiamonds(optimized_params=config["use_optimized"])
        if config.get("use_optimized", False):
            n_trials = config["n_trials"]
        else:
            n_trials = None
    else:
        raise ValueError(f"Unknown model: {model_to_use}")


    # Data preprocessing
    diamonds_processed = model.preprocessing(diamonds)
    x_train, x_test, y_train, y_test = train_test_split(
        diamonds_processed.drop(columns='price'), 
        diamonds_processed['price'], 
        test_size=config["test_size"], 
        random_state=config["random_state"]
    )

    if model_to_use == "XGBModel" and n_trials is not None:
        
        y_pred = model.fit_predict(x_train, y_train, x_test, y_test, n_trials=n_trials)
    else:
        y_pred = model.fit_predict(x_train, y_train, x_test)

    # Model evaluation
    results = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    print("-------------------------------")
    print(f"MAE {model_to_use}: ", results["MAE"])
    print(f"R2 {model_to_use}: ", results["R2"])

    # Save the results to a CSV file
    ut.save_results(results, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model specified in the configuration file.')
    parser.add_argument('config_path', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()
    main(args.config_path)