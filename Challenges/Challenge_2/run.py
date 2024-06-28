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

def load_config(config_path):
    """
    Load the configuration from a JSON file.

    Parameters:
    config_path (str): Path to the JSON configuration file.

    Returns:
    dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def load_and_preprocess_data(config):
    """
    Load and preprocess the diamonds dataset.

    Parameters:
    config (dict): Configuration dictionary containing the data path.

    Returns:
    DataFrame: Preprocessed diamonds dataset.
    """
    path = config["data_path"]
    diamonds = ut.load_preprocess_dataset(path)
    return fa.features_evaluation(diamonds)

def get_model(config):
    """
    Get the model specified in the configuration.

    Parameters:
    config (dict): Configuration dictionary specifying the model to use and its parameters.

    Returns:
    object: An instance of the specified model.
    """
    model_to_use = config["model_to_use"]
    if model_to_use == "LinearModel":
        linear_model_type = config["linear_model_type"]
        if linear_model_type == "Lasso":
            return LinearModelDiamonds(Lasso())
        elif linear_model_type == "LinearRegression":
            return LinearModelDiamonds(LinearRegression())
        else:
            raise ValueError(f"Unknown linear model type: {linear_model_type}")
    elif model_to_use == "XGBModel":
        return xgb.XGBoostDiamonds(optimized_params=config["use_optimized"])
    else:
        raise ValueError(f"Unknown model: {model_to_use}")

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate the model on the test data.

    Parameters:
    model (object): The model to be trained and evaluated.
    x_train (DataFrame): Training features.
    y_train (Series): Training target values.
    x_test (DataFrame): Testing features.
    y_test (Series): Testing target values.

    Returns:
    dict: Dictionary containing the Mean Absolute Error (MAE) and R-squared (R2) score of the model.
    """
    y_pred = model.fit_predict(x_train, y_train, x_test, y_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

def main(config_path):
    """
    Main function to orchestrate the loading of configuration, preprocessing of data, model training, evaluation, and saving of results.

    Parameters:
    config_path (str): Path to the JSON configuration file.
    """
    config = load_config(config_path)
    diamonds = load_and_preprocess_data(config)
    model = get_model(config)

    diamonds_processed = model.preprocessing(diamonds)
    x_train, x_test, y_train, y_test = train_test_split(
        diamonds_processed.drop(columns='price'), 
        diamonds_processed['price'], 
        test_size=config["test_size"], 
        random_state=config["random_state"]
    )

    results = evaluate_model(model, x_train, y_train, x_test, y_test)

    print("-------------------------------")
    print(f"MAE {config['model_to_use']}: ", results["MAE"])
    print(f"R2 {config['model_to_use']}: ", results["R2"])

    ut.save_results(results, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model specified in the configuration file.')
    parser.add_argument('config_path', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()
    main(args.config_path)