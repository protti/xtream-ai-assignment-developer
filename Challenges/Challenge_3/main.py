import util as ut
import features_analysis as fa
import Models.XGBoost_Diamonds as xgb
from Models.LinearModel_Diamonds import LinearModelDiamonds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    # Load the dataset
    path = "../../data/diamonds.csv"
    diamonds = ut.load_preprocess_dataset(path)
    

    # Evaluation of the features
    diamonds = fa.features_evaluation(diamonds)
    
    # Creation of the models
    LinearModel = LinearModelDiamonds(Lasso())
    XGBModel = xgb.XGBoostDiamonds()
    XGBModelOptimized = xgb.XGBoostDiamonds(optimized_params=True, n_trials=10)

    # Preprocessing of the data inside the model. 
    # I opt for this decision because it will be useful later when we need to evaluate the model with different preprocessing methods.
    diamonds_processed = XGBModel.preprocessing(diamonds)
    diamonds_processed_linear = LinearModel.preprocessing(diamonds)

    
    # Splitting of the data
    x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(diamonds_processed.drop(columns='price'), 
                                                                            diamonds_processed['price'], 
                                                                            test_size=0.2, 
                                                                            random_state=42)
    
    x_train_linear, x_test_linear, y_train_linear, y_test_linear = train_test_split(diamonds_processed_linear.drop(columns='price'), 
                                                                            diamonds_processed_linear['price'], 
                                                                            test_size=0.2, 
                                                                            random_state=42)
    
    
    # I left the user choose if we want to create the model with optimized hyperparameters or not.
    y_pred_optimized = XGBModelOptimized.fit_predict(x_train_xbg, y_train_xbg, x_test_xbg, y_test_xbg)
    y_pred = XGBModel.fit_predict(x_train_xbg, y_train_xbg, x_test_xbg)
    y_pred_linear = LinearModel.fit_predict(x_train_linear, y_train_linear, x_test_linear)
    
    # Evaluation of the model
    results = {
        "MAE": mean_absolute_error(y_test_xbg, y_pred),
        "R2": r2_score(y_test_xbg, y_pred),
    }

    results_linear = {
        "MAE": mean_absolute_error(y_test_linear, y_pred_linear),
        "R2": r2_score(y_test_linear, y_pred_linear),
    }

    results_optimized = {
        "MAE": mean_absolute_error(y_test_xbg, y_pred_optimized),
        "R2": r2_score(y_test_xbg, y_pred_optimized),
    }
    print("-------------------------------")
    print("MAE XGBoost: ", results["MAE"])
    print("R2 XGBoost: ", results["R2"])
    print("-------------------------------")
    print("MAE Linear: ", results_linear["MAE"])
    print("R2 Linear: ", results_linear["R2"])
    print("-------------------------------")
    print("MAE XGBoost Optimized: ", results_optimized["MAE"])
    print("R2 XGBoost Optimized: ", results_optimized["R2"])

    # Save the results in a csv file
    ut.save_results(results, XGBModel)
    ut.save_results(results_linear, LinearModel)
    ut.save_results(results_optimized, XGBModelOptimized)