import util as ut
import features_analysis as fa
import XGBoost_Diamonds as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    # Load the dataset
    path = "data/diamonds.csv"
    diamonds = ut.load_preprocess_dataset(path)
     
    # Normalization of the data
    # numeric_columns = diamonds.select_dtypes(include=[np.number]).columns
    # scaler = StandardScaler()
    # diamonds[numeric_columns] = scaler.fit_transform(diamonds[numeric_columns])


    # Evaluation of the features
    diamonds = fa.features_evaluation(diamonds)
    

    # Creation of the model
    XGBModel = xgb.XGBoostDiamonds()

    # Preprocessing of the data inside the model. 
    # I opt for this decision because it will be useful later when we need to evaluate the model with different preprocessing methods.
    diamonds_processed = XGBModel.preprocessing(diamonds)
    
    # Splitting of the data
    x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(diamonds_processed.drop(columns='price'), 
                                                                            diamonds_processed['price'], 
                                                                            test_size=0.2, 
                                                                            random_state=42)
    
    
    
    # I left the user choose if we want to create the model with optimized hyperparameters or not.
    # y_pred = XGBModel.fit_predict_optimized(x_train_xbg, y_train_xbg, x_test_xbg, y_test_xbg, n_trials=100)
    y_pred = XGBModel.fit_predict(x_train_xbg, y_train_xbg, x_test_xbg)
    
    
    # Evaluation of the model
    results = {
        "MAE": mean_absolute_error(y_test_xbg, y_pred),
        "R2": r2_score(y_test_xbg, y_pred),
    }

    
    print("MAE: ", results["MAE"])
    print("R2: ", results["R2"])
    # Save the results in a csv file
    ut.save_results(results, XGBModel)