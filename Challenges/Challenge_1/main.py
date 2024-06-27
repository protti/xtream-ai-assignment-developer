import util as ut
import features_analysis as fa
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from LinearModel_Diamonds import LinearModelDiamonds
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

if __name__ == "__main__":

    # Load the dataset
    path = "data/diamonds.csv"
    diamonds = ut.load_preprocess_dataset(path)
    LinearModel = LinearModelDiamonds(Lasso())
    diamonds = LinearModel.preprocessing(diamonds)

    x = diamonds.drop(columns='price')
    y = diamonds.price
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pred = LinearModel.fit_predict(x_train, y_train, x_test)
    
    # Evaluation of the model
    results = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

    
    print("MAE: ", results["MAE"])
    print("R2: ", results["R2"])
    # Save the results in a csv file
    ut.save_results(results, LinearModel)