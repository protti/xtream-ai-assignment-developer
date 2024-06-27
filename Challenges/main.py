import util as ut
import features_analysis as fa
import XGBoost_Diamonds as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

if __name__ == "__main__":
    path = "data/diamonds.csv"
    diamonds = ut.load_preprocess_dataset(path)
    diamonds = fa.features_evaluation(diamonds, 'price')
    
    XGBModel = xgb.XGBoostDiamonds(diamonds, 'price')
    diamonds_processed = XGBModel.preprocessing()
    x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(diamonds_processed.drop(columns='price'), 
                                                                            diamonds_processed['price'], 
                                                                            test_size=0.2, 
                                                                            random_state=42)
    
    y_pred = XGBModel.fit_predict(x_train_xbg, y_train_xbg, x_test_xbg)
    print(mean_absolute_error(y_test_xbg, y_pred))
    print(r2_score(y_test_xbg, y_pred))   
    # print(diamonds_processed.head())
    # scatter_matrix(diamonds.select_dtypes(include=['number']), figsize=(14, 10));
    # ut.features_evaluation(diamonds)
    