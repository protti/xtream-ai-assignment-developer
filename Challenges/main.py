import util as ut
import features_analysis as fa
from pandas.plotting import scatter_matrix


if __name__ == "__main__":
    path = "data/diamonds.csv"
    diamonds = ut.load_preprocess_dataset(path)
    diamonds = fa.features_evaluation(diamonds, 'price')
    

    print(diamonds.head())
    # scatter_matrix(diamonds.select_dtypes(include=['number']), figsize=(14, 10));
    # ut.features_evaluation(diamonds)
    