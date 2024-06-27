import pandas as pd
import numpy as np
from scipy import stats


def load_preprocess_dataset(path: str = None):

    diamonds = pd.read_csv(path)
    diamonds = diamonds.dropna()
    # Removing all the rows that do not respect the rules 
    diamonds = diamonds.drop(diamonds[(diamonds.x * diamonds.y * diamonds.z == 0) | 
                                      (diamonds.price <= 0) | 
                                      (diamonds.carat <= 0) | 
                                      (diamonds.depth <= 0) | 
                                      (diamonds.table <= 0)].index)
    diamonds = diamonds[diamonds['color'].between('D', 'Z')]
    valid_clarity_values = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
    diamonds = diamonds[diamonds['clarity'].isin(valid_clarity_values)]
    # Apply the categorical columns encoding
    
    return diamonds

