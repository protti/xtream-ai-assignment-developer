import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Removes highly correlated features from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    threshold (float): The correlation threshold above which features are considered highly correlated. Default is 0.9.

    Returns:
    pd.DataFrame: The DataFrame with highly correlated features removed.
    """
    corr_matrix = df.drop(columns=['price']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def remove_features_with_low_variance(diamonds: pd.DataFrame, variance_threshold: float = 0.01) -> pd.DataFrame:
    """
    Removes features with low variance from the DataFrame.

    Parameters:
    diamonds (pd.DataFrame): The input DataFrame containing the features.
    variance_threshold (float): The variance threshold below which features are considered to have low variance. Default is 0.01.

    Returns:
    pd.DataFrame: The DataFrame with low variance features removed.
    """
    low_variance_cols = diamonds.var(axis=0) < variance_threshold
    diamonds = diamonds.drop(columns=diamonds.columns[low_variance_cols])
    return diamonds

def remove_outliers(diamonds: pd.DataFrame, z_score_threshold: int = 3) -> pd.DataFrame:
    """
    Removes outliers from the DataFrame based on z-scores.

    Parameters:
    diamonds (pd.DataFrame): The input DataFrame containing the features.
    z_score_threshold (int): The z-score threshold above which data points are considered outliers. Default is 3.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    """
    z_scores = np.abs(zscore(diamonds.select_dtypes(include=[np.number])))
    diamonds = diamonds[(z_scores < z_score_threshold).all(axis=1)]
    return diamonds

def remove_unimportant_features(df: pd.DataFrame, target: str, threshold: float = 0.01) -> pd.DataFrame:
    """
    Removes unimportant features from the DataFrame based on feature importance from a RandomForestClassifier.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    target (str): The target column name.
    threshold (float): The importance threshold below which features are considered unimportant. Default is 0.01.

    Returns:
    pd.DataFrame: The DataFrame with unimportant features removed.
    """
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    print(importances)
    important_features = [feature for feature, importance in zip(X.columns, importances) if importance > threshold]
    return df[important_features + [target]]

def features_evaluation(diamonds: pd.DataFrame, variance_threshold: float = 0.01, correlation_threshold: float = 0.9, z_score_threshold: int = 3) -> pd.DataFrame:
    """
    Evaluates and selects important features from the DataFrame.

    Parameters:
    diamonds (pd.DataFrame): The input DataFrame containing the features.
    variance_threshold (float): The variance threshold below which features are considered to have low variance. Default is 0.01.
    correlation_threshold (float): The correlation threshold above which features are considered highly correlated. Default is 0.9.
    z_score_threshold (int): The z-score threshold above which data points are considered outliers. Default is 3.
    

    Returns:
    pd.DataFrame: The DataFrame with selected important features.
    """
    # Categorical Data Encoding
    diamonds_copy = diamonds.copy()
    label_encoders = {}
    categorical_columns = diamonds_copy.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        diamonds_copy[col] = le.fit_transform(diamonds_copy[col])
        label_encoders[col] = le

    # Features Selection
    diamonds_copy = remove_outliers(diamonds_copy, z_score_threshold)
    diamonds_copy = remove_features_with_low_variance(diamonds_copy, variance_threshold)
    diamonds_copy = remove_highly_correlated_features(diamonds_copy, correlation_threshold)
    
    # Selected Features
    selected_features = diamonds_copy.columns
    diamonds = diamonds[selected_features]

    return diamonds
