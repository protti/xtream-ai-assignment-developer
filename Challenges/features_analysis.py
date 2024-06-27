import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9):
    corr_matrix = df.drop(columns=['price']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def remove_features_with_low_variance(diamonds: pd.DataFrame, variance_threshold: float = 0.01):
    low_variance_cols = diamonds.var(axis=0) < variance_threshold
    diamonds = diamonds.drop(columns=diamonds.columns[low_variance_cols])
    return diamonds


def remove_outliers(diamonds: pd.DataFrame, z_score_threshold: int = 3):
    z_scores = np.abs(zscore(diamonds.select_dtypes(include=[np.number])))
    diamonds = diamonds[(z_scores < z_score_threshold).all(axis=1)]
    return diamonds


def remove_unimportant_features(df, target, threshold=0.01):
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    print(importances)
    important_features = [feature for feature, importance in zip(X.columns, importances) if importance > threshold]
    return df[important_features + [target]]


def features_evaluation(diamonds: pd.DataFrame, target_columns: str, variance_threshold: float = 0.01, correlation_threshold: float = 0.9, z_score_threshold: int = 3, random_forest_threshold: float = 0.01):
    diamonds_copy = diamonds.copy()
    label_encoders = {}
    categorical_columns = diamonds_copy.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        diamonds_copy[col] = le.fit_transform(diamonds_copy[col])
        label_encoders[col] = le

    diamonds_copy = remove_outliers(diamonds_copy, z_score_threshold)
    diamonds_copy = remove_features_with_low_variance(diamonds_copy, variance_threshold)
    diamonds_copy = remove_highly_correlated_features(diamonds_copy, correlation_threshold)
    selected_features = diamonds_copy.columns
    diamonds = diamonds[selected_features]
    return diamonds
