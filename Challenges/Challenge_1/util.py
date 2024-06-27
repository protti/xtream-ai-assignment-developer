import pandas as pd
import pickle
import time
import os
import csv

def load_preprocess_dataset(path: str = None):
    """
    Load and preprocess the diamonds dataset.

    Parameters:
    path (str): The path to the CSV file containing the diamonds dataset.

    Returns:
    pd.DataFrame: The preprocessed diamonds DataFrame.
    """
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

def save_results(results, model):
    """
    Save the model and its results to the filesystem.

    Parameters:
    results (dict): A dictionary containing the model evaluation metrics (MAE and R2).
    model: The trained model to be saved.

    Returns:
    None
    """
    # Get the path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the 'results' directory if it doesn't exist
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        print(f"Creating directory: {results_dir}")
        os.makedirs(results_dir)
    
    # Create the 'models' directory inside 'results' if it doesn't exist
    model_dir = os.path.join(results_dir, 'models')
    if not os.path.exists(model_dir):
        print(f"Creating directory: {model_dir}")
        os.makedirs(model_dir)
    
    name_path = model.save_model(model_dir)
      
    # Path of the CSV file
    csv_file_path = os.path.join(results_dir, 'model_results.csv')

    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Write the results to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'MAE', 'R2'])
        writer.writerow([name_path, results["MAE"], results["R2"]])

