import pandas as pd
import os
import csv
import json

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
    invalid_conditions = (
        (diamonds.x * diamonds.y * diamonds.z == 0) | 
        (diamonds.price <= 0) | 
        (diamonds.carat <= 0) | 
        (diamonds.depth <= 0) | 
        (diamonds.table <= 0)
    )
    diamonds = diamonds.drop(diamonds[invalid_conditions].index)
    
    diamonds = diamonds[diamonds['color'].between('D', 'Z')]
    
    valid_clarity_values = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
    diamonds = diamonds[diamonds['clarity'].isin(valid_clarity_values)]
    
    # Apply the categorical columns encoding
    
    return diamonds

def save_results(results, model, config):
    """
    Save the model and its results to the filesystem.

    Parameters:
    results (dict): A dictionary containing the model evaluation metrics (MAE and R2).
    model: The trained model to be saved.

    Returns:
    None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the 'results' directory if it doesn't exist
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create the 'models' directory inside 'results' if it doesn't exist
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    name_path = model.save_model(model_dir)
  
    # Path of the JSON file
    json_file_path = os.path.join(results_dir, 'model_results.json')

    # Save results to JSON
    _save_to_json(json_file_path, config, results, name_path)

def _save_to_json(json_file_path, config, results, name_path):
    """
    Save the results to a JSON file.

    Parameters:
    json_file_path (str): The path to the JSON file.
    config (dict): The configuration dictionary.
    results (dict): The results dictionary.
    name_path (str): The path where the model is saved.

    Returns:
    None
    """
    # Leggi il contenuto esistente del file JSON
    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Aggiungi le nuove informazioni
    config['name'] = name_path
    config.update(results)
    data.append(config)

    # Scrivi il contenuto aggiornato nel file JSON
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)
