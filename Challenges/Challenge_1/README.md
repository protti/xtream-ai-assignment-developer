# Challenge 1 - Automated Pipeline for Linear Model Training

## Overview

In this challenge, I have developed an automated pipeline that trains a machine learning model using fresh data. In this challenge, I have focused on train only linear model and maintaining a history of all trained models along with their performance metrics.

## Objectives

1. **Automate Model Training**: Create a pipeline that automatically trains a linear model.
2. **Model History**: Maintain a history of all trained models and their performance metrics.


## Explanations

In this challenge, I've created a class named `LinearModel_Diamonds` designed to generalize the behavior of different linear models from `sklearn`. This class can instantiate any type of linear model defined in `sklearn`. To train the model, you need to provide a configuration file containing the path to the dataset, the type of model, and the training parameters.

Here is an example of a configuration JSON file:

```json
{
    "model_name": "LinearRegression",
    "test_size": 0.2,
    "dataset_path": "data/diamonds.csv"
}
```
The model names are specified in the `run.py` file in the `get_model` function.

``` python
def get_model(model_name):
    models = {
        "LinearRegression": LinearRegression(),
        "LogisticRegression": LogisticRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    return models.get(model_name, Lasso())  # Default to Lasso if model_name is not found
```

To add a new model, simply include the model's name and its instance in the `models` dictionary. 
This approach allows you to easily and quickly integrate new models into the pipeline.

The trained model is saved in the `results/models` directory, with the filename containing the model's name and the training date. 
The model's performance metrics are saved in the `results/performance` directory, also named with the model and training date.

## Additional Information

The code provided some operations of data preprocessing. Indeed, I have proposed some techniques of features engineering that you can find in the `features_analysis.py` file.

In this file, you can find different techniques for evaluating the features of the dataset, and it permits you to improve the performance of the model. Moreover, the implementation permits you to add more features analysis in the future. For the moment, I only adopt the following features analysis:

- **Remove Features with Low Variance**: This analysis permits you to evaluate the remove features with low variance.
- **Remove Features with High Correlation**: This analysis permits you to evaluate the remove features with high correlation.
- **Remove Outliers**: This analysis permits the removal of the outliers of the dataset.
- **RandomForest Feature Importance**: This analysis permits the evaluation of the importance of the features by using the RandomForest model (implemented but not used).


## How to run

1. **Run the pipeline**:

```bash
python run.py config.json
```

2. **Results**:

The results of the model training are saved in the `results/models_results.json` file, and the models are saved in the `results/models` directory.