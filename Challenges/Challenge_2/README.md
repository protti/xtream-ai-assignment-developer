# Challenge 2 - Automated Pipeline for Any Model Training

## Overview

In this challenge, I focused on creating a pipeline that can train various types of models. Trying to make the code as general as possible, I have created a general model named `BaseModel_Diamond` that implement the basic behavior of any model (preprocessing, train, predict, evaluate, save the model).


## Objectives

1. **Create General Model**: Create a general model that can train any type of model.
2. **Automate Model Training**: Create a pipeline that automatically trains any type of model using new data.
3. **Model History**: Maintain a history of all trained models and their performance metrics.


## Explanations

The idea is to create a general class named `BaseModel_Diamond` that implements the basic behavior of any model, including preprocessing, training, predicting, evaluating, and saving the model. Then, we can create specific model classes by simply inheriting from `BaseModel_Diamond` and overriding the `fit`, `predict`, `score`, and `save` methods.

This approach ensures that the `run.py` file remains consistent across all models. The only changes required are in the configuration file and the `get_model` function.


Here is an example of a configuration JSON file:

```json
{
    "data_path": "../../data/diamonds.csv",
    "model_to_use": "XGBModel",
    "use_optimized": false,
    "enable_categorical": true,
    "test_size": 0.2,
    "random_state": 42
}
```
Another power of the adoption of the class is that the analysis of the input parameters is done in the specified class; for example, in the `XGBoost_Diamond` class, we have added the `use_optimized` parameter that permits the use of the optimized parameters of the XGBoost model.

The only changes need to be made in the `run.py` file is in the `get_model` function.

``` python
def get_model(config):
    """
    Get the model specified in the configuration.

    Parameters:
    config (dict): Configuration dictionary specifying the model to use and its parameters.

    Returns:
    object: An instance of the specified model.
    """
    model_to_use = config["model_to_use"]
    if model_to_use == "LinearModel":
        linear_model_type = config["linear_model_type"]
        if linear_model_type == "Lasso":
            return LinearModelDiamonds(Lasso())
        elif linear_model_type == "LinearRegression":
            return LinearModelDiamonds(LinearRegression())
        else:
            raise ValueError(f"Unknown linear model type: {linear_model_type}")
    elif model_to_use == "XGBModel":
        return xgb.XGBoostDiamonds(optimized_params=config["use_optimized"])
    else:
        raise ValueError(f"Unknown model: {model_to_use}")
```
This function allows for the flexible creation of different models by specifying the model type and parameters in the configuration file. You can easily add new models, such as `RandomForest_Diamond` or `CNN_Diamond`, by adding a new elif condition and implementing the necessary methods in the corresponding class.

## How to run

1. **Run the pipeline**:

```bash
python run.py config.json
```

2. **Results**:

The results of the model training are saved in the `results/models_results.json` file, and the models are saved in the `results/models` directory.