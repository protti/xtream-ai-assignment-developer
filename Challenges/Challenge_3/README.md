# Challenge 3 - API for Model Training

## Overview

In this challenge, I focused on creating a REST API structure for predicting diamond prices using a trained model, as well as retrieving diamonds with characteristics closest to those provided as input.


## Objectives

1. **Create REST API**: The REST_API structure permits to predict the diamond prices using a trained model, as well as retrieving diamonds with characteristics closest to those provided as input.
2. **Adopt the Pre-Trained Model**: Adopt the pretrained model for obtaining the predictions of the diamond prices.
3. **Retrieve Diamonds**: Retrieve the diamonds with characteristics closest to those provided as input.


## Explanations

In this challenge, I have created a REST API structure for predicting diamond prices using a trained model, as well as retrieving diamonds with characteristics closest to those provided as input. 

I have adopted the FLASK framework for the implementation of the API. The API is composed of two main classes:

- **ObtainCloser.py**: This class permits to retrieve the diamonds with characteristics closest to those provided as input. 
- **PredictPrice.py**: This class permits the prediction of the price of a diamond based on its characteristics.

Both the classes have two main methods:

- **load_model**: This method permits the loading of the pre-trained model (the model should be trained before; there is not yet an interface to train the model).
- **post**: a method for receiving the input data and returning the predictions of the diamond prices.

If the input data is incorrect, the API will return a message indicating the error.

The folder `tests` contains the unit tests for the classes `ObtainCloser` and `PredictPrice`. 
For the tests, I have used the `unittest` framework. Therefore you need to run the following command to test each classes:

```bash
python -m unittest tests/test_ObtainCloser.py
python -m unittest tests/test_PredictPrice.py
```

or you can run the following command to test all the classes:

```bash
python -m unittest discover -s tests
```
Moreover, I provide two web pages in templates folder:

- **[predict_price.html](templates/predict_price.html)**: This web page permits to input the characteristics of the diamonds to predict the price.
- **[find_closer.html](templates/find_closer.html)**: This web page permits to input the characteristics of the diamonds to retrieve the diamonds with characteristics closest to those provided as input.


## How to run

1. **Run the pipeline and create the models as the previous challenge**:

```bash
python run.py config.json
```

2. **Run the server**:

```bash
python rest_api_function.py
```

2. **Test the API**:
```bash
python -m unittest discover -s tests
```