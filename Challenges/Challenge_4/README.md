# Challenge 4 - Associate a database for saving all the requests made to the API

## Overview

In this challenge, I associate a database to save all the requests made to the API. I have created a table named `observability` that will save all the requests made to the API.

## Objectives

1. **Create REST API**: The REST_API structure permits to predict the diamond prices using a trained model, as well as retrieving diamonds with characteristics closest to those provided as input.
2. **Database association**: Associate a database to save all the requests made to the API.
3. **Adopt the Pre-Trained Model**: Adopt the pretrained model for obtaining the predictions of the diamond prices.
4. **Retrieve Diamonds**: Retrieve the diamonds with characteristics closest to those provided as input.


## Explanations

In this challenge, I have created a new class named Observability that permits the management of all the requests made to the API. Indeed, I will create a new table named `observability` with this format: 

| Field       | Type     | Description                                      |
|-------------|----------|--------------------------------------------------|
| id          | Integer  | Primary key for the record                       |
| timestamp   | DateTime | Timestamp of the operation                       |
| method      | String   | HTTP method used for the request                 |
| model       | String   | Model used for the operation                     |
| type_request| String   | Type of request (e.g., PredictPrice, CloserDiamond) |
| request     | String   | Request data associated with the operation       |
| response    | String   | Response data associated with the operation      |

This permits to keep update the table with all the requests made to the API. This operation could be improved by creating a table for each of the request made to the API, but for the sake of simplicity, I have decided to keep the table `observability` updated with all the requests made to the API.

For this operation, I have used the `sqlite3` library to create the database and the table `observability`. Moreover, I have updated the class of test by checking if the table `observability` is updated with all the requests made to the API and if it correctly saved the data.

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