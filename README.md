# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---
# Overview

This repository contains a series of independent challenges, each with its own folder and README instructions. Each challenge addresses a specific aspect of the overall project, allowing you to understand the implementation of each component separately. 

You can find the solutions for each challenge in the folder [Challenges](Challenges).

The final challenge combines all previous challenges into a cohesive project. This README provides comprehensive instructions to run the final challenge, but I strongly recommend to read the README of each challenge to understand the implementation of each component separately.


## Description

This challenge involves developing an automated pipeline that trains models based on a diamond dataset. The pipeline should allow for training various types of models and create a REST API to integrate the model into a web app. Additionally, the pipeline should save every request and response made to the API for observability purposes.

## Instructions to Run the Final Challenge

This challenge combines all previous challenges. Here, I'll provide the instructions to run the final challenge. Each challenge is independent, with its own README containing specific instructions and explanations for each solution adopted. 


## How to Create a Model

To generate a model from the main folder, use the following command:

```bash
python Challenges/Challenge_4/run.py config_linear.json
```

Otherwise, I suggest you to go to the folder `Challenges/Challenge_4` and run the command:

```bash
python run.py config_linear.json
```

I have provided two configuration files in the folder `Challenges/Challenge_4` to generate the models with different hyperparameters. However, you can create your own configuration file to generate anytype of model with the hyperparameters you want, by following the instructions in the README of the folders [Challenge2/README.md](Challenges/Challenge_2/README.md).

### Example Configuration Files

- [config_linear.json](Challenges/Challenge_4/config_linear.json)
- [config_xgboost.json](Challenges/Challenge_4/config_xgboost.json)

Once the command is executed, the model is saved in the `Challenges/Challenge_4/results/models` folder.

## How to Run the API

To start the server, run the following command:

```bash
python rest_api_functions.py
```

This will start the server on http://127.0.0.1:5000/.

### API Endpoints


#### POST /predict-price

This endpoint is used to predict the value of a diamond.

Parameters:
- "carat" : float
- "cut" : str
- "color" : str
- "clarity" : str
- "depth" : float
- "table" : float
- "x" : float
- "y" : float
- "z" : float
- "path" : str

If any parameter does not meet the required constraints, an error with a status code of 400 is returned. The output is a JSON with a single predicted_value key, whose value corresponds to the float generated by the model.

#### POST /closer-diamond

This endpoint is used to find the closest samples to a diamond.

Parameters:
- "carat" : float
- "cut" : str
- "color" : str
- "clarity" : str
- "depth" : float
- "table" : float
- "x" : float
- "y" : float
- "z" : float
- "path" : str
- "n_neighbors" : int


If any parameter does not meet the required constraints, an error with a status code of 400 is returned. The output is a JSON with a single "closer" key, whose value corresponds to a list of similar samples generated by the model.

### How to Run the Tests

To run the tests, use the following command:

```bash
python -m unittest tests
```

This will run the tests and provide a report of the results.

If you want to run a specific test, use the following command:

```bash
python -m unittest tests/test_predict.py
```

I have also provided a web page to test the API. You can find it in the `Challenges/Challenge_4/templates/` folder.

## Observability

The request and response data are saved in the `database.db` file. The `observability` table has the following format: 

| Field       | Type     | Description                                      |
|-------------|----------|--------------------------------------------------|
| id          | Integer  | Primary key for the record                       |
| timestamp   | DateTime | Timestamp of the operation                       |
| method      | String   | HTTP method used for the request                 |
| model       | String   | Model used for the operation                     |
| type_request| String   | Type of request (e.g., PredictPrice, CloserDiamond) |
| request     | String   | Request data associated with the operation       |
| response    | String   | Response data associated with the operation      |


## Summary

1. Create Model 

```bash
python run.py config.json
```

2. Run the API

```bash
python rest_api_functions.py
```

3. Run the tests

```bash
python -m unittest tests
```

This setup ensures a robust, consistent, and observable workflow for training diamond price prediction models and integrating them into a web application.