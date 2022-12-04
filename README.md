# hse_model_api

## Description
A simple example of ML models API

REST API for
- learning ML models with passed hyperparameters (CatBoost for regression and classification)
- methods to get allowed types of models and fitted instances on models
- getting prediction of fitted models (models are stored in the database)
- refitting model with new dataset on old hyperparameters
- deleting models from database

Swagger documentation is available

Postgresql is used

## Run app
```
pip install poetry
poetry install
PYTHONPATH='.' python model_api/__main__.py --port 8866 --host '0.0.0.0'
```
or
```
make build && docker-compose up 
```

## Code style check
```
make linters
```

## Docker image
https://hub.docker.com/r/googoogoojoob/model_api

## Demo
API demo [notebook](api_test.ipynb) 
