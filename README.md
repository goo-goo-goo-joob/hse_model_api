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

## Install dependencies
```
pip install poetry
poetry install
```

## Run app
```
PYTHONPATH='.' python model_api/__main__.py 8866 '0.0.0.0'
```

## Code style check
```
make linters
```

## Demo
API demo [notebook](api_test.ipynb) 
