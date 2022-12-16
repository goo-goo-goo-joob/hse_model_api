import psycopg2

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error
import pytest

from model_api.models import CatBoostClassifierModel, CatBoostRegressorModel, ModelTypes
from model_api.service import Service
from model_api.database import DataBase


def test_simple():
    assert 42 + 69 == 111


@pytest.mark.parametrize("generator,cls,params,err", [
    (make_classification, CatBoostClassifierModel,
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}, 0.05),
    (make_regression, CatBoostRegressorModel,
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}, 400),

])
def test_model(generator, cls, params, err):
    X, y = generator(n_samples=10_000, random_state=42)
    model = cls(params)
    model.fit(X, y)
    proba = model.predict(X)
    assert mean_squared_error(y, proba) < err
    model2 = cls.loads(model.dumps())
    assert isinstance(model2, cls)
    proba2 = model2.predict(X)
    assert mean_squared_error(proba, proba2) * len(proba) < 10 ** -6


@pytest.mark.parametrize("generator,model_type,params", [
    (make_classification, "catboost_classifier",
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}),
    (make_regression, "catboost_regressor",
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}),

])
def test_db(mocker, generator, model_type, params):
    mocker.patch("psycopg2.connect")
    mocker.patch.object(DataBase, "create_model")
    mocker.patch.object(DataBase, "get_model", {"id": 0, "type": model_type, "params": params, "binary": b""})
    s = Service()
    psycopg2.connect.assert_called_once()
    X, y = generator(n_samples=10_000, random_state=42)
    s.model_train(X, y, model_type, params)
    DataBase.create_model.assert_called_once()
    assert set(s.get_model_list()) == set(ModelTypes.keys())
