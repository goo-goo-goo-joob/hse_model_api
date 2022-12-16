from model_api.models import CatBoostClassifierModel, CatBoostRegressorModel
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error
import pytest


def test_simple():
    assert 42 + 69 == 111


@pytest.mark.parametrize("generator,cls,params,err", [
    (make_classification, CatBoostClassifierModel,
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}, 0.05),
    (make_regression, CatBoostRegressorModel,
     {"iterations": 100, "max_depth": 3, "random_state": 42, "logging_level": "Silent"}, 400),

])
def test_classification(generator, cls, params, err):
    X, y = generator(n_samples=10_000, random_state=42)
    model = cls(params)
    model.fit(X, y)
    proba = model.predict(X)
    assert mean_squared_error(y, proba) < err
    model2 = cls.loads(model.dumps())
    assert isinstance(model2, cls)
    proba2 = model2.predict(X)
    assert mean_squared_error(proba, proba2) * len(proba) < 10 ** -6
