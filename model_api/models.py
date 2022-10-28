import abc
import tempfile
import typing

import catboost
import pandas as pd


class BaseModel(abc.ABC):
    """
    Abstract class for base model
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: list):
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    @abc.abstractmethod
    def dumps(self) -> bytes:
        pass

    @staticmethod
    @abc.abstractmethod
    def loads(blob: bytes):
        pass


class CatBoostClassifierModel(BaseModel):
    """
    CatBoost model for classification task
    """

    def __init__(self, params: dict | None = None, obj=None):
        super().__init__()
        if obj is None:
            self.clf = catboost.CatBoostClassifier(**params)
        else:
            self.clf = obj

    def fit(self, X: pd.DataFrame, y: list):
        self.clf.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.clf.predict_proba(X)[:, 1]

    def dumps(self) -> bytes:
        with tempfile.NamedTemporaryFile() as t:
            self.clf.save_model(t.name)
            t.seek(0)
            return t.read()

    @staticmethod
    def loads(blob: bytes):
        clf = catboost.CatBoostClassifier()
        clf.load_model(blob=blob)
        return CatBoostClassifierModel(obj=clf)


class CatBoostRegressorModel(BaseModel):
    """
    CatBoost model for regression task
    """

    def __init__(self, params: dict | None = None, obj=None):
        super().__init__()
        if obj is None:
            self.reg = catboost.CatBoostRegressor(**params)
        else:
            self.reg = obj

    def fit(self, X: pd.DataFrame, y: list):
        self.reg.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.reg.predict(X)

    def dumps(self) -> bytes:
        with tempfile.NamedTemporaryFile() as t:
            self.reg.save_model(t.name)
            t.seek(0)
            return t.read()

    @staticmethod
    def loads(blob: bytes):
        reg = catboost.CatBoostRegressor()
        reg.load_model(blob=blob)
        return CatBoostRegressorModel(obj=reg)


# Possible model types and respective classes
ModelTypes = {
    'catboost_classifier': CatBoostClassifierModel,
    'catboost_regressor': CatBoostRegressorModel
}


def get_model_type(model_type: str) -> typing.Optional[type[BaseModel]]:
    """
    Get class of model by its type
    :param model_type: type of model
    :return: class of model
    """
    mt = ModelTypes.get(model_type)
    return mt


def load_model(model_type: str, blob: bytes) -> typing.Optional[BaseModel]:
    """
    Load model binary from its name
    :param model_type: type of model
    :param blob: binary representation of model
    :return: loaded (fitted) model of certain type
    """
    mt = ModelTypes.get(model_type)
    if mt is None:
        return None
    return mt.loads(blob)
