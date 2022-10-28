import uuid

import pandas as pd

from model_api.models import ModelTypes, get_model_type, load_model
from model_api.database import DataBase


class NotFoundError(Exception):
    pass


class Service:
    def __init__(self):
        self.db = DataBase()

    def get_model_list(self) -> list:
        return list(ModelTypes.keys())

    def get_model_instances(self) -> list[dict]:
        return self.db.get_models()

    def post_model_retrain(self, data: pd.DataFrame, target: list, model_name: str) -> str:
        model_dict = self.db.get_model(model_name)
        if model_dict is None:
            raise NotFoundError("Model not found")
        model_type = model_dict["type"]
        params = model_dict["params"]
        model_name = str(uuid.uuid4())
        clf = get_model_type(model_type)
        if clf is None:
            raise NotFoundError("Model type not found")
        clf = clf(params)
        clf.fit(data, target)
        self.db.create_model(model_name, model_type, params, clf.dumps())
        return model_name

    def post_model(self, data: pd.DataFrame, target: list, model_type: str, params: dict) -> str:
        model_name = str(uuid.uuid4())
        clf = get_model_type(model_type)
        if clf is None:
            raise NotFoundError("Model type not found")
        clf = clf(params)
        clf.fit(data, target)
        self.db.create_model(model_name, model_type, params, clf.dumps())
        return model_name

    def get_model(self, data: pd.DataFrame, model_name: str) -> list:
        model_dict = self.db.get_model(model_name)
        if model_dict is None:
            raise NotFoundError("Model not found")
        clf = load_model(model_dict["type"], model_dict["binary"])
        if clf is None:
            raise NotFoundError("Model type not found")
        predict = list(clf.predict(data))
        return predict

    def delete_model(self, model_name: str):
        self.db.delete_model(model_name)
