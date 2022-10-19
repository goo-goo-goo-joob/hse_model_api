import abc
import tempfile

import catboost


class BaseModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def dumps(self) -> bytes:
        pass

    @staticmethod
    @abc.abstractmethod
    def loads(blob: bytes):
        pass


class CatBoostClassifierModel(BaseModel):
    def __init__(self, params: dict | None = None, obj=None):
        super().__init__()
        if obj is None:
            self.clf = catboost.CatBoostClassifier(**params)
        else:
            self.clf = obj

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
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


def get_model_type(model_type: str) -> type[BaseModel]:
    if model_type == "catboost_classifier":
        return CatBoostClassifierModel
    raise Exception(f"Unknown model {model_type}")


def load_model(model_type: str, blob: bytes) -> BaseModel:
    if model_type == "catboost_classifier":
        return CatBoostClassifierModel.loads(blob)
    raise Exception(f"Unknown model {model_type}")
