import uuid

import pandas as pd
from flask import Flask, request, Response
from flask_restx import Resource, Api
from flask_restx import fields

from model_api.database import get_database
from models import get_model_type, load_model, ModelEnum

app = Flask(__name__)
api = Api(app, version='0.1.0', title='Model API',
          description='A simple example of ML models API')

types_output = api.model('Types output', {
    'types': fields.List(description="Possible types of models", cls_or_instance=fields.String)
})

info_model = api.model("metadata", {
    "id": fields.String(),
    "type": fields.String(),
    "params": fields.Wildcard(fields.String),

})

instances_output = api.model('Instances output', {
    'instances': fields.List(description="Available model instances", cls_or_instance=fields.Nested(info_model))
})

prediction_output = api.model('Prediction output', {
    'predict': fields.List(description="List of predictions", cls_or_instance=fields.Float)
})

model_name_output = api.model('Model name output', {
    'model_name': fields.String(description="Unique model ID")
})


@api.route('/model_list', doc={"description": "API for possible types of models"})
class ModelList(Resource):
    @api.marshal_with(types_output)
    def get(self):
        """
        Get all possible types of model to fit
        :return: model types
        """
        return {"types": [e.name for e in ModelEnum]}


@api.route('/model_instances', doc={"description": "API for all fitted models"})
class ModelInstances(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = get_database()

    @api.marshal_with(instances_output)
    def get(self):
        """
        Get info about all fitted models
        :return: fitted models info
        """
        return {"instances": self.db.get_models()}


@api.route('/model_retrain', doc={"description": "API for model retrain"})
class ModelRetrain(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = get_database()

    @api.marshal_with(model_name_output)
    @api.doc(params={'data': 'DataFrame with features in JSON format',
                     'target': 'Target list',
                     'model_name': 'unique model id to retrain'})
    def post(self):
        """
        Retrain model on new data with old parameters by model identifier
        :return: new unique model identifier
        """
        data = pd.DataFrame(request.json['data'])
        target = request.json['target']
        model_name = request.json["model_name"]
        model_dict = self.db.get_model(model_name)
        if model_dict is None:
            return Response(status=404)
        model_type = model_dict["type"]
        params = model_dict["params"]
        model_name = str(uuid.uuid4())
        clf = get_model_type(model_type)(params)
        clf.fit(data, target)
        self.db.create_model(model_name, model_type, params, clf.dumps())
        return {"model_name": model_name}


@api.route('/main', doc={"description": "API for main models operations"})
class ModelApi(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = get_database()

    @api.marshal_with(model_name_output)
    @api.doc(params={'data': 'DataFrame with features in JSON format',
                     'target': 'Target list',
                     'model_type': 'type of model to fit',
                     'params': 'parameters to fit model'})
    def post(self):
        """
        Fit model with parameters and data. Save model info to database
        :return: new unique model identifier
        """
        data = pd.DataFrame(request.json['data'])
        target = request.json['target']
        model_type = request.json["model_type"]
        params = request.json["params"]
        model_name = str(uuid.uuid4())
        clf = get_model_type(model_type)(params)
        clf.fit(data, target)
        self.db.create_model(model_name, model_type, params, clf.dumps())
        return {"model_name": model_name}

    @api.marshal_with(prediction_output)
    @api.doc(params={'data': 'DataFrame with features in JSON format',
                     'model_name': 'unique model id to get prediction'})
    def get(self):
        """
        Get model prediction on data by model identifier
        :return: predictions
        """
        data = pd.DataFrame(request.json['data'])
        model_name = request.json["model_name"]
        model_dict = self.db.get_model(model_name)
        if model_dict is None:
            return Response(status=404)
        clf = load_model(model_dict["type"], model_dict["binary"])

        return {"predict": list(clf.predict(data))}

    @api.doc(params={'model_name': 'unique model id to delete'})
    def delete(self):
        """
        Delete model from database by model identifier
        :return:
        """
        model_name = request.json["model_name"]
        self.db.delete_model(model_name)
        return Response(status=204)


if __name__ == '__main__':
    app.run(debug=True, port=8866, host='0.0.0.0')
