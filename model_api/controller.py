import pandas as pd
from flask import Flask, Response, request
from flask_restx import Api, Resource, fields
from model_api.service import Service, NotFoundError

app = Flask(__name__)
api = Api(app, version="0.1.0", title="Model API", description="A simple example of ML models API")

service = Service()

types_output = api.model(
    "Types output", {"types": fields.List(description="Possible types of models", cls_or_instance=fields.String)}
)

info_model = api.model(
    "Model info",
    {
        "id": fields.String(),
        "type": fields.String(),
        "params": fields.Wildcard(fields.String),
    },
)

instances_output = api.model(
    "Instances output",
    {"instances": fields.List(description="Available model instances", cls_or_instance=fields.Nested(info_model))},
)

prediction_output = api.model(
    "Prediction output", {"predict": fields.List(description="List of predictions", cls_or_instance=fields.Float)}
)

model_name_output = api.model("Model name output", {"model_name": fields.String(description="Unique model ID")})
error_output = api.model("Error output", {"message": fields.String(description="Error message")})

model_retrain_input = api.model(
    "Model retrain input",
    {
        "data": fields.List(
            description="DataFrame with features in JSON format", cls_or_instance=fields.Wildcard(fields.String)
        ),
        "target": fields.List(description="Target list", cls_or_instance=fields.Float),
        "model_name": fields.String(description="Unique model id to retrain"),
    },
)

model_fit_input = api.model(
    "Model fit input",
    {
        "data": fields.List(
            description="DataFrame with features in JSON format", cls_or_instance=fields.Wildcard(fields.String)
        ),
        "target": fields.List(description="Target list", cls_or_instance=fields.Float),
        "model_type": fields.String(description="Type of model to fit"),
        "params": fields.Wildcard(description="Parameters to fit model", cls_or_instance=fields.String),
    },
)

model_predict_input = api.model(
    "Model predict input",
    {
        "data": fields.List(
            description="DataFrame with features in JSON format", cls_or_instance=fields.Wildcard(fields.String)
        ),
        "model_name": fields.String(description="Unique model id to get prediction"),
    },
)

model_delete_input = api.model(
    "Model delete input", {"model_name": fields.String(description="Unique model id to delete")}
)


@api.route("/model_list", doc={"description": "API for possible types of models"})
class ModelList(Resource):
    @api.marshal_with(types_output)
    def get(self):
        """
        Get all possible types of model to fit
        :return: model types
        """
        return {"types": service.get_model_list()}


@api.route("/model_instances", doc={"description": "API for all fitted models"})
class ModelInstances(Resource):
    @api.marshal_with(instances_output)
    def get(self):
        """
        Get info about all fitted models
        :return: fitted models info
        """
        return {"instances": service.get_model_instances()}


@api.route("/model_retrain", doc={"description": "API for model retrain"})
class ModelRetrain(Resource):
    @api.response(200, "Successfully retrained", model_name_output)
    @api.response(404, "Not found error", error_output)
    @api.expect(model_retrain_input)
    def post(self):
        """
        Retrain model on new data with old parameters by model identifier
        :return: new unique model identifier
        """
        data = pd.DataFrame(request.json["data"])
        target = request.json["target"]
        model_name = request.json["model_name"]
        try:
            model_name = service.post_model_retrain(data, target, model_name)
        except NotFoundError as e:
            return {"message": str(e)}, 404
        return {"model_name": model_name}


@api.route("/model", doc={"description": "API for main models operations"})
class ModelApi(Resource):
    @api.response(200, "Successfully trained", model_name_output)
    @api.response(404, "Not found error", error_output)
    @api.expect(model_fit_input)
    def post(self):
        """
        Fit model with parameters and data. Save model info to database
        :return: new unique model identifier
        """
        data = pd.DataFrame(request.json["data"])
        target = request.json["target"]
        model_type = request.json["model_type"]
        params = request.json["params"]
        try:
            model_name = service.post_model(data, target, model_type, params)
        except NotFoundError as e:
            return {"message": str(e)}, 404
        return {"model_name": model_name}

    @api.response(200, "Successfully retrained", prediction_output)
    @api.response(404, "Not found error", error_output)
    @api.expect(model_predict_input)
    def get(self):
        """
        Get model prediction on data by model identifier
        :return: predictions
        """
        data = pd.DataFrame(request.json["data"])
        model_name = request.json["model_name"]
        try:
            predict = service.get_model(data, model_name)
        except NotFoundError as e:
            return {"message": str(e)}, 404
        return {"predict": predict}

    @api.response(204, "Deleted")
    @api.expect(model_delete_input)
    def delete(self):
        """
        Delete model from database by model identifier
        :return:
        """
        model_name = request.json["model_name"]
        service.delete_model(model_name)
        return Response(status=204)
