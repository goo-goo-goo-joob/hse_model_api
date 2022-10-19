import glob
import os
import uuid

import pandas as pd
from flask import Flask, request
from flask_restx import Resource, Api
from flask_restx import fields, reqparse

from model_api.database import get_database
from models import CatBoostClassifierModel, get_model_type, load_model

app = Flask(__name__)
api = Api(app)

# prediction_output = api.model('Prediction output', {
#     'predictions': fields.List(description="List of predictions for input data")
# })
#
# prediction_input = api.model('Prediction input', {
#     'model ID': fields.String(description="ID of fitted model", requered=True, location='args'),
#     'data': fields.(description="Some model unique id", requered=True, location='json'),
# })

# parser = reqparse.RequestParser()
# parser.add_argument('name', type=str, help='Rate cannot be converted', location='args')
# parser.add_argument('name2', type=str, help='123', location='json', requered=True)



@api.route('/hello')
class HelloWorld(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = get_database()

    # @api.marshal_with(prediction_output, as_list=True)
    # @api.expect(resource_fields)
    # @api.expect(parser)
    def post(self):
        """Some docstring"""
        # args = parser.parse_args()
        data = pd.DataFrame(request.json['data'])
        target = pd.DataFrame(request.json['target'])
        model_type = request.json["model_type"]
        params = request.json["params"]
        model_name = str(uuid.uuid4())
        clf = get_model_type(model_type)(params)
        clf.fit(data, target)
        self.db.create_model(model_name, model_type, params, clf.dumps())
        return {"model_name": model_name}

    def get(self):
        data = pd.DataFrame(request.json['data'])
        model_name = request.json["model_name"]
        model_dict = self.db.get_model(model_name)

        clf = load_model(model_dict["type"], model_dict["binary"])

        return {"predict": list(clf.predict(data))}


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True, port=8866, host='0.0.0.0')
