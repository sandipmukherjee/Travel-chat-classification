# -*- coding: utf-8 -*-
import json
import logging
import os
import argparse
from flask import request, jsonify

from cattr import structure, unstructure
from flask import Flask
from errors import EmptyTextException, NonAlphabeticTextException, APIException, \
    BODYMissingException, InternalServerError
from intent_classifier import IntentClassifier, IntentRequest, IntentResponse

logger = logging.getLogger(__name__)

app = Flask(__name__)
model: IntentClassifier = None


@app.errorhandler(InternalServerError)
def handle_500(e):
    response = {
        "message": str(e),
        "label": "INTERNAL_SERVER_ERROR"
    }
    return jsonify(response), 500


@app.errorhandler(APIException)
def handle_exception(err):
    """Return JSON instead of HTML for MyCustomError errors."""
    response = {
      "message": err.message,
      "label": err.label
    }
    return jsonify(response), err.code


@app.route('/ready')
def ready():
    if model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent():
    if not request.data:
        raise BODYMissingException
    intent_request = structure(json.loads(request.data), IntentRequest)
    if not intent_request.text:
        raise EmptyTextException
    if not isinstance(request.json["text"], str):
        raise NonAlphabeticTextException
    try:
        prediction = model.predict(intent_request, n_preds=3)
    except Exception as e:
        raise InternalServerError(e)
    return json.dumps(unstructure(prediction))


def _init_model(model_path: str):
    global model
    if not model:
        model = IntentClassifier(model_path)
    logger.info("Trying sample text")
    model.predict(IntentRequest(text="I want a flight to Boston"), n_preds=3)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')
    args = arg_parser.parse_args()
    _init_model(args.model)
    app.run(port=args.port)


if __name__ == '__main__':
    main()
