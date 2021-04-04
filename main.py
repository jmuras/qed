import sys

import flask
from flask import request, jsonify

from Model import Model
from Status import Status

app = flask.Flask(__name__)
app.config["DEBUG"] = False

# Trained model ready for predictions
current_model = None
# Model in training
model_in_training = None

"""
Callback function invoked whem model finished training
"""
def model_ready(model):

    global current_model

    if model is not None and model.status == Status.READY:
        current_model = model

@app.route('/status', methods=['GET'])
def status():

    global model_in_training

    # No model trained
    if model_in_training is None:
        response = {
            "status": "no model"
        }
    # Model training is running
    elif model_in_training.status == Status.TRAINING:
        response = {
            "status": model_in_training.status.value,
            "start_time": str(model_in_training.start_time)
        }
    # Errors occurred during model training
    elif model_in_training.status == Status.FAULT:
        response = {
            "status": model_in_training.status.value,
            "start_time": str(model_in_training.start_time),
            "finish_time": str(model_in_training.finish_time),
            "errors": model_in_training.get_exceptions()
        }
    # Model trained and ready for prediction
    else:
        response = {
            "status": model_in_training.status.value,
            "start_time": str(model_in_training.start_time),
            "finish_time": str(model_in_training.finish_time)
        }

    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():

    global current_model

    # Get objects data
    objects = request.get_json()

    if current_model is not None:
        return jsonify(current_model.predict(objects))

    raise Exception("No model")


@app.route('/train/<int:l>', methods=['POST'])
def train(l=1):

    global model_in_training

    # Get 'k' argument
    k = None
    if 'k' in request.args:
        try:
            k = int(request.args['k'])
        except:
            print("Problem in reading k parameter")

    # Get objects data
    objects = request.get_json()

    # Check if model is not being trained
    if model_in_training is None or model_in_training.status != Status.TRAINING:
        # If model not being trained start new training
        model_in_training = Model(objects, l, k, model_ready)
        response = {
            "status": model_in_training.status.value
        }
        model_in_training.start()
        return jsonify(response)

    # Model is in training
    response = {
        "status": model_in_training.status.value,
        "start_time": str(model_in_training.start_time)
    }
    return jsonify(response)

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    status_code = 500
    response = {
        'error': {
            'type': 'UnexpectedException',
            'message': 'An unexpected error has occurred.',
            'description': str(sys.exc_info())
        }
    }
    return jsonify(response), status_code

if __name__ == "__main__":
    app.run(host='0.0.0.0')
