from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import joblib
import numpy as np
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)
model = joblib.load("src/model.joblib")
VERSION = os.environ.get("MODEL_VERSION", "unknown")

@app.route("/predict", methods=["POST"])
def predict():
    features = np.array(request.get_json()["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0]), "version": VERSION})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
