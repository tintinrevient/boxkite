import pickle

from flask import Flask, request

from boxkite.monitoring.collector import BaselineMetricCollector
from boxkite.monitoring.service import ModelMonitoringService

print("Loading model and histogram from local files")
model_file = "./model.pkl"
histogram_file = "./histogram.txt"

with open(model_file, "rb") as f:
    model = pickle.load(f)

monitor = ModelMonitoringService(
    baseline_collector=BaselineMetricCollector(path=histogram_file)
)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    features = request.json
    score = model.predict([features])[0]
    pid = monitor.log_prediction(
        request_body=request.data,
        features=features,
        output=score,
    )
    return {"result": score, "prediction_id": pid}


@app.route("/metrics", methods=["GET"])
def metrics():
    return monitor.export_http()[0]


if __name__ == "__main__":
    app.run()
