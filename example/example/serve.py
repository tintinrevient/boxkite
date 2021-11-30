from flask import Flask, request, jsonify
import mlflow
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import pandas as pd

# Option 1 - MLFlow Gunicorn to serve the model
# mlflow models serve -m ./mlruns/[experiment_id]/[run_id]/artifacts/model -h 0.0.0.0 -p 5001

# export MLFLOW_TRACKING_URI=http://localhost:5000
# mlflow models serve -m "models:/[model_name]/[model_version]" -h 0.0.0.0 -p 5001

# Option 2 - Flask to serve the model
from boxkite.monitoring.service import ModelMonitoringService
from boxkite.monitoring.collector.baseline import BaselineMetricCollector

# https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html
# Option 2.1 - DB
tracking_uri = "postgresql://mlflow:mlflow@127.0.0.1:5432/mlflow"
# Option 2.2 - REST API
# tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

# Basic setting
histogram_path = "histogram"
run_id = '40d1aab251bb4342ac89e7f8886f7af2'
model_name = "regression"
model_version = 7

# Load the artifact registry - depends on mlflow.set_tracking_uri()
runs_uri = f"runs:/{run_id}/{histogram_path}"
histogram_dir = RunsArtifactRepository.get_underlying_uri(runs_uri)
print('hist_dir: {}'.format(histogram_dir))
hist_path = f'{histogram_dir}/histogram.txt'.replace('file://', '')

monitor = ModelMonitoringService(
    baseline_collector=BaselineMetricCollector(path=hist_path)
)

app = Flask(__name__)

@app.route("/invocations", methods=["POST"])
def predict():

    features = request.json
    # print(features)
    df = pd.DataFrame(columns=features["columns"], data=features["data"])
    print(df)

    # Load the model - depends on mlflow.set_tracking_uri()
    lr = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

    score = lr.predict(df)[0]

    pid = monitor.log_prediction(
        request_body=request.data,
        features=[df],
        output=score
    )

    return f"Score is {score}\n"

@app.route("/metrics", methods=["GET"])
def metrics():
    return monitor.export_http()[0]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)

# curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://0.0.0.0:5001/invocations
