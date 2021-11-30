import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

from boxkite.monitoring.service import ModelMonitoringService

# Basic setting
model_path = "model"
model_name = "regression"
histogram_path = "histogram"

if __name__ == "__main__":

    # STEP 1 - tracking_uri
    tracking_uri = "postgresql://mlflow:mlflow@127.0.0.1:5432/mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    # STEP 2 - experiment_id w/ artifact_location
    experiment_name = "Boxkite Example Experiment"
    default_artifact_root = "file:///Users/zhaoshu/Documents/workspace/boxkite/example/mlruns"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name, artifact_location=default_artifact_root)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

        print("active run_id: {}".format(run.info.run_id))
        print("artifact_uri: {}".format(mlflow.get_artifact_uri()))

        np.random.seed(40)

        # Read the wine-quality csv file from the URL
        csv_url = (
            "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        )
        try:
            data = pd.read_csv(csv_url, sep=";")

        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        alpha = 0.5
        l1_ratio = 0.5

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        # features = [("fixed acidity", [5.4, 7.5, 6.4, ...]), ("volatile acidity": [0.74, 0.38, 0.63, ...])]
        print(train_x.head())
        col_names = list(train_x.columns)
        # print(col_names)
        features = zip(col_names, train_x.T.values.tolist())
        # print(list(features)[0])

        inference = list(predicted_qualities)

        # Step 1 - Log "histogram/histogram.txt" as the artifacts
        with tempfile.TemporaryDirectory() as temp_dir:

            os.makedirs(os.path.join(temp_dir, histogram_path))
            temp_histogram_dir = Path(temp_dir, histogram_path)
            temp_histogram_file = Path(temp_dir, histogram_path, 'histogram.txt')
            # print(histogram_file)

            hist = ModelMonitoringService.export_text(
                features=features, inference=inference, path=temp_histogram_file
            )

            mlflow.log_artifact(temp_histogram_dir)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Step 2 - Log the parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Step 3 - Log the model
        mlflow.sklearn.log_model(lr, artifact_path=model_path)

        # Step 4.1 - Register model name in the model registry
        client = MlflowClient()
        if not client.get_registered_model(model_name):
            client.create_registered_model(model_name)

        # Step 4.2 - Create a new version of the model under the registered model name
        desc = "A new version of the model"
        runs_uri = f"runs:/{run.info.run_id}/{model_path}"
        model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
        print("model_src: {}".format(model_src))

        mv = client.create_model_version(model_name, model_src, run.info.run_id, description=desc)

        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Description: {}".format(mv.description))
        print("Status: {}".format(mv.status))
        print("Stage: {}".format(mv.current_stage))