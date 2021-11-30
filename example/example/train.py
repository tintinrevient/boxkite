import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

from boxkite.monitoring.service import ModelMonitoringService

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

            print("Active run_id: {}".format(run.info.run_id))
            print(mlflow.get_artifact_uri())

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

            ModelMonitoringService.export_text(
                features=features, inference=inference, path="./histogram.txt"
            )

            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(lr, "model")
            mlflow.log_artifact("./histogram.txt")