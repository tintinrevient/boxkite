export MLFLOW_HOME=/Users/zhaoshu/Documents/workspace/boxkite/example

echo "Default artifact root: ${MLFLOW_HOME}/mlruns"

mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow \
  --default-artifact-root file://"${MLFLOW_HOME}"/mlruns \
  --port 5000