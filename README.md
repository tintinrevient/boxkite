![Boxkite logo](https://github.com/boxkite-ml/boxkite/raw/master/docs/images/boxkite-text.png)

[![PyPI version](https://badge.fury.io/py/boxkite.svg)](https://pypi.python.org/pypi/boxkite/)
[![PyPI license](https://img.shields.io/pypi/l/boxkite.svg)](https://pypi.python.org/pypi/boxkite/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/boxkite.svg)](https://pypi.python.org/pypi/boxkite/)
[![CI workflow](https://github.com/boxkite-ml/boxkite/actions/workflows/ci.yml/badge.svg)](https://github.com/boxkite-ml/boxkite/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/boxkite-ml/boxkite/branch/master/graph/badge.svg?token=0qgLm01XN3)](https://codecov.io/gh/boxkite-ml/boxkite)

_{Fast, Correct, Simple} - pick three_

# Easily compare training and production ML data & model distributions

## Goals

Boxkite is an instrumentation library designed from ground up for tracking **concept drift** in HA (Highly Available) model servers. It integrates well with existing DevOps tools (ie. Grafana, Prometheus, fluentd, kubeflow, etc.), and scales horizontally to multiple replicas with no code or infrastructure change.

- **Fast**
  - 0.5 seconds to process 1 million data points (training)
  - Sub millisecond p99 latency (serving)
  - Supports sampling for large data sets
- **Correct**
  - Aggregates histograms from multiple server replicas (using PromQL)
  - Separate counters for discrete and continuous variables (ie. categorical and numeric features)
  - Initialises serving histogram bins from training data set (based on Freedman-Diaconis rule)
  - Handles unseen data, `nan`, `None`, `inf`, and negative values
- **Simple**
  - One metric for each counter type (no confusion over which metric to choose)
  - Default configuration supports both feature and inference monitoring (easy to setup)
  - Small set of dependencies: prometheus, numpy, and fluentd
  - Extensible metric system (support for image classification coming soon)

Some non-goals of this project are:

- Adversarial detection

If you are interested in alternatives, please refer to our discussions in [FAQ](#FAQ).

## Getting Started

Follow one of our tutorials to easily get started and see how Boxkite works with other tools:

- [Prometheus & Grafana](https://boxkite.ml/en/latest/tutorials/grafana-prometheus) in Docker Compose locally
- [Kubeflow & MLflow](https://boxkite.ml/en/latest/tutorials/kubeflow-mlflow) on Kubernetes with **easy online test drive in the browser**

See [Installation](https://boxkite.ml/en/latest/installing) & [User Guide](https://boxkite.ml/en/latest/using) for how to use Boxkite in any environment.

## FAQ

1. Does boxkite support anomaly / outlier detection?

Prometheus has supported outlier detection in time series data since 2015. Once you've setup KL divergence and K-S test metrics, outlier detection can be configured on top using alerting rules. For a detailed example, refer to this tutorial: https://prometheus.io/blog/2015/06/18/practical-anomaly-detection/.

2. Does boxkite support adversarial detection?

Adversarial detection concerns with identifying single OOD (Out Of Distribution) samples rather than comparing whole distributions. The algorithms are also highly model specific. For these reasons, we do not have plans to support them in boxkite at the moment. As an alternative, you may look into Seldon for such capabilities https://github.com/SeldonIO/alibi-detect#adversarial-detection.

3. Does boxkite support concept drift detection for text / NLP models?

Not yet. This is still an actively researched area that we are keeping an eye on.

4. Does boxkite support tensorflow / pytorch?

Yes, our instrumentation library is framework agnostic. It expects input data to be a `list` or `np.array` regardless of how the model is trained.

## Contributors

The following people have contributed to the original concept and code

- [Han Qiao](https://github.com/sweatybridge)
- [Nguyen Hien Linh](https://github.com/nglinh)
- [Luke Marsden](https://github.com/lukemarsden)
- [Mariappan Ramasamy](https://github.com/Mariappan)

A full list of contributors, which includes individuals that have contributed entries, can be found [here](https://github.com/boxkite-ml/boxkite/graphs/contributors).

## Shameless plug

Boxkite is a project from BasisAI, who offer an MLOps Platform called Bedrock.

[Bedrock](https://basis-ai.com/product) helps data scientists own the end-to-end deployment of machine learning workflows. Boxkite was originally part of the Bedrock client library, but we've spun it out into an open source project so that it's useful for everyone!

## Boxkite dummy example

<p>
  <img src="./pix/boxkite.png" width="800" />
</p>

1. Create the virtual environment for this example, and install the required modules.
```bash
cd boxkite_example

python -m venv venv

pip install pip-tools
pip-compile
pip-sync
```

2. Install the postgresql for the MLflow server as "tracking_uri".
```bash
brew install postgresql

brew services start postgresql
createuser -s postgres
```

3. Initialize the postgresql for the MLflow server, and start the MLflow server.
```bash
cd mlflow

sh mlflow_init.sh
sh mlflow_start_server.sh
```

4. Check the weh UI of the MLflow server: http://localhost:5000

5. Check the postgresql DB of the MLflow "tracking_uri".
```bash
psql -U mlflow

mlflow=> \l
mlflow=> \du

mlflow=> \c mlflow
You are now connected to database "mlflow" as user "mlflow".

mlflow=> \dt
mlflow=> select * from experiments;
```

6. Train an example model, and log the trained model with its histogram in the MLflow server.
```bash
cd boxkite_example

python boxkite_example/train.py
```

7. Serve the trained model in the Flask server: http://localhost:5001
```bash
cd boxkite_example

python boxkite_example/serve.py
```

8. Query the trained model by cURL.
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" \
  --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' \
  http://0.0.0.0:5001/invocations
```

9. The Boxkite metrics are exposed: http://localhost:5001/metrics

10. Install the Prometheus using [pre-compiled binaries](https://prometheus.io/docs/prometheus/latest/installation/)

11. Start the Prometheus server with the updated config "prometheus.yml": http://localhost:9090
```bash
./prometheus
```

```bash
scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "boxkite"
    static_configs:
      - targets: [ "localhost:5001" ]
```

12. Install the Grafana using [macOS binaries](https://grafana.com/docs/grafana/latest/installation/mac/#installing-on-macos)

13. Start the Grafana server: http://localhost:3000
```bash
./bin/grafana-server web
```

14. Configure the Prometheus data source in the Grafana server.
<p>
  <img src="./pix/grafana-prometheus-data-source.png" width="500" />
</p>

15. Create the Boxkite dashboard using the [defined model JSON](https://github.com/boxkite-ml/boxkite/blob/master/examples/grafana-prometheus/metrics/dashboards/model.json)
<p>
  <img src="./pix/grafana-boxkite-dashboard.png" width="700" />
</p>

## Virtual environment

1. Use **setuptools** to package the project whose structure is as follows:
  * **pyproject.toml** - declare you want to use setuptools to package your project.
  * **setup.py** - specify your package information, such as metadata, contents, dependencies.

```
├── <dir_name>                
|   ├── <package_name>
|   |   ├── __init__.py
|   |   ├── <library_name>
|   |   |   ├── __init__.py
|   |   └── <library_name>
|   |   |   ├── __init__.py
|   ├── <tests>
|   |   ├── __init__.py
|   |   ├── <package_name>
|   |   |   ├── __init__.py
|   ├── dist
|   ├── venv
|   ├── pyproject.toml
|   ├── setup.py
```

2. Use **venv** to create and activate the virtual environment, which contains the would-be installed python packages and binaries.
```bash
python -m venv venv
source venv/bin/activate
```

3. Install **pip-tools** to compile and sync the local development environment.
```bash
pip install pip-tools

pip-compile # generate "requirements.txt"
pip-sync # install the required packages
```

4. Use **build** to build the package in the distribution directory "dist", e.g. either a "tar.gz" file or a ".whl" file.
```bash
python -m build --sdist
python -m build --wheel
```

5. Use **twine** to upload your package to PyPI.
```bash
twine upload dist/*

Uploading distributions to https://upload.pypi.org/legacy/
```

6. Use **pipdeptree** to display the direct and transitive dependencies in the virtual environment.
```bash
pipdeptree
pipdeptree --exclude pip,setuptools,venv,pip-tools,wheel,pipdeptree,build,twine,readme-renderer --graph-output png > "dependencies.png" 
```

<p>
  <img src="./pix/pipdeptree-dependencies.png" width="800" />
</p>

7. Check [Python Module Index](https://docs.python.org/3.9/py-modindex.html) for the built-in packages that comes with the installed python version.

8. Use **pytest** to perform the unit test for the package "boxkite_example".
```bash
cd boxkite_example

pytest
```

9. Following the above steps, create and activate the virtual environment for the package "click_example".

10. Install the "boxkite_example" package.
```bash
pip install -e ../boxkite_example
```

11. Check whether the method of "click_example" can import and use the method of "boxkite_example".
```bash
cd click_example

python click_example/app.py --start_date="2000-01-01"
```

## References
* https://blog.basis-ai.com/introducing-boxkite-open-source-model-observability-for-mlops-teams
* https://grafana.com/blog/2021/08/02/how-basisai-uses-grafana-and-prometheus-to-monitor-model-drift-in-machine-learning-workloads/  
* https://pypi.org/project/psycopg2/
* https://gist.github.com/ibraheem4/ce5ccd3e4d7a65589ce84f2a3b7c23a3
* https://setuptools.pypa.io/en/latest/userguide/quickstart.html
* https://www.mlflow.org/docs/latest/rest-api.html
* https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html
* https://www.mlflow.org/docs/latest/tracking.html
* https://www.mlflow.org/docs/latest/model-registry.html
* https://docs.pytest.org/en/6.2.x/goodpractices.html
* https://docs.pytest.org/en/6.2.x/example/index.html
* https://tox.wiki/en/latest/example/pytest.html
* https://w3techs.com/technologies/comparison/ws-gunicorn,ws-jetty,ws-nodejs

## Commands

```bash
lsof -i -P | grep LISTEN
kill -9 xxxxx
```
