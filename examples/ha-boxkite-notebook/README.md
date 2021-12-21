# Boxkite Demo

## Notebook
```bash
jupyter notebook
```

If Jupyter is not installed:
```bash
pip install jupyterlab
```

## Data format

Boxkite is basically useful for a regression or classification model whose prediction problem can be generalized as <code>[feature-1, feature-2, ..., feature-n] => inference</code>.

For the training (a.k.a. baseline) features and inference, it is logged by the function <code>export_text()</code>.
```python
from boxkite.monitoring.service import ModelMonitoringService

ModelMonitoringService.export_text(
    features=features, inference=inference, path='hist.txt',
)
```

The data format of features and inference are as below:
```python
features = [("age", [33, 23, 54]), ("sex", [1, 0, 1])]
inference = ["cat1", "cat2", "cat1"]
```

Both the features and inference can be of continuous or discrete values. For discrete values, they needs to be encoded to number in advance.

https://github.com/tintinrevient/boxkite/blob/master/boxkite/monitoring/collector/feature.py#L123-L126

## References
* https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
* https://kubernetes.io/docs/tasks/access-application-cluster/service-access-application-cluster/
* https://kubernetes.io/docs/tasks/configure-pod-container/configure-persistent-volume-storage/
* https://docs.docker.com/engine/reference/commandline/run/