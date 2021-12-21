from boxkite.monitoring.collector import BaselineMetricCollector
from boxkite.monitoring.service import ModelMonitoringService

monitor = ModelMonitoringService(
    baseline_collector=BaselineMetricCollector(path="hist.txt")
)

features = [33, 1]
score = "cat1"

for i in range(5):
    monitor.log_prediction(
        request_body="",
        features=features,
        output=score,
    )

result = monitor.export_http()[0]
print(result)

for i in range(7):
    monitor.log_prediction(
        request_body=None,
        features=features,
        output=score,
    )

result = monitor.export_http()[0]
print(result)
