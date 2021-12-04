import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
from boxkite.monitoring.service import ModelMonitoringService

@pytest.fixture(scope="module")
def df(request):

    os.chdir(request.fspath.dirname)

    df = pd.read_csv('data/winequality-red.csv', sep=';')

    yield df

    os.chdir(request.config.invocation_dir)

@pytest.fixture(scope='module')
def hist(request):

    os.chdir(request.fspath.dirname)

    with open('data/hist.txt', 'r', encoding='utf8') as _f:
        hist = _f.read()

    yield hist

    os.chdir(request.config.invocation_dir)

def test_hist(df, hist):

    preds = df[['quality']]
    train_df = df.drop(['quality'], axis=1)
    feature_names = train_df.columns

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file = Path(temp_dir, 'temp_baseline_hist.txt')

        ModelMonitoringService.export_text(
            features=zip(*[feature_names, train_df.values.T]), inference=preds, path=temp_file
        )

        with temp_file.open('r', encoding='utf8') as _f:
            baseline_hist = _f.read()

    assert baseline_hist == hist
