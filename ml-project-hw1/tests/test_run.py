from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch
import pytest

from maincode.run import app as typer_app

runner = CliRunner()


@patch('run.DATA_CONFIG_FILE')
@patch('run.PREDICT_CONFIG_FILE')
def test_app_make_predictions(mock_data_config, mock_predict_config):
    mock_data_config = Path('tests/data_config_for_test.yaml')
    mock_predict_config = Path('tests/predict_config_for_tests.yaml')
    runner = CliRunner()
    result = runner.invoke(typer_app, ["make-predictions"])
    assert result is None


# @patch('run.DATA_CONFIG_FILE')
# @patch('run.TRAIN_CONFIG_FILE')
# def test_app_fit_classifier(mock_data_config, mock_train_config):
#     print(runner, typer_app)
#     mock_data_config.__str__ = 'tests/data_config_for_test.yaml'
#     mock_train_config.__str__ = 'tests/train_config_for_test.yaml'
#     result = runner.invoke(typer_app, ["fit-classifier"])
#     assert result is None