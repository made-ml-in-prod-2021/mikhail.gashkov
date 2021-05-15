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
    result = runner.invoke(typer_app, ["make-predictions"])
    assert result is None

