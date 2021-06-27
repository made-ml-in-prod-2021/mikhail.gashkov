import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import typer

from data_model import (
    CATEGORICAL_FEATURES_LIST,
    NUMERICAL_FEATURES_LIST,
)

REQUEST_URL = 'http://0.0.0.0:8000/predict'

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def do_predictions(data_path: Path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f'Read data error: {e}')
        sys.exit(1)
    logger.info(f'Request URL is {REQUEST_URL}')
    request_features = df.columns
    logger.info(f'Features in request {request_features}')
    request_data = df.values.to_list()

    logger.debug('Prediction request')

    response = requests.post(
        REQUEST_URL,
        json={
            'features': request_features,
            'data': request_data
        }
    )
    logger.debug(f'Response status code is {response.status_code}, body of response is {response.json()}')
    logger.info('Prediction is finished')

if __name__ == '__main__':
    app()