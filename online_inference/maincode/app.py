import logging
import os
from pathlib import Path
from typing import List

import uvicorn as uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
import numpy as np
import pandas as pd

from .backend import make_predictions
from .data_model import HealthModel, HealthResponse
logger = logging.getLogger(__name__)

app = FastAPI()

GREETING_MESSAGE = 'Hello, this is prediction health service. Go to /predict endopoint to predict on your data'


@app.get('/')
def read_root_handler():
    return {'text': GREETING_MESSAGE}


@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.post('/predict', response_model=List[HealthResponse])
def predict_handler(request: HealthModel):
    logger.info('Making predictions')
    return make_predictions(request.data, request.features)


if __name__ == "__main__":
    uvicorn.run("app", host="0.0.0.0", port=os.getenv("PORT", 8000))