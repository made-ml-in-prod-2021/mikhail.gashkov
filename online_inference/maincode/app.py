import logging
import os

from fastapi import FastAPI
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

app = FastAPI()

GREETING_MESSAGE = 'Hello, this is prediction health service. Go to /predict endopoint to predict on your data'


@app.get('/')
def read_root():
    return {'text': GREETING_MESSAGE}


@app.get('/predict')
def predict_handler():
    pass