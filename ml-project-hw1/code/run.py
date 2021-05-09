import logging
import os
from pathlib import Path
import pickle
import sys

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import typer

from load_config import (
    load_config,
    DATA_CONFIG_FILE,
    TRAIN_CONFIG_FILE,
    PREDICT_CONFIG_FILE
)
from model_pipeline.classifier import Classifier
from model_pipeline.data_prepare import DataPreparation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = typer.Typer()


@app.command()
def make_predictions() -> np.ndarray:
    '''
    Make predictions on income data
    '''
    predictions_params = load_config(PREDICT_CONFIG_FILE)
    logger.info(f"making predictions ...")
    logger.info(f"loading data from {predictions_params['csv_to_predict']}")
    data_params = load_config(DATA_CONFIG_FILE)
    income_data_df = pd.read_csv(predictions_params['csv_to_predict'])
    prepared_data = DataPreparation(
        df=income_data_df,
        categorical_features=data_params['categorical_features'],
        numerical_features=data_params['numerical_features'],
        target_column=data_params['target_col']
    )
    prepared_data.prepare_data()
    logger.info(f"loading model from {predictions_params['using_model_path']}")
    clf = Classifier(model_type='Log_reg',
                     path=predictions_params['using_model_path'],
                     load_data_from_file=True)
    predictions = clf.predict(prepared_data.X)
    logger.info(f"Predictions are made with classifier {clf}")
    if predictions_params['save_predictions_to_income_data_csv']:
        prepared_data.y = predictions
        prepared_data.save_data(predictions_params['csv_to_predict'])
    else:
        pd.DataFrame(predictions).to_csv(predictions_params['predictions_saving_path'], index=False)


@app.command()
def fit_classifier():
    model_params = load_config(TRAIN_CONFIG_FILE)
    data_params = load_config(DATA_CONFIG_FILE)
    clf = Classifier(model_type=model_params['train_params']['classifier_type'])
    df = pd.read_csv(data_params['input_data_path'])
    prepared_data = DataPreparation(
        df=df,
        categorical_features=data_params['categorical_features'],
        numerical_features = data_params['numerical_features'],
        target_column = data_params['target_col']
    )
    prepared_data.prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        prepared_data.X, prepared_data.y,
        train_size=model_params['splitting_params']['train_size'],
        random_state=model_params['splitting_params']['random_state']
    )
    clf.fit(X_train, y_train)
    clf_report = clf.get_classification_report(X_test, y_test)
    clf.save_model(model_save_path=Path(__file__).parents[1] / model_params['output_model_path'])
    pd.DataFrame(clf_report).transpose().to_csv(
        model_params['classification_report_path'], index=False
    )


if __name__ == '__main__':
    app()