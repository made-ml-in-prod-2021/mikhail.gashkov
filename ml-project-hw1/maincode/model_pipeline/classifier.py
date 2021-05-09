import os
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .custom_exceptions import ModelNotCreatedException, NotExistingModelException

CUR_DIR: Path = Path(os.path.abspath(__file__))


class Classifier:
    """
    Creates classifier, depending on type, that user selected
    """

    def __init__(
        self,
        model_type: str,
        path: Path = os.path.curdir,
        load_data_from_file: bool = False,
    ) -> None:
        self.model_type = model_type
        self.features_num = None
        if model_type == 'Log_reg':
            self.model = LogisticRegression()
        elif model_type == 'Grad_boost':
            self.model = GradientBoostingClassifier()
        else:
            raise NotExistingModelException
        if load_data_from_file:
            if os.path.exists(path):
                self.model_type, self.features_num, self.model = joblib.load(path)
            else:
                raise FileNotFoundError

    def save_model(self, model_save_path: Path):
        joblib.dump((self.model_type, self.features_num, self.model), model_save_path)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            raise ModelNotCreatedException
        assert (
            X.shape[0] == y.shape[0]
        ), f'Wrong datashapes! X.shape = {X.shape}, y.shape = {y.shape}'
        self.model.fit(X, y)
        self.features_num = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ModelNotCreatedException
        assert (
            X.shape[1] == self.features_num
        ), f'Model uses {self.features_num} features, your X is {X.shape}'
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ModelNotCreatedException
        assert (
            X.shape[1] == self.features_num
        ), f'Model uses {self.features_num} features, your X is {X.shape}'
        return self.predict_proba(X)

    def get_classification_report(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, dict]:
        y_pred = self.model.predict(X)
        return classification_report(y_true, y_pred, output_dict=True)
