from pathlib import Path
from typing import Union, List

import pandas as pd

from ml_project.maincode.model_pipeline.classifier import Classifier
from ml_project.maincode.model_pipeline.data_prepare import DataPreparation
from .data_model import HealthResponse, CATEGORICAL_FEATURES_LIST, NUMERICAL_FEATURES_LIST

MODEL_PATH = Path('..models/model.pkl')


def make_predictions(data: List[Union[float, int]],
                     features: List[str]
                     ) -> List[HealthResponse]:
    clf = Classifier(
        model_type='Log_reg',
        path=MODEL_PATH,
        load_data_from_file=True,
    )
    data_df = pd.DataFrame(data, columns=features)
    data_prep = DataPreparation(data_df,
                                categorical_features=CATEGORICAL_FEATURES_LIST,
                                numerical_features=NUMERICAL_FEATURES_LIST,
                                target_column='not_here')
    data_prep.prepare_data()
    preds = clf.predict(data_prep.X)
    return [HealthResponse(target=pred) for pred in preds]