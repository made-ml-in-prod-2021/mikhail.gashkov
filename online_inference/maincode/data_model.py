from typing import List, Union

from pydantic import BaseModel, conlist, validator

CATEGORICAL_FEATURES_LIST = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
NUMERICAL_FEATURES_LIST = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
INPUT_FEATURES_LIST = [*CATEGORICAL_FEATURES_LIST, *NUMERICAL_FEATURES_LIST]
WRONG_FEATURES_MESSAGE = 'You have loaded wrong data'


class HealthModel(BaseModel):
    data: List[conlist(Union[float, str], min_items=1)]
    features: List[str]

    @validator('features')
    def validate_features(cls, features):
        if features != INPUT_FEATURES_LIST:
            raise ValueError(WRONG_FEATURES_MESSAGE)
        return features


class HealthResponse(BaseModel):
    target: List[int]
