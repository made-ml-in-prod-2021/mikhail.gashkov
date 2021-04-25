from typing import List

from sklearn.preprocessing import StandardScaler
import pandas as pd

CAT_FEATURES = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


class DataPreparation:
    '''
    Class for data preparation
    '''
    def __init__(self,
                 df: pd.DataFrame,
                 categorical_features: List[str] = CAT_FEATURES,
                 numerical_features: List[str] = NUM_FEATURES,
                 target_column: str = 'target',
                 ) -> None:
        self.cat_features = categorical_features
        self.num_features = numerical_features
        self.y = df[target_column]
        self.X = df.drop(target_column, axis=1)
        self.is_data_prepared = False

    def _prepare_cat_features(self) -> None:
        self.X = pd.get_dummies(self.X, columns=self.cat_features)

    def _prepare_num_features(self) -> None:
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X[self.num_features])

    def prepate_data(self):
        self._prepare_cat_features()
        self._prepare_num_features()
        self.X = self.X.values
        self.is_data_prepared = True
