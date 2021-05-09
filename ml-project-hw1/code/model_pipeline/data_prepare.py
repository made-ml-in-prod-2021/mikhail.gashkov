from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler

# CAT_FEATURES = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
# NUM_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


class DataPreparation:
    """
    Class for data preparation
    """

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_features: List[str],  # = CAT_FEATURES,
        numerical_features: List[str],  # = NUM_FEATURES,
        target_column: str = 'target',
    ) -> None:
        self.df = df
        self.cat_features = categorical_features
        self.num_features = numerical_features
        self.X = df.drop(target_column, axis=1)
        self.is_data_prepared = False
        self.scaler = None
        if target_column in df.columns:
            self.y = df[target_column]
        else:
            self.y = None

    def _prepare_cat_features(self) -> None:
        self.X = pd.get_dummies(self.X, columns=self.cat_features)

    def _prepare_num_features(self) -> None:
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X[self.num_features])
        else:
            self.X = self.scaler.transform((self.X[self.num_features]))

    def prepare_data(self):
        self._prepare_cat_features()
        self._prepare_num_features()
        self.X = self.X
        self.is_data_prepared = True

    def save_data(self, df_path: Path):
        self.df['target'] = self.y
        self.df.to_csv(df_path, index=False)
