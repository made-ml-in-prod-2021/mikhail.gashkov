import pickle
from pathlib import Path
import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
import click


@click.command()
@click.option("--splitted-path")
@click.option("--model-path")
def train(
        splitted_path: str,
        model_path: str
) -> None:
    splitted_path = Path(splitted_path)
    model_path = Path(model_path)
    train_data = pd.read_csv(splitted_path / Path('train_data.csv'))
    y = train_data['target']
    X = train_data.drop(columns=['target'], axis='columns')
    lg = LogisticRegression(max_iter=5_000).fit(X, y)
    with open(model_path / Path('model.pkl'), 'wb') as f:
        pickle.dump(lg, f)


if __name__ == '__main__':
    train()