import pickle
from pathlib import Path
import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def train(
        base_path: Path,
) -> None:
    train_data = pd.read_csv(base_path / Path('splitted') / Path('train_data.csv'))
    y = train_data['target']
    X = train_data.drop(columns=['target'], axis='columns')
    lg = LogisticRegression(max_iter=5_000).fit(train_data)
    with open (base_path / Path('models') / Path('model.pkl'), 'wb') as f:
        pickle.dump(lg, f)


if __name__ == '__main__':
    app()