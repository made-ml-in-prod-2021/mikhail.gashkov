from pathlib import Path
import os
import pickle

import pandas as pd
import click


@click.command()
@click.option("--model-path")
@click.option("--raw-path")
def predict(model_path: str, raw_path: str) -> None:
    model_path = Path(model_path)
    raw_path = Path(raw_path)
    os.makedirs(model_path, exist_ok=True)
    with open(model_path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    data = pd.read_csv(raw_path / Path('features.csv'))
    data['predictions'] = model.predict(data)
    data.to_csv(raw_path / Path('predictions.csv'), index=False)


if __name__ == '__main__':
    predict()