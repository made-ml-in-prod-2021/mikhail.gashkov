from pathlib import Path
import os

import pandas as pd
import click


@click.command()
@click.option("--raw-path")
@click.option("--preprocessed-path")
def preprocess(
        raw_path: str,
        preprocessed_path: str
) -> None:
    raw_path = Path(raw_path)
    income_features = pd.read_csv(raw_path / Path('features.csv'))
    income_target = pd.read_csv(raw_path / Path('target.csv'))

    preprocessed_path = Path(preprocessed_path)
    data = income_features
    data['target'] = income_target
    data.dropna(inplace=True, how='any')
    os.makedirs(preprocessed_path, exist_ok=True)
    data.to_csv(preprocessed_path / Path('preprocessed_data.csv'), index=False)


if __name__ == '__main__':
    preprocess()