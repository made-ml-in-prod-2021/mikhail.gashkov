from pathlib import Path
import os

from sklearn.model_selection import train_test_split
import click
import pandas as pd


@click.command()
@click.option("--preprocessed-path")
@click.option("--splitted-path")
@click.option("--train-size")
def split(
        preprocessed_path: str,
        splitted_path: str,
        train_size: float = 0.85
) -> None:
    preprocessed_path = Path(preprocessed_path)
    data = pd.read_csv(preprocessed_path / Path('preprocessed_data.csv'))

    train_data, test_data = train_test_split(data, train_size=train_size)
    splitted_path = Path(splitted_path)
    os.makedirs(splitted_path, exist_ok=True)
    train_data.to_csv(splitted_path / Path('train_data.csv'), index=False)
    test_data.to_csv(splitted_path / Path('test_data.csv'), index=False)


if __name__ == '__main__':
    split()