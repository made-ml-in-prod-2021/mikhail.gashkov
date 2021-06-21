from pathlib import Path
import os

from sklearn.datasets import load_breast_cancer
import click


@click.command()
@click.option("--raw-path")
def generate_data(raw_path: str = 'data') -> None:
    raw_path = Path(raw_path)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(raw_path, exist_ok=True)
    X.to_csv(raw_path / Path('features.csv'), index=False)
    y.to_csv(raw_path / Path('target.csv'), index=False)


if __name__ == '__main__':
    generate_data()