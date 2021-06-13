from pathlib import Path
import os

from sklearn.datasets import load_breast_cancer
import typer

app = typer.Typer()


@app.command()
def generate_data(base_path: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(base_path, exist_ok=True)
    X.to_csv(base_path / Path('raw') / Path('features.csv'), index=False)
    y.to_csv(base_path / Path('raw') / Path('target.csv'), index=False)


if __name__ == '__main__':
    app()