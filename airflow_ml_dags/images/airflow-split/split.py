from pathlib import Path
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def split(
        base_path: Path,
        train_size: float = 0.85
) -> None:
    data = pd.read_csv(base_path / Path('preprocessed') / Path('preprocessed_data.csv'))

    train_data, test_data = train_test_split(data, train_size=train_size)

    train_data.to_csv(base_path / Path('splitted') / Path('train_data.csv'), index=False)
    test_data.to_csv(base_path / Path('splitted') / Path('test_data.csv'), index=False)


if __name__ == '__main__':
    app()