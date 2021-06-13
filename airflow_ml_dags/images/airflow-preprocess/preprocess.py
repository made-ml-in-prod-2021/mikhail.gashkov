from pathlib import Path
import os

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def preprocess(
        base_path: Path
) -> None:
    income_features = pd.read_csv(base_path / Path('raw') / Path('features.csv'))
    income_target = pd.read_csv(base_path / Path('raw') / Path('target.csv'))

    data = income_features
    data['target'] = income_target
    data.dropna(inplace=True, how='any')
    data.to_csv(base_path / Path('preprocessed') / Path('preprocessed_data.csv'), index=False)


if __name__ == '__main__':
    app()