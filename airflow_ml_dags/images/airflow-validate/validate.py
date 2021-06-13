from pathlib import Path
import pickle

from sklearn.metrics import f1_score
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def generate_data(base_path: Path) -> None:
    with open(base_path / Path('model') / Path('model.pkl'), 'rb') as f:
        model = pickle.load(f)
    test_data = pd.read_csv(base_path / Path('splitted') / Path('test_data.csv'))
    X = test_data.drop(columns=['target'], axis='columns')
    y = test_data['target'].values
    with open(base_path / Path('model') / Path('f1_score.txt'), 'w') as f:
        f.write(
            f1_score(
                y, model.predict(X)
            ))


if __name__ == '__main__':
    app()