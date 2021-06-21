from pathlib import Path
import pickle

from sklearn.metrics import f1_score
import pandas as pd
import click


@click.command()
@click.option("--model-path")
@click.option("--splitted-path")
def predict(
        model_path: str,
        splitted_path: str
) -> None:
    model_path = Path(model_path)
    with open(model_path / Path('model.pkl'), 'rb') as f:
        model = pickle.load(f)
    splitted_path = Path(splitted_path)
    test_data = pd.read_csv(splitted_path / Path('test_data.csv'))
    X = test_data.drop(columns=['target'], axis='columns')
    y = test_data['target'].values
    with open(model_path / Path('f1_score.txt'), 'w') as f:
        f.write(
            f1_score(
                y, model.predict(X)
            ))


if __name__ == '__main__':
    predict()