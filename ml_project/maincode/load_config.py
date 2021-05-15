from pathlib import Path
from typing import Dict

import yaml

DATA_CONFIG_FILE = Path(__file__).parents[1] / Path("config/data_config.yaml")
TRAIN_CONFIG_FILE = Path(__file__).parents[1] / Path("config/train_config.yaml")
PREDICT_CONFIG_FILE = Path(__file__).parents[1] / Path("config/predict_config.yaml")


def load_config(input_file: Path) -> Dict[str, str]:
    with open(input_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)
