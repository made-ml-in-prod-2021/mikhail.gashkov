from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "mgashkov",
    "email": ["mihael.gashkov@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}
MODEL_PATH = "/data/model/{{ ds }}"
RAW_DATA_PATH = "/data/raw/{{ ds }}"
PREPROCESSED_PATH = "/data/preprocessed/{{ ds }}"
SPLITTED_PATH = "/data/preprocessed/{{ ds }}"
TRAIN_SIZE = 0.85
VOLUME = "/Users/mailru_made/ml_prod/mikhail.gashkov/airflow_ml_dags/data:/data"