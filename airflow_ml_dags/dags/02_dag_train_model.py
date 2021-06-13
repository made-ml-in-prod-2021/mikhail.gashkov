from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    MODEL_PATH,
    PREPROCESSED_PATH,
    RAW_DATA_PATH,
    SPLITTED_PATH,
    TRAIN_SIZE,
    VOLUME
)


with DAG(
        "2_dag_train_model",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:
    start_ml = DummyOperator(task_id='begin-do-ml')

    preprocess = DockerOperator(
        image="mgashkov/airflow-preprocess",
        command=f"--raw-path {RAW_DATA_PATH} --preprocessed-path {PREPROCESSED_PATH}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    split = DockerOperator(
        image="mgashkov/airflow-split",
        command=f"--preprocessed-path {PREPROCESSED_PATH} --preprocessed-path {SPLITTED_PATH} --train-size {TRAIN_SIZE}",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    train = DockerOperator(
        image="mgashkov/airflow-train-model",
        command=f"--splitted-path {SPLITTED_PATH} --model-path {MODEL_PATH}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    validate = DockerOperator(
        image="mgashkov/airflow-validate",
        command=f"--model-path {MODEL_PATH} --splitted-path {SPLITTED_PATH}",
        task_id="docker-airflow-validate-model",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    start_ml >> preprocess >> split >> train >> validate