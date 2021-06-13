from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    MODEL_PATH,
    RAW_DATA_PATH,
    VOLUME
)


with DAG(
        "3_dag_predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    get_predictions = DockerOperator(
        image="mgashkov/airflow-predict",
        command=f"--model-path {MODEL_PATH} --raw-path {RAW_DATA_PATH}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "Got new predictions for today!"',
    )

    get_predictions >> notify