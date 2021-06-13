from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    RAW_DATA_PATH,
    VOLUME)


with DAG(
        "1_dag_generate_data",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:

    start_download = DummyOperator(task_id='begin-download-data')

    generate = DockerOperator(
        image="mgashkov/airflow-generate-data",
        command=f"--raw-path {RAW_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "Generated data!"',
    )

    start_download >> generate >> notify