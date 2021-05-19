FROM python:3.9
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY models/model.pkl ./model.pkl
COPY maincode ./maincode

WORKDIR .

#ENV PATH_TO_MODEL="/model.pkl"

CMD ['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '80']