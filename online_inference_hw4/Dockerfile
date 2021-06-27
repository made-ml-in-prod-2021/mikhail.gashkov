FROM python:3.8

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN mkdir moidels
COPY models/model.pkl /models/model.pkl
RUN mkdir maincode
COPY maincode /maincode

WORKDIR .

#ENV PATH_TO_MODEL="/model.pkl"

CMD ["python", "maincode/app.py"]