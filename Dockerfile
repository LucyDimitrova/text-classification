FROM python:3.8

WORKDIR /app

COPY /trainer/train.py /app/train.py

COPY /trainer/requirements.txt /app/requirements.txt

COPY /algorithms /app/algorithms

COPY /models /app/models

COPY /logs /app/logs

COPY /preprocessing /app/preprocessing

RUN pip install -r requirements.txt

CMD [ "python", "-u", "-m", "memory_profiler", "train.py" ]