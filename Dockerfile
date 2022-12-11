FROM python:3.9

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt requirements.txt
COPY recommendations/requirements.txt requirements-reco.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-reco.txt

COPY . .

ENV GUNICORN_WORKERS 4

CMD gunicorn main:app -c gunicorn.config.py
