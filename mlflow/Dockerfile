FROM python:3.10-slim

EXPOSE 5001

COPY requirements.txt .

RUN pip install -U pip &&\
    pip install --no-cache-dir -r requirements.txt