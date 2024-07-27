FROM python:3.10-slim

EXPOSE 5001

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip &&\
    pip install --no-cache-dir -r requirements.txt