FROM python:3.11-slim

COPY requirements.txt .

RUN set -ex; \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /

ADD write_to_kafka.py .
ADD fraudTest.csv .

CMD python -u write_to_kafka.py
