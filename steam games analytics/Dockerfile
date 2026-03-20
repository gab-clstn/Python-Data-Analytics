FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset.csv .
COPY analytics.py .

ENV OUTPUT_DIR=/app/output

CMD ["sh", "-c", "python analytics.py && tail -f /dev/null"]