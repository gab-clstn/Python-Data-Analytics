FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OUTPUT_DIR=/app/output
VOLUME ["/app/output"]

CMD ["python", "analytics.py"]