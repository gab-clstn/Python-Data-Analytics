FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ensure this matches your script name exactly!
CMD ["python", "analytics_bps.py"]