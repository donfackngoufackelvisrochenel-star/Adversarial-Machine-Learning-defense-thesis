FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# Default: run full pipeline
# Override via Railway / docker-compose for specific services
CMD ["python", "scripts/run_pipeline.py"]
