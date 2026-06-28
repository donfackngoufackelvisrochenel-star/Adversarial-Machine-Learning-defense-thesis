FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# Default: run Streamlit dashboard (for Railway deployment)
# Override via docker-compose for local multi-service setup
CMD streamlit run src/dashboard/streamlit_app.py --server.port 8080 --server.address 0.0.0.0
