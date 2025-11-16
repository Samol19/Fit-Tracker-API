# Dockerfile optimizado para Google Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Instala dependencias del sistema necesarias para scipy y scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements y los instala
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código de la API y los modelos
COPY main.py ./
COPY model/ ./model/

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Usar múltiples workers para mejor rendimiento
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2 --loop uvloop --ws websockets