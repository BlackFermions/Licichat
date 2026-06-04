FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema para rarfile
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    unrar-free \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5007

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5007", \
    "--timeout", "120", \
    "--worker-class", "gthread", \
    "--threads", "2", \
    "--keep-alive", "5", \
    "chat_engine:app"]
