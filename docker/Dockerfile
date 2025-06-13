FROM python:3.9-slim

# Instala dependências do SO
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o requirements primeiro
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Agora copia o restante do código
COPY . .

ENV MPLBACKEND TkAgg
ENV PYTHONUNBUFFERED 1


