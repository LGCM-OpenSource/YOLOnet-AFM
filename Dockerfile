# Imagem base do Python
FROM python:3.9-slim

# Atualiza o sistema e instala dependências necessárias
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt e instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

# Copie o restante do código do projeto
COPY . .

# Configuração para o backend TkAgg do Matplotlib
ENV MPLBACKEND TkAgg

# Configurar variáveis de ambiente para evitar problemas com buffers
ENV PYTHONUNBUFFERED=1

# Comando padrão para executar quando o contêiner iniciar
CMD ["python", "/app/dev/apps/main.py"]