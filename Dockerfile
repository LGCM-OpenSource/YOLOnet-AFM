# Imagem base do Python
FROM python:3.9-slim

# Diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt e instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

# Copie o restante do código do projeto
COPY . .

# Comando padrão para executar quando o contêiner iniciar
CMD ["python", "/app/dev/apps/main.py"]
