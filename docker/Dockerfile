FROM python:3.9-slim

# Instala dependências do sistema para GUI, Tk, OpenCV, Matplotlib, etc.
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxrandr2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxfixes3 \
    libxcb1 \
    libxcursor1 \
    libxi6 \
    libxcomposite1 \
    libasound2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia dependências Python
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY . .

# Configura backend gráfico do Matplotlib
ENV MPLBACKEND=TkAgg
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1
