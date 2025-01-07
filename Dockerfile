# Base Python image
FROM python:3.9-slim

# Update the system and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy the rest of the project code
COPY . .

# Configuration for the TkAgg backend of Matplotlib
ENV MPLBACKEND TkAgg

# Configure environment variables to avoid buffer issues
ENV PYTHONUNBUFFERED=1

# Default command to execute when the container starts
CMD ["python", "/app/dev/apps/main.py"]