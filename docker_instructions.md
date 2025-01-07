# Docker Setup Guide

## Install Docker

### Windows
Use DockerHUB.

### Linux
Install `docker.io`:
```bash
sudo apt update  
sudo apt install docker.io  
```

### Debian
Follow the Linux installation steps.

---

## Data and Models Folders
Navigate to the project folder and create the following directories:
- `data`
- `models`

---

## Edit `docker-compose.yml`
Add the `data` folder to the `volumes` section of the configuration.

---

## Build Command
Run the following command to build the Docker image and install dependencies:

1. Navigate to the project folder:
```bash
cd <project folder>  
```

2. Build and start the container:
```bash
sudo docker compose up -d --build  
```

---

## Start Containers Without Rebuilding the Image
```bash
docker-compose up  
```

---

## Logs
View logs using:
```bash
docker compose logs  
```

---

## Stop Container
```bash
docker-compose stop  
```

---

## Start Container
```bash
docker-compose start  
```

---

## Check Running Containers
```bash
docker ps  
```

---

## Access Container
Access the container using:
```bash
docker compose exec unet_afm_container /bin/bash  
```

---

## Run Script
Navigate to the project folder and execute the following command:
```bash
sudo docker exec -it unet_afm_container python /app/dev/apps/main.py  
