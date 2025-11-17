SERVICE_NAME = yolonet_afm
CONTAINER = YOLOnet-AFM
COMPOSE_FILE = docker/docker-compose.yml

.PHONY: build up down restart logs shell clean

build:
	docker compose -f $(COMPOSE_FILE) build

up:
	xhost +SI:localuser:$(shell whoami)
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

restart: down up

logs:
	docker compose -f $(COMPOSE_FILE) logs -f $(SERVICE_NAME)

shell:
	docker exec -it $(CONTAINER) /bin/bash

run:
	docker exec -it $(CONTAINER) python /app/dev/apps/main.py

clean:
	docker system prune -f
	docker volume prune -f
