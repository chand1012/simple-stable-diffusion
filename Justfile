redis:
    docker run --name redis-sd-api -p 6379:6379 --rm redis:latest

build:
    docker compose -f docker/docker-compose.yml build

up:
    docker compose -f docker/docker-compose.yml up -d

serve:
    docker compose -f docker/docker-compose.yml up

dev: build serve
