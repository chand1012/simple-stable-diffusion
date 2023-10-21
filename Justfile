redis:
    docker run --name redis-sd-api -p 6379:6379 --rm redis:latest

docker:
    docker compose -f docker/docker-compose.yml up -d
