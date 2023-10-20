api: sleep 10 && python3 -m uvicorn --port 8000 --host 0.0.0.0 app:app
worker: sleep 10 && python3 -m celery -A worker worker --loglevel=INFO -c 1
redis: docker run --name redis-sd-api -p 6379:6379 --rm redis:latest
minio: docker run -p 9000:9000 -p 9001:9001 -v $(pwd)/data:/data quay.io/minio/minio server /data --console-address ":9001"
