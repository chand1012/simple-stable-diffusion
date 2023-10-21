import os
from celery import Celery
from sd import StableDiffusion, DEFAULT_CONFIG
import io  # For BytesIO
from minio import Minio
import uuid
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "Q3AM3UQ867SPQQA43P2F")
MINIO_SECRET_KEY = os.getenv(
    "MINIO_SECRET_KEY", "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG")
MINIO_URL = os.getenv("MINIO_URL", "http://127.0.0.1:9000")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "imagegen")

celery = Celery("sd-worker", broker=REDIS_URL, backend=REDIS_URL)

# stable_diff = StableDiffusion(
#         Models.DREAMLIKE, nsfw=True, upscale_factor=UPSCALE_FACTOR)


@celery.task(name="imagegen")
def imagegen_task(prompt: str, negative_prompt: str = '', add_trigger: bool = True, upscale_factor: int = 1, model: str = 'dreamlike', opts: dict = DEFAULT_CONFIG):
    # if model is not already loaded, load the new model
    # if stable_diff.model != model:
    #     stable_diff.load_model(model, nsfw=True, upscale_factor=upscale_factor)
    stable_diff = StableDiffusion(
        model, nsfw=True, upscale_factor=upscale_factor)
    # Generate image using StableDiffusion
    # Assuming generated_image is a PIL Image object
    generated_image = stable_diff.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        add_trigger=add_trigger,
        opts=opts
    )

    if stable_diff.postprocessor != None:
        # run the upscaler
        generated_image = stable_diff.upscale()

    # Upload image to minio
    filename = upload_image(generated_image)
    # return the url of the minio image
    return filename


def upload_image(img):
    client = Minio(
        MINIO_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Make 'ImageGen' bucket if not exist.
    found = client.bucket_exists(MINIO_BUCKET_NAME)
    if not found:
        client.make_bucket(MINIO_BUCKET_NAME)
    else:
        print(f"Bucket '{MINIO_BUCKET_NAME}' already exists")

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    file_name = f"{uuid.uuid4()}.jpg"
    length = len(buffered.getvalue())
    buffered.seek(0)
    client.put_object(
        MINIO_BUCKET_NAME,
        file_name,
        data=buffered,
        length=length,
        content_type='image/jpeg'
    )
    print(f'uploaded image {file_name} to minio')
    url = client.presigned_get_object(
        MINIO_BUCKET_NAME, file_name, timedelta(days=7))

    return url
