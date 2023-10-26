import os
from typing import Literal
from celery import Celery
import io  # For BytesIO
import base64
from minio import Minio
from sd import DEFAULT_CONFIG, StableDiffusion
from sd.postprocess import PostProcessor
import uuid
from dotenv import load_dotenv
from datetime import timedelta
from PIL import Image

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "Q3AM3UQ867SPQQA43P2F")
MINIO_SECRET_KEY = os.getenv(
    "MINIO_SECRET_KEY", "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG")
MINIO_URL = os.getenv("MINIO_URL", "http://127.0.0.1:9000")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "imagegen")

celery = Celery("sd-worker", broker=REDIS_URL, backend=REDIS_URL)


@celery.task(name="imagegen")
def imagegen_task(prompt: str, negative_prompt: str = '', upscale_factor: int = 1, add_trigger: bool = True, model: str = 'dreamlike', opts: dict = DEFAULT_CONFIG):
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

    if upscale_factor > 1:
        generated_image = stable_diff.upscale(generated_image)

    # Upload image to minio
    filename = upload_image(generated_image)
    # return the url of the minio image
    return filename

# upscales the image. Takes in a base64 encoded image and returns a base64 encoded image


@celery.task(name="upscale")
def image_upscale_task(image: str, upscale_factor: Literal[2, 4] = 2):
    p = PostProcessor(upscale=upscale_factor)
    buf = io.BytesIO(base64.b64decode(image))
    img = Image.open(buf)
    img = p.process(img)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    buffered.seek(0)  # idk if this is needed or not
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
