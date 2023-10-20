from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
# assuming the StableDiffusion class is in stable_diffusion.py
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.models import DiffusionRequest
from sd import MODELS
from worker import imagegen_task, celery

import base64  # For Base64 encoding
import requests


# Initialize FastAPI app
app = FastAPI()


# Mount static files in public folder
app.mount("/public", StaticFiles(directory="public"), name="public")

# Serve index.html at root URL
@app.get("/")
def read_root():
    return FileResponse("public/index.html")

@app.post("/generate_image/")
def generate_image_blocking(diffusion_req: DiffusionRequest = Body(...), model: str = 'dreamlike', response_format: str | None = None):        
    
    if not model in MODELS.keys():
        raise HTTPException(422, 'Unprocessable Entity')
    
    t = imagegen_task.delay(**diffusion_req.model_dump(), model=model)
    image_url = celery.AsyncResult(t.id).get()
    if response_format == 'b64_json':
        response = requests.get(image_url)
        if response.status_code == 200:
            # Convert the image bytes to base64
            base64_encoded = base64.b64encode(response.content).decode()
            # Determine the content type (e.g., 'image/jpeg', 'image/png')
            content_type = response.headers.get('content-type')
            return {'data': content_type, 'b64_json': base64_encoded }
        else:
            raise HTTPException(400, 'cannot decode image, minio down?')
    return {'image_url': image_url}

@app.post("/generate/")
def generate(diffusion_req: DiffusionRequest = Body(...), model: str = 'dreamlike'):        
    
    if not model in MODELS.keys():
        raise HTTPException(422, 'Unprocessable Entity')
    
    t = imagegen_task.delay(**diffusion_req.model_dump(), model=model)
    return {'task_id': t.id}

@app.get('/task/{task_id}')
def task(task_id: str, response_format: str | None = None):
    image_url = celery.AsyncResult(task_id).get()
    if response_format == 'b64_json':
        response = requests.get(image_url)
        if response.status_code == 200:
            # Convert the image bytes to base64
            base64_encoded = base64.b64encode(response.content).decode()
            # Determine the content type (e.g., 'image/jpeg', 'image/png')
            content_type = response.headers.get('content-type')
            return {'data': content_type, 'b64_json': base64_encoded }
        else:
            raise HTTPException(400, 'cannot decode image, minio down?')
    return {'image_url': image_url}

@app.get('/model_list')
def get_model_list():
    return MODELS

@app.get("/{path:path}")
def read_files(path: str):
    filepath = Path("public") / path
    if filepath.exists() and filepath.is_file():
        return FileResponse(filepath)
    else:
        raise HTTPException(status_code=404, detail="File not found")
