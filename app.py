# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
# assuming the StableDiffusion class is in stable_diffusion.py
from sd import StableDiffusion, Models, DEFAULT_CONFIG, MODELS
import io  # For BytesIO
import base64  # For Base64 encoding
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Define Pydantic model for request body validation


class DiffusionRequest(BaseModel):
    prompt: str
    negative_prompt: str = ''
    add_trigger: bool = True
    opts: dict = DEFAULT_CONFIG


stable_diff = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stable_diff
    # Allow NSFW, its too sensitive sometimes
    stable_diff = StableDiffusion(
        Models.DREAMLIKE, nsfw=True)
    yield
    stable_diff = None


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Mount static files in public folder
app.mount("/public", StaticFiles(directory="public"), name="public")

# Serve index.html at root URL


@app.get("/")
def read_root():
    return FileResponse("public/index.html")


@app.post("/generate_image/")
async def generate_image(diffusion_request: DiffusionRequest = Body(...), model: str = 'dreamlike'):
    """
    Generate image based on the provided model and prompt, then return as Base64 encoded JPEG.
    """
    global stable_diff
    try:
        if stable_diff is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if stable_diff.model != model:
            stable_diff.load_model(model, nsfw=True)

        # Generate image using StableDiffusion
        # Assuming generated_image is a PIL Image object
        generated_image = stable_diff.generate(
            prompt=diffusion_request.prompt,
            negative_prompt=diffusion_request.negative_prompt,
            add_trigger=diffusion_request.add_trigger,
            opts=diffusion_request.opts
        )

        # Convert to PNG and Base64 encode
        buffered = io.BytesIO()
        generated_image.save(buffered, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"b64_json": image_base64}

    except Exception as e:
        print(e)
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))


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
