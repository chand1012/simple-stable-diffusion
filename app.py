# app.py

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
# assuming the StableDiffusion class is in stable_diffusion.py
from sd import StableDiffusion
import io  # For BytesIO
import base64  # For Base64 encoding

# Define Pydantic model for request body validation


class DiffusionRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: str = ''
    add_trigger: bool = True
    opts: dict = {}


# Initialize FastAPI app
app = FastAPI()

# Initialize StableDiffusion class globally
stable_diff = None


@app.on_event("startup")
async def load_model():
    global stable_diff
    # Assuming "default_model" is the model you want to load
    stable_diff = StableDiffusion(model="default_model")


@app.post("/generate_image/")
async def generate_image(diffusion_request: DiffusionRequest = Body(...)):
    """
    Generate image based on the provided model and prompt, then return as Base64 encoded PNG.
    """
    global stable_diff
    try:
        if stable_diff is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

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
        generated_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {"status": "success", "image": image_base64}

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))
