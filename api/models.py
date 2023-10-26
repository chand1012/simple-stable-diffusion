from pydantic import BaseModel

from sd import DEFAULT_CONFIG

class DiffusionRequest(BaseModel):
    prompt: str
    negative_prompt: str = ''
    add_trigger: bool = True
    upscale_factor: int = 2
    opts: dict = DEFAULT_CONFIG

class UpscaleRequest(BaseModel):
    image: str
    upscale_factor: int = 2
