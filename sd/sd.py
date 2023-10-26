from enum import Enum
from typing import Union

import torch
from diffusers import (DPMSolverMultistepScheduler,
                       LMSDiscreteScheduler, DiffusionPipeline)
from PIL import Image

from util.resolve_device import resolve_device
from sd.postprocess import PostProcessor

# Models Lookup table.
# All of these can be viewed by adding https://huggingface.co/ to the beginning of the model name.
MODELS = {
    'sd-1.5': 'runwayml/stable-diffusion-v1-5',
    'sd-2.1': 'stabilityai/stable-diffusion-2-1',
    'sd-2.0': 'stabilityai/stable-diffusion-2-0',
    '2-base': 'stabilityai/stable-diffusion-2-base',
    '4x-upscaler': 'stabilityai/stable-diffusion-4x-upscaler',
    'inpainting': 'runwayml/stable-diffusion-inpainting',
    'inpainting2': 'stabilityai/stable-diffusion-2-inpainting',
    'depth': 'stabilityai/stable-diffusion-2-depth',
    'waifu': 'hakurei/waifu-diffusion-v1-3',
    'openjourney': 'prompthero/openjourney',
    'openjourney2': 'prompthero/openjourney-v4',
    'inkpunk': 'Envvi/Inkpunk-Diffusion',
    'analog': 'wavymulder/Analog-Diffusion',
    'spiderverse': 'nitrosocke/spider-verse-diffusion',
    'nitro': 'nitrosocke/Nitro-Diffusion',
    'ghibli': 'nitrosocke/Ghibli-DIffusion',
    'future': 'nitrosocke/Future-Diffusion',  # uses 2.0
    'funko': 'prompthero/funko-diffusion',
    'edgerunners': 'DGSpitzer/Cyberpunk-Anime-Diffusion',
    'elden': 'nitrosocke/elden-ring-diffusion',
    'magic': 'Qilex/magic-diffusion',
    'disney': 'nitrosocke/mo-di-diffusion',
    'pokemon': 'lambdalabs/sd-pokemon-diffusers',
    'synthwave-punk': 'ItsJayQz/SynthwavePunk-v2',
    'sprite': 'Onodofthenorth/SD_PixelArt_SpriteSheet_Generator',
    'synthwave': 'PublicPrompts/Synthwave',
    'photoreal': 'dreamlike-art/dreamlike-photoreal-2.0',
    'dreamlike': 'dreamlike-art/dreamlike-diffusion-1.0',
    # 'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0', # doesn't work on our GPU properly
    'ssd-1b': 'segmind/SSD-1B'
}


# Enum for models.
class Models(Enum):
    SD_1_5 = 'sd-1.5'
    SD_2_1 = 'sd-2.1'
    SD_2_0 = 'sd-2.0'
    SD_2_BASE = '2-base'
    SD_4x_UPSCALER = '4x-upscaler'
    SD_INPAINTING = 'inpainting'
    SD_INPAINTING2 = 'inpainting2'
    SD_DEPTH = 'depth'
    WAIFU = 'waifu'
    OPENJOURNEY = 'openjourney'
    OPENJOURNEY2 = 'openjourney2'
    INKPUNK = 'inkpunk'
    ANALOG = 'analog'
    SPIDERVERSE = 'spiderverse'
    NITRO = 'nitro'
    GIBLI = 'ghibli'
    FUTURE = 'future'
    FUNKO = 'funko'
    EDGERUNNERS = 'edgerunners'
    ELDEN = 'elden'
    MAGIC = 'magic'
    DISNEY = 'disney'
    POKEMON = 'pokemon'
    SYNTHWAVE_PUNK = 'synthwave-punk'
    SPRITE = 'sprite'
    SYNTHWAVE = 'synthwave'
    PHOTOREAL = 'photoreal'
    DREAMLIKE = 'dreamlike'
    SDXL = 'sdxl'
    SSD = 'ssd-1b'

    def __str__(self) -> str:
        return self.value


# models have a trigger word that can be used to select them.
# these are used to look up the trigger word for a model.
MODEL_TRIGGERS = {
    'inkpunk': 'nvinkpunk, ',
    'openjourney': 'mdjrny-v4 style, ',
    'funko': 'funko style, ',
    'ghibli': 'ghibli style, ',
    'analog': 'analog style, ',
    'nitro': 'arcane archer modern disney ',
    'edgerunners': 'dgs illustration style, ',
    'spiderverse': 'spiderverse style, ',
    'elden': 'elden ring style ',
    'magic': 'mtg style ',
    'disney': 'modern disney style ',
    'pokemon': 'pokemon style ',
    'synthwave-punk': 'snthwve style nvinkpunk,',
    'sprite': 'PixelartFSS',
    'synthwave': 'snthwve style ',
    'photoreal': 'photo ',
    'dreamlike': 'dreamlikeart '
}

# these will be the only two schedulers available to the user.
SCHEDULERS = {
    'ddim': DPMSolverMultistepScheduler,
    'k_lms': LMSDiscreteScheduler,
}

# default config for the model.
DEFAULT_CONFIG = {
    'height': 512,
    'width': 512,
    'num_inference_steps': 50,
    'guidance_scale': 7.5,
}


class StableDiffusion:
    '''Stable Diffusion wrapper class.'''

    def __init__(self, model: Union[str, Models] = 'sd-1.5', nsfw=False, scheduler='k_lms', upscale_factor=1, upscale_cpu=False, attention_slicing=True, xformers=True, device=None):
        self.load_model(model, nsfw, scheduler,
                        upscale_factor, upscale_cpu, attention_slicing, xformers, device)

    def load_model(self, model: Union[str, Models] = 'sd-1.5', nsfw=False, scheduler='k_lms', upscale_factor=1, upscale_cpu=False, attention_slicing=True, xformers=True, device=None):
        '''Loads a new model.'''
        if upscale_factor not in [1, 2, 4]:
            raise ValueError(
                f"Upscale factor must be 1, 2, or 4. Not {upscale_factor}")

        # string of the model. Allows for both enum and string.
        self.model = str(model)
        self.model_name = MODELS.get(self.model, self.model)
        self.refiner = None
        self.pipe = None

        if self.model not in MODELS:
            print("WARNING: Model is not supported. Use at your own risk.")

        print("Loading model: " + self.model_name)
        # guess the device
        if device is None:
            self.device = resolve_device()
        else:
            self.device = device
        # load the model
        if device == 'cpu':
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name, torch_dtype=torch.float32)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name, torch_dtype=torch.float16)
            if xformers:
                self.pipe.enable_xformers_memory_efficient_attention()
        # set the scheduler
        if scheduler not in SCHEDULERS:
            raise ValueError(
                f"Scheduler must be one of {list(SCHEDULERS.keys())}. Not {scheduler}")
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(
            self.pipe.scheduler.config)
        # disable the nsfw checker if nsfw is allowed.
        if nsfw:
            self.pipe.safety_checker = None
        # set the device
        self.pipe = self.pipe.to(self.device)

        if attention_slicing and self.device != 'mps':
            # this wasn't working properly on the MPS device.
            self.pipe.enable_attention_slicing()
        # image and upscaled image placeholders
        self.image = None
        self.upscaled_image = None
        # if upscale factor is not 1, load the postprocessor.
        if upscale_factor > 1:
            print("Loading RealESRGAN & GFPGAN")
            self.postprocessor = PostProcessor(
                upscale=upscale_factor, cpu=upscale_cpu)
        else:
            self.postprocessor = None

    def upscale(self, image=None) -> Image.Image:
        '''Runs the postprocessor on the image.'''
        if image is None:
            image = self.image
        if self.postprocessor is None:
            print("Upscaling is disabled")
            return image
        if image is None:
            raise ValueError("No image to upscale")
        self.upscaled_image = self.postprocessor.process(image)
        return self.upscaled_image

    def generate(self, prompt: str, negative_prompt: str = '', add_trigger=True, opts: dict = DEFAULT_CONFIG) -> Image.Image:
        '''Generates an image from the prompt.'''
        # this is recommended to "warp up" the MPS device.
        if self.device == 'mps':
            _ = self.pipe(prompt, num_inference_steps=1)
        # if the trigger is desired, add it to the prompt.
        if add_trigger:
            prompt = MODEL_TRIGGERS.get(self.model, '') + prompt
        # generate the image

        if not self.refiner:
            if self.device == 'cuda':
                with torch.autocast('cuda'):
                    self.image = self.pipe(
                        prompt, negative_prompt=negative_prompt, **opts).images[0]
            else:
                self.image = self.pipe(
                    prompt, negative_prompt=negative_prompt, **opts).images[0]
        else:
            if self.device == 'cuda':
                with torch.autocast('cuda'):
                    self.image = self.pipe(
                        prompt=prompt, negative_prompt=negative_prompt, output_type="latent", **opts).images
                    self.image = self.refiner(
                        prompt=prompt, negative_prompt=negative_prompt, denoising_start=0.8, image=self.image).images[0]
            else:
                self.image = self.pipe(
                    prompt, negative_prompt=negative_prompt, output_type="latent", **opts).images
                self.image = self.refiner(
                    prompt=prompt, negative_prompt=negative_prompt, denoising_start=0.8, image=self.image).images[0]
        return self.image
