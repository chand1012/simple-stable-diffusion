from gfpgan import GFPGANer
import torch
import os
import numpy as np
from PIL import Image

ARCH = 'clean'
CHANNEL_MULTIPLIER = 2
MODEL_NAME = 'GFPGANv1.4'
MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'

# if GFPGAN doesn't want to run on M1 mac,
# try this: https://stackoverflow.com/a/63073119/5178731

REAL_ESRGAN_MODELS = {
    2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}


class PostProcessor:
    '''Post processing for GFPGAN and RealESRGAN. Upscales the image either 2 or 4 times and enhances human faces.'''

    def __init__(self, upscale=4, cpu=False):
        # we will only support 2 and 4 times upscaling
        if upscale != 2 and upscale != 4:
            raise ValueError(
                f'Invalid upscale value. Must be 2 or 4, got {upscale}.')
        # determine model paths
        # default model paths
        model_path = os.path.join(
            'experiments/pretrained_models', MODEL_NAME + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', MODEL_NAME + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = MODEL_URL
        # this is how the example code imports these
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        cuda = torch.cuda.is_available()
        if not cuda or cpu:
            print('Warning: using CPU mode for upscaler. This will be very slow.')
        # define the GAN upscaler
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=upscale)
        device = 'cuda' if cuda and not cpu else 'cpu'
        bg_upsampler = RealESRGANer(
            scale=upscale,
            model_path=REAL_ESRGAN_MODELS[upscale],
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=cuda and not cpu,  # need to set False in CPU mode
            device=torch.device(device))
        # define the GFPGAN face enhancer
        self.gfpgan = GFPGANer(
            model_path=model_path,
            arch=ARCH,
            channel_multiplier=CHANNEL_MULTIPLIER,
            bg_upsampler=bg_upsampler,
            upscale=upscale,
        )

    def process(self, image: Image) -> Image:
        '''Process an image with GFPGAN and REALESRGAN.'''
        cv_image = np.array(image.convert('RGB'))
        # this also returns cropped and restored faces,
        # but we just want the processed image
        _, _, processed_image = self.gfpgan.enhance(
            cv_image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.8
        )
        if processed_image is None:
            raise RuntimeError('GFPGAN failed to process the image')
        return Image.fromarray(processed_image)
