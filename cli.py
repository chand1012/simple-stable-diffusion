from sd import StableDiffusion, Models, DEFAULT_CONFIG
import fire


def gen(prompt, negative_prompt='', add_trigger=True, width=512, height=512, num_steps=50, guidance=7.5, output='image.png'):
    sd = StableDiffusion(Models.OPENJOURNEY2, nsfw=True, device='cpu')
    image = sd.generate(prompt, negative_prompt, add_trigger, {
                        'width': width, 'height': height, 'num_inference_steps': num_steps, 'guidance_scale': guidance})
    image.save(output)


if __name__ == "__main__":
    fire.Fire(gen)
