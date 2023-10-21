from sd import StableDiffusion, Models
import fire


def gen(prompt, negative_prompt='', add_trigger=True, width=512, height=512, num_steps=50, guidance=7.5, output='image.png', device=None):
    sd = StableDiffusion(Models.DREAMLIKE, nsfw=True, device=device)
    image = sd.generate(prompt, negative_prompt, add_trigger, {
                        'width': width, 'height': height, 'num_inference_steps': num_steps, 'guidance_scale': guidance})
    image.save(output)


if __name__ == "__main__":
    fire.Fire(gen)
