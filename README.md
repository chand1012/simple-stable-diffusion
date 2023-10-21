# Simple Stable Diffusion
Simple Stable Diffusion is a REST API that allows you to generate images through a queuing service. The service has workers dedicated to producing these images based on your request. This README will guide you through setting up and using the API effectively.

## Prerequisits: 
Install [Docker GPU requirements](https://docs.docker.com/config/containers/resource_constraints/#gpu)

1. Clone the Simple Stable Diffusion Repository
`git clone https://github.com/chand1012/simple-stable-diffusion.git`
2. Navigate to the project directory
`cd simple-stable-diffusion`
3. run docker compose up command
`docker compose -f docker/docker-compose.yml up -d`

That is it! You are up and running! But how do you use this?

[Minio Dev Console](http://localhost:9001): Navigate to [localhost:9001](http://localhost:9001) to see and configure the minio dev console. This is where all the images that are generated are stored.

# How do I generate an image?
Great question! The easiest way is to navigate to the [Web Console](http://localhost:8000) and put in a prompt and hit generate! There is currently no state in this web console so if you refresh the page, the only way to get your images back is through the [Minio Dev Console](http://localhost:9001).

Alternatively you can create a CURL request in your command line.
```bash
curl 'http://localhost:8000/generate_image/?model=dreamlike' -X POST -H 'Content-Type: application/json' --data-raw '{"prompt":"a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans, In style of by Jordan Grimmer and greg rutkowski, crisp lines and color, complex background, particles, lines, wind, concept art, sharp focus, vivid colors","negative_prompt":"cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry, depth of field, missing fingers","add_trigger":true,"opts":{"guidance_scale":7.5,"height":512,"num_inference_steps":22,"width":512}}'
```
