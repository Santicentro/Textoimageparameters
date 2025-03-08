#from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

#pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = "Brutalist building like cyberpunk, backlight, centered composition, masterpiece, photorealistic, 8k"

generator = torch.Generator(device="cuda").manual_seed(100)  
num_inference_steps = 50  
guidance_scale = 7.5  

image = pipeline(
    prompt=prompt,
    generator=generator,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
).images[0]

image.save("generated_image_solver{}.png".format(1))