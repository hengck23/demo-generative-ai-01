# generate image : prompt to image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
	torch_dtype=torch.float16,
	variant="fp16",
	use_safetensors=True,
).to("cuda")

prompts = [
	"jellyfish swimming, photo",
	"a cardboard jellyfish swiming in the blue sky with some clouds",
	"small mechanical robot chrome jellyfish in center of the image",
]

for j,prompt in enumerate(prompts):
	images = pipeline_text2image(
		prompt=prompt,
		height=1024,
		width=1024,
		num_images_per_prompt=3,
	).images
	for i,image in enumerate(images):
		image.save(f'gen_image_prompt{j}.out{i}.png')
