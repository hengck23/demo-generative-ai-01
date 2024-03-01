# generate video : image to video
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    #"stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()


# conditioning image
image_files=[
	'gen_image_prompt0.out1.png',
	'gen_image_prompt1.out1.png',
	'gen_image_prompt2.out0.png',
]
for image_file in image_files:
	image = load_image(image_file)
	image = image.resize((1024, 576))

	generator = torch.manual_seed(42)
	frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
	export_to_video(frames, image_file.replace('.png','.mp4'), fps=7)
