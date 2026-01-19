import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causvid.models.wan.causal_inference import InferencePipeline
from causvid.models.wan.wan_wrapper import WanVAEWrapper
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--latent_length", type=int, default=21)
parser.add_argument("--start_frame_path", type=str, default=None)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

def _encode_start_frame(image_path, device, dtype, repeat_frames, latent_hw):
    image = Image.open(image_path).convert("RGB")
    latent_h, latent_w = latent_hw
    target_h = latent_h * 8
    target_w = latent_w * 8
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), resample=Image.BICUBIC)
    image_np = np.array(image, dtype=np.float32)
    frame = torch.tensor(image_np, device=device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    frame = frame * 2 - 1
    frame = frame.unsqueeze(2).to(dtype)

    vae = WanVAEWrapper().to(device=device, dtype=dtype)
    device, dtype = frame.device, frame.dtype
    scale = [vae.mean.to(device=device, dtype=dtype),
             1.0 / vae.std.to(device=device, dtype=dtype)]
    latent = vae.model.encode(frame, scale)
    latent = latent.to(dtype).transpose(2, 1)

    if repeat_frames > 1:
        latent = latent.repeat(1, repeat_frames, 1, 1, 1)
    return latent

sampled_noise = torch.randn(
    [1, args.latent_length, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    start_latents = None
    if args.start_frame_path is not None:
        start_latents = _encode_start_frame(
            args.start_frame_path,
            device="cuda",
            dtype=torch.bfloat16,
            repeat_frames=getattr(config, "num_frame_per_block", 1),
            latent_hw=(sampled_noise.shape[-2], sampled_noise.shape[-1])
        )

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        start_latents=start_latents
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
