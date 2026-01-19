import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causvid.models.wan.wan_wrapper import WanVAEWrapper
from causvid.util import launch_distributed_job
import torch.distributed as dist
import imageio.v3 as iio
from tqdm import tqdm
import argparse
import torch
import json
import math
import csv
import numpy as np

torch.set_grad_enabled(False)


ACTION_FIELDS = ("ws", "ad", "attack", "pitch", "yaw", "pitch_delta", "yaw_delta")
PITCH_FIELDS = {"pitch", "yaw", "pitch_delta", "yaw_delta"}


def video_to_numpy(video_path):
    """
    Reads a video file and returns a NumPy array containing all frames.

    :param video_path: Path to the video file.
    :return: NumPy array of shape (num_frames, height, width, channels)
    """
    return iio.imread(video_path, plugin="pyav")  # Reads the entire video as a NumPy array


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def _safe_get_action(entry, key):
    if entry is None:
        return 0.0
    return entry.get(key, 0.0)


def _extract_actions(action_path, frame_indices):
    with open(action_path, "r") as f:
        data = json.load(f)

    actions = data["actions"]
    last_entry = None
    output = []
    for idx in frame_indices:
        entry = actions.get(str(idx), None)
        if entry is None:
            entry = last_entry
        else:
            last_entry = entry

        frame_vec = []
        for key in ACTION_FIELDS:
            value = float(_safe_get_action(entry, key))
            if key in PITCH_FIELDS:
                value = value / 180.0
            frame_vec.append(value)
        output.append(frame_vec)
    return np.array(output, dtype=np.float32)


def _sample_frame_indices(num_frames, target_frames):
    if target_frames <= 0 or target_frames == num_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, target_frames).round().astype(np.int64)


def _load_metadata_csv(metadata_csv_path):
    rows = []
    with open(metadata_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str,
                        help="Path to the folder containing input videos.")
    parser.add_argument("--output_latent_folder", type=str,
                        help="Path to the folder where output latents will be saved.")
    parser.add_argument("--info_path", type=str,
                        help="Path to the info file containing video metadata.")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Optional metadata.csv path for action-conditioned datasets.")
    parser.add_argument("--metadata_dir", type=str, default=None,
                        help="Directory containing per-agent action json files.")
    parser.add_argument("--target_frames", type=int, default=81,
                        help="Number of frames to sample before VAE encoding.")
    parser.add_argument("--temporal_stride", type=int, default=4,
                        help="Temporal stride of the VAE to map actions to latents.")

    args = parser.parse_args()

    # Step 1: Setup the environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Step 2: Create the generator
    use_distributed = all(k in os.environ for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
    if use_distributed:
        launch_distributed_job()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    device = torch.cuda.current_device()

    if args.info_path:
        with open(args.info_path, "r") as f:
            video_info = json.load(f)
        video_paths = sorted(list(video_info.keys()))
        metadata_rows = None
    elif args.metadata_csv:
        if not args.metadata_dir:
            raise ValueError("--metadata_dir is required when using --metadata_csv")
        metadata_rows = _load_metadata_csv(args.metadata_csv)
        video_paths = [row["video"] for row in metadata_rows]
        video_info = None
    else:
        raise ValueError("Provide either --info_path or --metadata_csv")

    model = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    os.makedirs(args.output_latent_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(video_paths) / world_size))), disable=rank != 0):
        global_index = index * world_size + rank
        if global_index >= len(video_paths):
            break

        video_path = video_paths[global_index]
        
        # Check if output latent file already exists
        output_latent_path = os.path.join(args.output_latent_folder, f"{global_index:08d}.pt")
        if os.path.exists(output_latent_path):
            if rank == 0:
                print(f"Skipping {video_path} - latent file already exists: {output_latent_path}")
            continue
        
        if video_info is not None:
            prompt = video_info[video_path]
            action_left = None
            action_right = None
        else:
            row = metadata_rows[global_index]
            prompt = row["prompt"]
            action_left = os.path.join(args.metadata_dir, row["action_left"])
            action_right = os.path.join(args.metadata_dir, row["action_right"])

        try:
            array = video_to_numpy(os.path.join(
                args.input_video_folder, video_path))
        except:
            print(f"Failed to read video: {video_path}")
            continue

        frame_indices = _sample_frame_indices(array.shape[0], args.target_frames)
        array = array[frame_indices]

        video_tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0).permute(
            0, 4, 1, 2, 3
        ) / 255.0
        video_tensor = video_tensor * 2 - 1
        video_tensor = video_tensor.to(torch.bfloat16)
        encoded_latents = encode(model, video_tensor).transpose(2, 1)

        if action_left and action_right:
            actions_left = _extract_actions(action_left, frame_indices)
            actions_right = _extract_actions(action_right, frame_indices)
            actions = np.concatenate([actions_left, actions_right], axis=1)

            # map actions to latent timestep
            actions = actions[::args.temporal_stride]
            target_latent_frames = encoded_latents.shape[1]
            if actions.shape[0] < target_latent_frames:
                pad = np.repeat(actions[-1:], target_latent_frames - actions.shape[0], axis=0)
                actions = np.concatenate([actions, pad], axis=0)
            elif actions.shape[0] > target_latent_frames:
                actions = actions[:target_latent_frames]

            # start frame latent
            start_frame = video_tensor[:, :, :1]
            start_latent = encode(model, start_frame).transpose(2, 1)

            torch.save(
                {
                    "prompt": prompt,
                    "video": video_path,
                    "latents": encoded_latents.cpu().detach(),
                    "actions": actions,
                    "start_latent": start_latent.cpu().detach()
                },
                output_latent_path
            )
        else:
            torch.save(
                {prompt: encoded_latents.cpu().detach()},
                output_latent_path
            )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
