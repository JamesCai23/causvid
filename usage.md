## Prepare Data
1. Dataset Preparation (tree_cut_filtered)

2. Compute VAE latents + actions + start-frame latent
```bash
python distillation_data/compute_vae_latent.py \
  --metadata_csv /home/lff/data1/cym/worldmodel/worldmodel_data/results/tree_cut_filtered/metadata.csv \
  --metadata_dir /home/lff/data1/cym/worldmodel/worldmodel_data/results/tree_cut_filtered/metadata \
  --input_video_folder /home/lff/data1/cym/worldmodel/worldmodel_data/results/tree_cut_filtered/videos \
  --output_latent_folder /home/lff/data1/cym/worldmodel/CausVid/output_latent/tree_cut_filtered \
  --target_frames 300 \
  --temporal_stride 4
```

3. Build LMDB
```bash
python causvid/ode_data/create_lmdb_iterative.py \
  --data_path /home/lff/data1/cym/worldmodel/CausVid/output_latent/tree_cut_filtered \
  --lmdb_path /home/lff/data1/cym/worldmodel/CausVid/output_latent/tree_cut_filtered_lmdb \
  --no_dedupe_prompts
```

## Training Session
```bash
torchrun --nproc_per_node 8 causvid/train_distillation.py \
  --config_path configs/wan_causal_dmd_action.yaml
```

## Verification
- Check LMDB keys: `latents`, `actions`, `start_latent`, `prompts`
- Run `minimal_inference/autoregressive_inference.py` with a checkpoint trained from the action config

