# Set number of GPUs
export NUM_GPUS=1

# Reduce CUDA fragmentation (optional; helps on some drivers)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fine-tune on g1-pick-apple with modality aligned to datasets/g1-pick-apple/meta/modality.json
# (see examples/g1-pick-apple/g1_pick_apple_modality_config.py)
#
# Small-VRAM: per-device batch = global_batch_size / num_gpus; gradient accumulation;
# gradient checkpointing; tune_top_llm_layers=0 freezes full Eagle LLM;
# --no-tune-diffusion-model freezes DiT (like N1.5 4090 tip) but still trains projector+vlln.
# If VRAM allows better quality, try --tune-top-llm-layers 4 and/or --tune-diffusion-model.
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /home/jiangeng/Isaac-GR00T/datasets/g1-pick-apple \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /home/jiangeng/Isaac-GR00T/examples/g1-pick-apple/g1_pick_apple_modality_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir /home/jiangeng/Isaac-GR00T/outputs/g1_pick_apple_ft \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 2000 \
    --use-wandb \
    --global-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --gradient-checkpointing \
    --no-tune-diffusion-model \
    --tune-top-llm-layers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4



python gr00t/eval/real_robot/G1/eval_g1_0330.py \
  --policy_host 127.0.0.1 \
  --policy_port 5556 \
  --action_horizon 16 \
  --control_frequency 30 \
  --lang_instruction "Pick up the object." \
  --camera_host localhost \
  --camera_port 5555 \
  --init_state gr00t/eval/real_robot/G1/g1_0330_episode0_frame0_init_state.txt \
  --urdf ~/xr_teleoperate/assets/g1/g1_body29_hand14.urdf \
  --mesh_dir ~/xr_teleoperate/assets/g1/



