from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Jiangeng/G1_0330",
    repo_type="dataset",
    local_dir="./cache/Jiangeng/G1_0330",
)

print(f"Dataset downloaded to: {local_dir}")
