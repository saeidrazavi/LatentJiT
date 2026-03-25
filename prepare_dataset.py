# /// script
# dependencies = [
#   "huggingface_hub[cli]",
#   "requests",
# ]
# ///

import os
import shutil
import glob
import requests
from pathlib import Path
from huggingface_hub import snapshot_download

def setup_dataset():
    # 1. Define paths
    base_dir = Path("ImageNet-Latents")
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    print("Creating directory structure...")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 2. Download the Dataset using huggingface_hub API
    # This replaces 'huggingface-cli download'
    print("Downloading dataset shards and metadata from Hugging Face...")
    snapshot_download(
        repo_id="G-REPA/imagenet-latents-davae-vavae-align1.5-400k",
        repo_type="dataset",
        local_dir=base_dir,
        allow_patterns=["*.arrow", "*.json", "val/*"]
    )

    # 3. Move root files to /train
    # (huggingface_hub snapshot_download puts files in local_dir root)
    print("Organizing files into train folder...")
    
    # Move .arrow files
    for arrow_file in glob.glob(str(base_dir / "*.arrow")):
        shutil.move(arrow_file, train_dir / os.path.basename(arrow_file))
        
    # Move .json files
    for json_file in glob.glob(str(base_dir / "*.json")):
        shutil.move(json_file, train_dir / os.path.basename(json_file))

    # 4. Download Labels using requests
    # This replaces 'wget'
    labels_url = "https://huggingface.co/datasets/G-REPA/imagenet_labels/resolve/main/imagenet_train_labels.txt"
    labels_path = base_dir / "imagenet_train_labels.txt"
    
    print(f"Downloading labels to {labels_path}...")
    response = requests.get(labels_url, stream=True)
    response.raise_for_status()
    with open(labels_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("\n✅ Setup Complete!")
    print(f"Structure created at: {base_dir.absolute()}")

if __name__ == "__main__":
    setup_dataset()