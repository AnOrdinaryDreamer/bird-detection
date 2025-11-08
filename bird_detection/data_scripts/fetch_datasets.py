from pathlib import Path
from download_kaggle_dataset import bulk_download

BIRD_DATASET = "wenewone/cub2002011"
SQUIRREL_DATASET = "olgreyfox/openimagev7-raccoonsquirrelskunkmouserabbit"

if __name__ == "__main__":
    results = bulk_download(
        [BIRD_DATASET, SQUIRREL_DATASET],
        raw_dir=Path("data/raw"),
        extracted_dir=Path("data/extracted"),
    )
    for r in results:
        print(f"{r.dataset} → {r.extract_dir}")