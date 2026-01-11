"""Extract ZIP archives from data/raw/ to data/extracted/"""

import shutil
import zipfile
from pathlib import Path


def extract_archives():
    raw_dir = Path("data/raw")
    extracted_dir = Path("data/extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(raw_dir.glob("*.zip"))
    
    if not zip_files:
        print("No ZIP files found in data/raw/")
        return
    
    print(f"Found {len(zip_files)} ZIP archive(s) to extract")
    
    for zip_file in zip_files:
        dataset_name = zip_file.stem
        extract_path = extracted_dir / dataset_name
        
        print(f"\n📦 Processing {zip_file.name}...")
        
        # Remove old extraction if exists
        if extract_path.exists():
            print(f"   Removing old extraction: {extract_path}")
            shutil.rmtree(extract_path)
        
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Extract
        print(f"   Extracting to {extract_path}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_path)
        
        print(f"   ✓ Done: {zip_file.name}")
    
    print(f"\n✅ All {len(zip_files)} archive(s) extracted successfully!")


if __name__ == "__main__":
    extract_archives()
