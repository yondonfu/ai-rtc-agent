import os

from pathlib import Path


def civitai_model_path(filename: str) -> Path:
    cache_dir = Path(os.getenv("CIVITAI_CACHE", "./models/civitai"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Path(cache_dir) / filename
