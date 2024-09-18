import requests
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from lib.utils import civitai_model_path


class HFModel(BaseModel):
    id: str


class CivitaiModel(BaseModel):
    id: int
    version_id: int
    name: str


HF_MODELS = [
    {"id": "lykon/dreamshaper-8"},
    {"id": "latent-consistency/lcm-lora-sdv1-5"},
    {"id": "madebyollin/taesd"},
]

CIVITAI_MODELS = [
    {"id": 6526, "version_id": 7657, "name": "studio-ghibli-style-lora"},
]


def download_civitai_model(model: CivitaiModel):
    # TODO: Skip download if file already exists locally
    url = f"https://civitai.com/api/download/models/{model.version_id}"

    resp = requests.get(url, allow_redirects=True)
    content_disp = resp.headers.get("Content-Disposition")
    if content_disp:
        filename = content_disp.split("filename=")[1].strip('"')
    else:
        filename = f"{model.name}"

    model_path = civitai_model_path(filename)
    with open(model_path, "wb") as f:
        f.write(resp.content)


def download():
    for model in HF_MODELS:
        snapshot_download(model["id"])

    for model in CIVITAI_MODELS:
        download_civitai_model(CivitaiModel(**model))


if __name__ == "__main__":
    download()
