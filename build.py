import os
import torch
from pathlib import Path
from lib.utils import civitai_model_path

from lib.wrapper import StreamDiffusionWrapper

DEFAULT_T_INDEX_LIST = [18, 26, 35, 45]


def build():
    model_id_or_path = "lykon/dreamshaper-8"

    ghibli_path = civitai_model_path("ghibli_style_offset.safetensors")
    lora_dict = {ghibli_path: 1.0}

    # Build TensorRT engines
    StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        device="cuda",
        dtype=torch.float16,
        t_index_list=DEFAULT_T_INDEX_LIST,
        frame_buffer_size=2,
        lora_dict=lora_dict,
        use_lcm_lora=True,
        output_type="pt",
        mode="img2img",
        use_denoising_batch=True,
        use_tiny_vae=True,
        cfg_type="self",
        engine_dir=os.getenv("TRT_ENGINES_CACHE", "./models/engines"),
    )


if __name__ == "__main__":
    build()
