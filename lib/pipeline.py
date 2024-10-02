import torch
import cvcuda
import numpy as np
import os

from typing import List
from lib.wrapper import StreamDiffusionWrapper

DEFAULT_PROMPT = "fireworks in the night sky"
DEFAULT_T_INDEX_LIST = [18, 26, 35, 45]
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 0.0


class StreamDiffusionPipeline:
    def __init__(self, model_id: str):
        self.prompt = DEFAULT_PROMPT
        self.t_index_list = DEFAULT_T_INDEX_LIST
        self.device = "cuda"

        self.model = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            device=self.device,
            dtype=torch.float16,
            t_index_list=self.t_index_list,
            frame_buffer_size=2,
            use_lcm_lora=True,
            output_type="pt",
            mode="img2img",
            use_denoising_batch=True,
            use_tiny_vae=True,
            cfg_type="self",
            engine_dir=os.getenv("TRT_ENGINES_CACHE", "./models/engines"),
        )

        self.model.prepare(
            prompt=self.prompt,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
        )

    def update_prompt(self, prompt: str):
        self.prompt = prompt

    def update_t_index_list(self, t_index_list: List[int]):
        if t_index_list == self.model.stream.t_list:
            return

        self.model.prepare(
            prompt=self.prompt,
            t_index_list=t_index_list,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
        )

    def preprocess(self, frame: torch.Tensor):
        # dtype=uint8 -> dtype=float32
        frame = cvcuda.convertto(frame, np.float32, scale=1 / 255)
        # NHWC -> NCHW
        frame = cvcuda.reformat(frame, "NCHW")
        # StreamDiffusion expects a torch.Tensor without the batch dimension
        frame = torch.as_tensor(frame.cuda(), device=self.device).squeeze(0)

        return frame

    def predict(self, frame: torch.Tensor):
        return self.model(image=frame, prompt=self.prompt)

    def postprocess(self, frame: torch.Tensor):
        # dtype=float16 -> dtype=uint8
        return (frame * 255.0).clamp(0, 255).to(dtype=torch.uint8).unsqueeze(0)

    def __call__(self, frame: torch.Tensor):
        frame = self.preprocess(frame)
        frame = self.predict(frame)
        frame = self.postprocess(frame)

        return frame
