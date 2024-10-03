import torch
import cvcuda
import numpy as np
import os
import nvcv
import av

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
            frame_buffer_size=1,
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

    def preprocess(self, frame: nvcv.Tensor | av.VideoFrame):
        if not isinstance(frame, nvcv.Tensor) and not isinstance(frame, av.VideoFrame):
            raise Exception("invalid frame type")

        if isinstance(frame, av.VideoFrame):
            frame = frame.to_ndarray(format="rgb24")
            frame = cvcuda.as_tensor(
                torch.from_numpy(frame).unsqueeze(0).to(self.device), "NHWC"
            )

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

    def __call__(
        self, frame: torch.Tensor | av.VideoFrame
    ) -> torch.Tensor | av.VideoFrame:
        pre_output = self.preprocess(frame)
        pred_output = self.predict(pre_output)
        post_output = self.postprocess(pred_output)

        if not os.getenv("NVENC"):
            output = post_output.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            output = av.VideoFrame.from_ndarray(output)

            # At the moment, we require that if the requested output type is av.VideoFrame then
            # the input type is also av.VideoFrame
            assert isinstance(frame, av.VideoFrame)

            output.pts = frame.pts
            output.time_base = frame.time_base

            return output

        return post_output
