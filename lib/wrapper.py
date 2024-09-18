# Copied from https://github.com/yondonfu/StreamDiffusion

import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import shutil
import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CudaStreamPtr:
    def __init__(self, cuda_stream_handle):
        self.ptr = cuda_stream_handle


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        controlnet_id_or_path: Optional[str] = None,
        controlnet_processor_id: Optional[str] = "hed",
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        cuda_stream_handle: Optional[int] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        self.sd_turbo = "turbo" in model_id_or_path

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            controlnet_id_or_path=controlnet_id_or_path,
            controlnet_processor_id=controlnet_processor_id,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            cuda_stream_handle=cuda_stream_handle,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        t_index_list: List[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        if t_index_list is not None:
            if len(t_index_list) != len(self.stream.t_list):
                raise Exception(
                    f"new and current t_index_list length do not match: {len(t_index_list)} != {len(self.stream.t_list)}"
                )
            self.stream.t_list = t_index_list

        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor, output_type=output_type)
        else:
            return postprocess_image(image_tensor, output_type=output_type)[0]

    def _load_trt_model(
        self,
        model_id_or_path: str,
        vae_id_or_path: str,
        trt_vae_encoder_path: str,
        trt_vae_decoder_path: str,
        trt_unet_path: str,
        t_index_list: List[int],
        do_add_noise: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        cuda_stream_handle=None,
    ):
        from polygraphy import cuda
        from streamdiffusion.acceleration.tensorrt.engine import (
            AutoencoderKLEngine,
            UNet2DConditionModelEngine,
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import UNet2DConditionModel, DEISMultistepScheduler

        if cuda_stream_handle is None:
            cuda_stream = cuda.Stream()
        else:
            cuda_stream = CudaStreamPtr(cuda_stream_handle=cuda_stream_handle)

        # Setting cache_dir explicitly should not be necessary if the HF_HUB_CACHE env var is set
        # but for some reason this doesn't work for load_config and we need to explicitly set cache_dir
        cache_dir = os.getenv("HF_HUB_CACHE", "./models/hub")

        vae_config = AutoencoderTiny.from_config(
            AutoencoderTiny.load_config(
                vae_id_or_path, cache_dir=cache_dir, local_files_only=True
            )
        ).config
        vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1)
        vae = AutoencoderKLEngine(
            trt_vae_encoder_path,
            trt_vae_decoder_path,
            cuda_stream,
            vae_scale_factor,
            use_cuda_graph=False,
        )
        setattr(vae, "config", vae_config)
        setattr(vae, "dtype", self.dtype)

        unet_config = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(
                model_id_or_path,
                subfolder="unet",
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).config
        unet = UNet2DConditionModelEngine(
            trt_unet_path, cuda_stream, use_cuda_graph=False
        )
        setattr(unet, "config", unet_config)

        text_encoder = CLIPTextModel.from_pretrained(
            model_id_or_path, subfolder="text_encoder", local_files_only=True
        ).to(self.device, dtype=self.dtype)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id_or_path, subfolder="tokenizer", local_files_only=True
        )
        scheduler = DEISMultistepScheduler.from_config(
            DEISMultistepScheduler.load_config(
                model_id_or_path,
                subfolder="scheduler",
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )

        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        gc.collect()
        torch.cuda.empty_cache()

        return stream

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        controlnet_id_or_path: Optional[str] = None,
        controlnet_processor_id: Optional[str] = "hed",
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        cuda_stream_handle: Optional[int] = None,
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """

        if acceleration == "tensorrt":
            try:
                engine_path = Path(engine_dir)

                model_path = Path(model_id_or_path)
                if model_path.exists():
                    model_prefix = model_path.stem.replace("/", "--")
                else:
                    model_prefix = model_id_or_path.replace("/", "--")

                engine_prefix = os.path.join(engine_path, f"engines--{model_prefix}")

                trt_vae_encoder_path = os.path.join(engine_prefix, "vae_encoder.engine")
                trt_vae_decoder_path = os.path.join(engine_prefix, "vae_decoder.engine")
                trt_unet_path = os.path.join(engine_prefix, "unet.engine")

                return self._load_trt_model(
                    model_id_or_path=model_id_or_path,
                    vae_id_or_path="madebyollin/taesd",
                    trt_vae_encoder_path=trt_vae_encoder_path,
                    trt_vae_decoder_path=trt_vae_decoder_path,
                    trt_unet_path=trt_unet_path,
                    t_index_list=t_index_list,
                    do_add_noise=do_add_noise,
                    cfg_type=cfg_type,
                    seed=seed,
                    cuda_stream_handle=cuda_stream_handle,
                )
            except Exception:
                traceback.print_exc()
                print(
                    "Error trying to directly load TensorRT model - will try to compile"
                )

        controlnet = None
        controlnet_processor = None

        if controlnet_id_or_path is not None:
            try:
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_id_or_path, device=self.device, torch_dtype=self.dtype
                )
            except ValueError:
                controlnet = ControlNetModel.from_single_file(
                    controlnet_id_or_path, device=self.device, torch_dtype=self.dtype
                )
            except Exception:
                traceback.print_exc()
                print("ControlNet model load has filed. Doesn't exist,")
                exit()

            if controlnet_processor_id == "hed":
                from controlnet_aux import HEDCudadetector

                controlnet_processor = HEDCudadetector.from_pretrained(
                    "lllyasviel/Annotators"
                ).to("cuda")
            else:
                traceback.print_exc()
                print("ControlNet conditioning not supported.")
                exit()

        try:  # Load from local directory
            if controlnet:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id_or_path, controlnet=controlnet
                ).to(device=self.device, dtype=self.dtype)
            else:
                pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            if controlnet:
                pipe = StableDiffusionControlNetPipeline.from_single_file(
                    model_id_or_path, controlnet=controlnet
                ).to(device=self.device, dtype=self.dtype)
            else:
                pipe: StableDiffusionPipeline = (
                    StableDiffusionPipeline.from_single_file(
                        model_id_or_path,
                    ).to(device=self.device, dtype=self.dtype)
                )
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusion(
            pipe=pipe,
            controlnet_processor=controlnet_processor,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                    UNet2DConditionControlNetModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    UNetControlNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)

                    use_controlnet = False
                    if controlnet:
                        use_controlnet = True

                    if maybe_path.exists():
                        return f"{maybe_path.stem}--controlnet-{use_controlnet}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"
                    else:
                        return f"{model_id_or_path}--controlnet-{use_controlnet}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    if controlnet:
                        unet_model = UNetControlNet(
                            fp16=True,
                            device=stream.device,
                            max_batch_size=stream.trt_unet_batch_size,
                            min_batch_size=stream.trt_unet_batch_size,
                            embedding_dim=stream.text_encoder.config.hidden_size,
                            unet_dim=stream.unet.config.in_channels,
                        )
                    else:
                        unet_model = UNet(
                            fp16=True,
                            device=stream.device,
                            max_batch_size=stream.trt_unet_batch_size,
                            min_batch_size=stream.trt_unet_batch_size,
                            embedding_dim=stream.text_encoder.config.hidden_size,
                            unet_dim=stream.unet.config.in_channels,
                        )

                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                if cuda_stream_handle is None:
                    cuda_stream = cuda.Stream()
                else:
                    cuda_stream = CudaStreamPtr(cuda_stream_handle=cuda_stream_handle)

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                if controlnet:
                    stream.unet = UNet2DConditionControlNetModelEngine(
                        unet_path, cuda_stream, use_cuda_graph=False
                    )
                else:
                    stream.unet = UNet2DConditionModelEngine(
                        unet_path, cuda_stream, use_cuda_graph=False
                    )

                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                # Move engine files
                model_path = Path(model_id_or_path)
                if model_path.exists():
                    model_prefix = model_path.stem.replace("/", "--")
                else:
                    model_prefix = model_id_or_path.replace("/", "--")

                dst_engine_prefix = os.path.join(engine_dir, f"engines--{model_prefix}")

                Path(dst_engine_prefix).mkdir(parents=True, exist_ok=True)

                dst_vae_encoder_path = os.path.join(
                    dst_engine_prefix, "vae_encoder.engine"
                )
                dst_vae_decoder_path = os.path.join(
                    dst_engine_prefix, "vae_decoder.engine"
                )
                dst_unet_path = os.path.join(dst_engine_prefix, "unet.engine")

                shutil.move(vae_encoder_path, dst_vae_encoder_path)
                shutil.move(vae_decoder_path, dst_vae_decoder_path)
                shutil.move(unet_path, dst_unet_path)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
