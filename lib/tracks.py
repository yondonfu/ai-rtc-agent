import os
import logging
import threading
import queue
import time
import torch

from aiortc import MediaStreamTrack

logger = logging.getLogger(__name__)


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, pipeline):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.warmup_frame_idx = 0
        self.warmup_frames = os.getenv("WARMUP_FRAMES", 10)

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.process_frame_thread = threading.Thread(
            target=self._process_frame_loop, daemon=True
        )
        self.process_frame_thread.start()

    async def recv(self):
        # while self.warmup_frame_idx < self.warmup_frames:
        #     logger.info(f"dropping warmup frames {self.warmup_frame_idx}")
        #     frame = await self.track.recv()
        #     self.pipeline(frame)
        #     self.warmup_frame_idx += 1

        # The decoded frame returned by aiortc will be a torch.Tensor which
        # is then passed into the pipeline.
        # The pipeline returns the processed frame as a torch.Tensor which
        # is then passed to aiortc to be encoded into a frame.
        for _ in range(2):
            frame = await self.track.recv()
            self.input_queue.put(frame)

        return self.output_queue.get()

    def _process_frame_loop(self):
        batch = []
        batch_size = 2

        while True:
            frame = self.input_queue.get()
            if frame is None:
                break

            batch.append(self.pipeline.preprocess(frame).unsqueeze(0))

            if len(batch) < batch_size:
                time.sleep(0.005)
                continue

            inputs = torch.cat(batch)
            print(batch[0].shape)
            print(batch[1].shape)
            print(inputs.shape)
            batch.clear()

            outputs = self.pipeline.model.stream(inputs)

            for output in outputs:
                self.output_queue.put(self.pipeline.postprocess(output))
