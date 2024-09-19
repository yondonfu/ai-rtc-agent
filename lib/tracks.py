import os
import logging

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

    async def recv(self):
        while self.warmup_frame_idx < self.warmup_frames:
            logger.info(f"dropping warmup frames {self.warmup_frame_idx}")
            frame = await self.track.recv()
            self.pipeline(frame)
            self.warmup_frame_idx += 1

        # The decoded frame returned by aiortc will be a torch.Tensor which
        # is then passed into the pipeline.
        # The pipeline returns the processed frame as a torch.Tensor which
        # is then passed to aiortc to be encoded into a frame.
        frame = await self.track.recv()
        return self.pipeline(frame)
