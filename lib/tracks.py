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

        frame = await self.track.recv()
        return self.pipeline(frame)
