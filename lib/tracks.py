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
        self.drop_frames = int(os.getenv("DROP_FRAMES", 0))

    async def recv(self):
        while self.warmup_frame_idx < self.warmup_frames:
            logger.info(f"dropping warmup frames {self.warmup_frame_idx}")
            frame = await self.track.recv()
            self.pipeline(frame)
            self.warmup_frame_idx += 1

        # Drop frame which might help with playback in certain scenarios.
        # Dropping every other frame seems to address stuttering playback when receiving streams from OBS's x264 encoder
        # even though there is no playback issue when receiving streams from Chrome VideoToolBox without dropping frames.
        for _ in range(self.drop_frames):
            await self.track.recv()

        frame = await self.track.recv()
        # The decoded frame returned by aiortc will be a torch.Tensor if NVDEC is used and
        # an av.VideoFrame if a SW decoder is used.
        # The pipeline returns the processed frame as a torch.Tensor if NVENC is used and
        # an av.VideoFrame if a SW encoder is used.which
        return self.pipeline(frame)
