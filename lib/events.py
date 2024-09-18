import os
import time
import requests
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WebhookEvent(BaseModel):
    stream_id: str
    room_id: str
    timestamp: int


class StreamStartedEvent(WebhookEvent):
    event: str = "StreamStarted"


class StreamEndedEvent(WebhookEvent):
    event: str = "StreamEnded"


class StreamEventHandler:
    def __init__(self):
        self.webhook_url = os.getenv("WEBHOOK_URL")
        self.token = os.getenv("AUTH_TOKEN")

    def send_request(self, event_name: str, stream_id: str, room_id: str):
        if self.webhook_url is None or self.token is None:
            return

        if event_name == "StreamStarted":
            event = StreamStartedEvent(
                stream_id=stream_id, room_id=room_id, timestamp=int(time.time())
            )
        elif event_name == "StreamEnded":
            event = StreamEndedEvent(
                stream_id=stream_id, room_id=room_id, timestamp=int(time.time())
            )
        else:
            raise Exception("unknown event")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        res = requests.post(self.webhook_url, headers=headers, json=event.dict())

        if res.status_code != 200:
            logger.error(f"failed to send {event_name} event with {res.status_code}")

    def handle_stream_started(self, stream_id: str, room_id: str):
        return self.send_request("StreamStarted", stream_id, room_id)

    def handle_stream_ended(
        self,
        stream_id: str,
        room_id: str,
    ):
        return self.send_request("StreamEnded", stream_id, room_id)
