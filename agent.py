import asyncio
import json
import argparse
import os
import logging
import uuid

from twilio.rest import Client
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.rtcrtpsender import RTCRtpSender

from lib.pipeline import StreamDiffusionPipeline
from lib.tracks import VideoStreamTrack
from lib.events import StreamEventHandler

logger = logging.getLogger(__name__)


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecPrefs = [codec for codec in codecs if codec.mimeType == forced_codec]
    transceiver.setCodecPreferences(codecPrefs)


def get_twilio_token():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid is None or auth_token is None:
        return None

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token


def get_ice_servers():
    ice_servers = []

    token = get_twilio_token()
    if token is not None:
        # Use Twilio TURN servers
        for server in token.ice_servers:
            if server["url"].startswith("turn:"):
                turn = RTCIceServer(
                    urls=[server["urls"]],
                    credential=server["credential"],
                    username=server["username"],
                )
                ice_servers.append(turn)

    return ice_servers


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]
    stream_event_handler = request.app["stream_event_handler"]

    params = await request.json()

    room_id = params["room_id"]
    stream_id = str(uuid.uuid4())

    offer_params = params["offer"]
    offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=get_ice_servers()))
    pcs.add(pc)

    tracks = {"video": None}

    # Prefer h264
    transceiver = pc.addTransceiver("video")
    caps = RTCRtpSender.getCapabilities("video")
    prefs = list(filter(lambda x: x.name == "H264", caps.codecs))
    transceiver.setCodecPreferences(prefs)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        async def on_message(message):
            if tracks["video"]:
                logger.info(f"received config: {message}")
                config = json.loads(message)

                t_index_list = config.get("t_index_list", None)
                if t_index_list is not None:
                    pipeline.update_t_index_list(t_index_list)

                prompt = config.get("prompt", None)
                if prompt is not None:
                    pipeline.update_prompt(prompt)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            videoTrack = VideoStreamTrack(track, pipeline)
            tracks["video"] = videoTrack
            sender = pc.addTrack(videoTrack)

            codec = "video/H264"
            force_codec(pc, sender, codec)

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            stream_event_handler.handle_stream_ended(stream_id, room_id)
        elif pc.connectionState == "connected":
            stream_event_handler.handle_stream_started(stream_id, room_id)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app):
    app["pipeline"] = StreamDiffusionPipeline(app["model_id"])
    app["pcs"] = set()
    app["stream_event_handler"] = StreamEventHandler()


async def on_shutdown(app):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent")
    parser.add_argument(
        "--model-id", default="lykon/dreamshaper-8", help="Set the HuggingFace model ID"
    )
    parser.add_argument("--port", default=8888, help="Set the port to listen on")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    app = web.Application()
    app["model_id"] = args.model_id

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_post("/offer", offer)
    app.router.add_get("/", health)

    web.run_app(app, host="0.0.0.0", port=args.port)
