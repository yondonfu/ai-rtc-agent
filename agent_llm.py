import asyncio
import json
import argparse
import os
import logging
import random
import types

from typing import List, Tuple
from twilio.rest import Client
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
)
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.contrib.media import MediaPlayer
from aiohttp_middlewares import cors_middleware

from lib.pipeline import StreamDiffusionPipeline
from lib.tracks import VideoStreamTrack

logger = logging.getLogger(__name__)


# Original issue: https://github.com/aiortc/aioice/pull/63
# Copied from: https://github.com/toverainc/willow-inference-server/pull/17/files
def patch_loop_datagram(local_ports: List[int]):
    loop = asyncio.get_event_loop()
    if getattr(loop, "_patch_done", False):
        return

    # Monkey patch aiortc to control ephemeral ports
    old_create_datagram_endpoint = loop.create_datagram_endpoint

    async def create_datagram_endpoint(
        self, protocol_factory, local_addr: Tuple[str, int] = None, **kwargs
    ):
        # if port is specified just use it
        if local_addr and local_addr[1]:
            return await old_create_datagram_endpoint(
                protocol_factory, local_addr=local_addr, **kwargs
            )
        if local_addr is None:
            return await old_create_datagram_endpoint(
                protocol_factory, local_addr=None, **kwargs
            )
        # if port is not specified make it use our range
        ports = list(local_ports)
        random.shuffle(ports)
        for port in ports:
            try:
                ret = await old_create_datagram_endpoint(
                    protocol_factory, local_addr=(local_addr[0], port), **kwargs
                )
                logger.debug(f"create_datagram_endpoint chose port {port}")
                return ret
            except OSError as exc:
                if port == ports[-1]:
                    # this was the last port, give up
                    raise exc
        raise ValueError("local_ports must not be empty")

    loop.create_datagram_endpoint = types.MethodType(create_datagram_endpoint, loop)
    loop._patch_done = True


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


# Create Link headers for a list of ICE servers to be used for a WHIP endpoint
def get_link_headers(ice_servers: List[RTCIceServer]) -> List[str]:
    links = []
    for srv in ice_servers:
        url = srv.urls[0]
        link = f'<{url}>; rel="ice-server"; username="{srv.username}"; credential="{srv.credential}";'
        links.append(link)

    return links


async def whep(request):
    if request.method == "DELETE":
        return web.Response(status=200)

    if request.content_type != "application/sdp":
        return web.Response(status=400)

    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    offer = await request.text()

    offer = RTCSessionDescription(sdp=offer, type="offer")

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    player = MediaPlayer("test-fluid-sim.mp4", loop=True)
    track = VideoStreamTrack(player.video, pipeline)
    sender = pc.addTrack(track)

    codec = "video/H264"
    force_codec(pc, sender, codec)

    await pc.setRemoteDescription(offer)

    # Required to support OBS WHIP
    # As of OBS 30.2.0-beta4, OBS will NOT gather local ICE candidates before sending a SDP offer via WHIP.
    # This means that the WHIP endpoint impl *must* gather ICE candidates before responding with its SDP answer
    # in order to establish a connection.
    # See this issue for more details: https://github.com/obsproject/obs-studio/issues/10910
    # aiortc.RTCPeerConnection does not expose __gather() as a public method so as a workaround we
    # use the class's name-mangled version of the method.
    await pc._RTCPeerConnection__gather()

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Link headers based on this example:
    # https://github.com/Sean-Der/whip-turn-test/blob/master/main.go
    # We don't use Link headers for now
    # links = get_link_headers(ice_servers)
    return web.Response(
        status=201,
        content_type="application/sdp",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            # "Link": ",".join(links),
            "Location": "/whep",
        },
        text=answer.sdp,
    )


def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app):
    if app["udp_ports"]:
        patch_loop_datagram(app["udp_ports"])

    app["pipeline"] = StreamDiffusionPipeline(app["model_id"])
    app["pcs"] = set()


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
        "--udp-ports", default=None, help="Set the UDP ports to receive WebRTC media on"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    app = web.Application(middlewares=[cors_middleware(allow_all=True)])
    app["udp_ports"] = args.udp_ports.split(",") if args.udp_ports else None
    app["model_id"] = args.model_id

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_post("/whep", whep)
    app.router.add_delete("/whep", whep)
    app.router.add_get("/", health)

    web.run_app(app, host="0.0.0.0", port=args.port)
