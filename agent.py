import asyncio
import json
import argparse
import os
import logging
import uuid
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
from aiortc.contrib.media import MediaRelay
from aiohttp_middlewares import cors_middleware

from lib.pipeline import StreamDiffusionPipeline
from lib.tracks import VideoStreamTrack
from lib.events import StreamEventHandler

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


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]
    stream_event_handler = request.app["stream_event_handler"]

    params = await request.json()

    room_id = params["room_id"]
    stream_id = str(uuid.uuid4())

    offer_params = params["offer"]
    offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])

    # pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=get_ice_servers()))
    pc = RTCPeerConnection()
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


async def whep(request):
    if request.method == "DELETE":
        return web.Response(status=200)

    if request.content_type != "application/sdp":
        return web.Response(status=400)

    source_track = request.app["state"].get("source_track", None)
    if source_track is None:
        return web.Response(status=401)

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

    # track = relay.subscribe(source_track, buffered=False)
    sender = pc.addTrack(source_track)

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


async def whip(request):
    if request.method == "DELETE":
        return web.Response(status=200)

    if request.content_type != "application/sdp":
        return web.Response(status=400)

    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    offer = await request.text()

    offer = RTCSessionDescription(sdp=offer, type="offer")

    # TURN does not work when connecting with OBS because OBS currently does not support trickle ICE meaning it
    # will not send ICE candidates after the initial SDP exchange during signaling. The TURN workflow requires a client behind a NAT
    # to send a STUN binding request to the TURN server in order to get its IP:port which can be sent as a candidate to WHIP server so
    # that it can create a permission with the TURN server for the allocated Relayed Transport Address. Since OBS does not send ICE candidates
    # after signaling, this permission creation process is not possible.
    #
    # In order to avoid using TURN we do two things:
    #
    # 1. The WHIP server uses STUN to discover its public IP.
    # aiortc should automatically use a STUN server if RTCPeerConnection is initialized without a config.
    # Another option would be to discover the public IP by other means and then modify candidates like what broadcast-box + Pion do:
    # https://github.com/pion/webrtc/blob/5bf7c9465c794763812e3ec2aaf6d29815a87e27/settingengine.go#L240
    # https://github.com/Glimesh/broadcast-box/blob/5d538c24077d0c8d13dea3ecb7bd5dd1ed634098/internal/webrtc/webrtc.go#L196
    #
    # 2. Force specific UDP media ports that definitely support inbound/oubound traffic instead of using random ephemeral ports
    # This is done by monkeypatching aioice using patch_loop_datagram
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Prefer h264
    transceiver = pc.addTransceiver("video")
    caps = RTCRtpSender.getCapabilities("video")
    prefs = list(filter(lambda x: x.name == "H264", caps.codecs))
    transceiver.setCodecPreferences(prefs)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        async def on_message(message):
            logger.info(f"received config: {message}")
            config = json.loads(message)

            t_index_list = config.get("t_index_list", None)
            if t_index_list is not None:
                pipeline.update_t_index_list(t_index_list)

            prompt = config.get("prompt", None)
            if prompt is not None:
                pipeline.update_prompt(prompt)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            videoTrack = VideoStreamTrack(track, pipeline)
            request.app["state"]["source_track"] = videoTrack

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
            "Location": "/whip",
        },
        text=answer.sdp,
    )


# websocket would be preferrable but the client in TouchDesigner is not working right now
# async def update_config(request):
#     pipeline = request.app["pipeline"]

#     ws = web.WebSocketResponse()
#     await ws.prepare(request)

#     async for msg in ws:
#         if msg.type == WSMsgType.TEXT:
#             if msg.data == "close":
#                 await ws.close()
#             else:
#                 config = json.loads(msg.data)
#                 logger.info(f"received config: {config}")

#                 t_index_list = config.get("t_index_list", None)
#                 if t_index_list is not None:
#                     pipeline.update_t_index_list(t_index_list)

#                 prompt = config.get("prompt", None)
#                 if prompt is not None:
#                     pipeline.update_prompt(prompt)
#         elif msg.type == WSMsgType.ERROR:
#             print("ws connection closed with exception %s" % ws.exception())

#     print("ws connection closed")

#     return ws


async def update_config(request):
    config = await request.json()
    logger.info(f"received config: {config}")

    pipeline = request.app["pipeline"]

    t_index_list = config.get("t_index_list", None)
    if t_index_list is not None:
        pipeline.update_t_index_list(t_index_list)

    prompt = config.get("prompt", None)
    if prompt is not None:
        pipeline.update_prompt(prompt)

    return web.Response(content_type="application/json", text="OK")


def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app):
    if app["udp_ports"]:
        patch_loop_datagram(app["udp_ports"])

    app["pipeline"] = StreamDiffusionPipeline(app["model_id"])
    app["pcs"] = set()
    app["stream_event_handler"] = StreamEventHandler()

    app["relay"] = MediaRelay()
    # aiohttp will emit a deprecation warning for mutating top-level state on the
    # app object so we store mutable state in an object and update the object instead
    app["state"] = {"source_track": None}


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
    app["udp_ports"] = args.udp_ports.split(",")
    app["model_id"] = args.model_id

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_post("/whip", whip)
    app.router.add_delete("/whip", whip)
    app.router.add_post("/whep", whep)
    app.router.add_delete("/whep", whep)
    app.router.add_post("/offer", offer)
    app.router.add_post("/config", update_config)
    app.router.add_get("/", health)

    web.run_app(app, host="0.0.0.0", port=args.port)
