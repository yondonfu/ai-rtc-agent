# Run

The following options for running the agent are supported:

- [Local](./local.md)
- [Docker](./docker.md)

In order for the agent to connect with another peer (i.e. a browser) without any additional dependencies you will need to allow inbound/outbound
UDP traffic on ports 1024-65535 ([source](https://github.com/aiortc/aiortc/issues/490#issuecomment-788807118)). If you are running the agent in a restrictive network environment where this is not possible, you will need to use a [TURN server](https://bloggeek.me/webrtc-turn/).

At the moment, the agent supports using Twilio's TURN servers (although it is easy to make the update to support arbitrary TURN servers):

1. Sign up for a [Twilio](https://www.twilio.com/en-us) account.
2. Copy the Account SID and Auth Token from https://console.twilio.com/.
3. Set the `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` environment variables.

```
# Local
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
python agent.py

# Docker
docker run -e TWILIO_ACCOUNT_SID=... -e TWILIO_AUTH_TOKEN=... --gpus all --network="host" -v ./models:/models ai-rtc-agent:latest
```

The agent does not support using other TURN servers at the moment, but it is easy to make the change in `agent.py` by updating the `iceServers` param in the `RTCConfiguration` when creating a `RTCPeerConnection`.

The agent can be configured with additional [environment variables](./environment.md).