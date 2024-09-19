# Deploy on Runpod

1. Sign up for a [Runpod](https://www.runpod.io/) account.
2. Sign up for a [Twilio](https://www.twilio.com/en-us) account to use their [TURN servers](https://www.twilio.com/docs/stun-turn).
    - Copy the Account SID and Auth Token from https://console.twilio.com/.
3. Use the [ai-rtc-agent template](https://www.runpod.io/console/deploy?template=2ke1zpx40y) to deploy a GPU pod.
    - Select a GPU.
        - Recommendation: Nvidia RTX 4090
    - Click "Edit Template".
        - Add the environment variable `TWILIO_ACCOUNT_SID` and paste your Twilio Account SID.
        - Add the environment variable `TWILIO_AUTH_TOKEN` and paste your Twilio Auth Token.
        - Click "Set Overrides"
    - Click "Deploy Pod"
4. Once the pod is up, click "Connect".
    - Click "Connect to HTTP Service".
    - The URL, which will look like `https://<POD_ID>-<PORT>.proxy.runpod.net`, can be used to connect to the agent.
