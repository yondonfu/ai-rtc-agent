# Runpod Serverless Worker

This directory contains files to support running the agent as a [Runpod Serverless worker](https://www.runpod.io/serverless-gpu).

When the handler is invoked, the agent will be started in the background and the job will be kept alive for `AGENT_TIMEOUT` so that peer connections can be made with the agent.
