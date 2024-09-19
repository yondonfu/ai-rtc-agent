import runpod
import requests
import time
import json
import os

AGENT_HEALTH_ENDPOINT = "http://localhost:8888"
AGENT_TIMEOUT = 60 * 10  # 10 minutes


def check_server(url, poll_timeout=60, poll_interval=0.5) -> bool:
    start_time = time.time()

    while time.time() - start_time < poll_timeout:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                print("agent ready")
                return True
        except requests.RequestException as e:
            pass

        time.sleep(poll_interval)

    print("could not connect to agent")

    return False


def handler(job):
    job_input = job["input"]
    agent_timeout = job_input.get("agent_timeout")
    if agent_timeout is None:
        agent_timeout = AGENT_TIMEOUT

    print(f"agent timeout: {agent_timeout}s")

    # Ensure that the agent is ready
    check_server(AGENT_HEALTH_ENDPOINT)

    update = {
        "pod_id": os.getenv("RUNPOD_POD_ID"),
        "public_ip": os.getenv("RUNPOD_PUBLIC_IP"),
        "public_port": os.getenv("RUNPOD_TCP_PORT_8888"),
    }
    # Send progress update so client can check if agent is running
    runpod.serverless.progress_update(job, json.dumps(update))

    # Keep job active
    time.sleep(agent_timeout)

    return {"status": "SUCCESS"}


runpod.serverless.start({"handler": handler})
