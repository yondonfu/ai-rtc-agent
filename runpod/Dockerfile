FROM ai-rtc-agent-weights:latest

WORKDIR /app

RUN pip install --no-cache-dir runpod 

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh

CMD ["bash", "start.sh"]