FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-distutils ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

COPY requirements.txt /app/

RUN python3.11 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=10000
EXPOSE 10000

CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--timeout-keep-alive", "900", "--timeout-graceful-shutdown", "900"]