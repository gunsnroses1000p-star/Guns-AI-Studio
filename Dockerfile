FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install official Wan 2.2 native inference code
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2

# Install Wan dependencies
RUN pip install --no-cache-dir -r /opt/Wan2.2/requirements.txt

# Install our RunPod worker dependencies
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt
RUN pip uninstall -y torchaudio || true
COPY . .

ENV PYTHONPATH="/opt/Wan2.2"

CMD ["python", "-u", "handler.py"]

