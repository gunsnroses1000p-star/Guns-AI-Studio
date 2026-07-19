FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/Wan2.2"

# Install Wan dependencies except flash-attn first.
RUN grep -v -i "flash_attn\|flash-attn" \
    /opt/Wan2.2/requirements.txt > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir -r /tmp/wan-requirements.txt

# Install build dependencies and Flash Attention 2.
RUN pip install --no-cache-dir packaging ninja \
    && MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# RunPod serverless runtime.
RUN pip install --no-cache-dir runpod

# Extra Wan dependencies previously required by the worker.
RUN pip install --no-cache-dir decord librosa einops peft

# Verify Flash Attention during the Docker build.
RUN python -c "import flash_attn; print('Flash Attention OK:', flash_attn.__version__)"

COPY . .

CMD ["python", "-u", "handler.py"]
