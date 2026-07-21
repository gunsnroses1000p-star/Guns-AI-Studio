FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    ninja-build \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone Wan2.2 repo
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2

# CUDA env
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/Wan2.2"

# Python deps: upgrade pip tooling first (helps with builds)
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install Wan deps except flash-attn first
RUN grep -v -i "flash_attn\|flash-attn" /opt/Wan2.2/requirements.txt > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir -r /tmp/wan-requirements.txt

# Build/install Flash Attention 2
RUN pip install --no-cache-dir packaging ninja \
    && MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# Core runtime deps for your handler
# (diffusers stack + HF downloads + HTTP)
RUN pip install --no-cache-dir \
    runpod \
    requests \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub

# Extra Wan deps you mentioned
RUN pip install --no-cache-dir decord librosa einops peft

# Verify flash-attn import during build
RUN python -c "import flash_attn; print('Flash Attention OK:', flash_attn.__version__)"

# Copy your serverless worker code
COPY . .

# Serverless worker entrypoint
CMD ["python", "-u", "handler.py"]

