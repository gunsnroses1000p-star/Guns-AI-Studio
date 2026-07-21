FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    ninja-build \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# WAN 2.2
# ============================================================

RUN git clone --depth 1 \
    https://github.com/Wan-Video/Wan2.2.git \
    /opt/Wan2.2

# ============================================================
# ENVIRONMENT
# ============================================================

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/Wan2.2:${PYTHONPATH}"

# ============================================================
# PYTHON BUILD TOOLS
# ============================================================

RUN pip install --no-cache-dir -U \
    pip \
    setuptools \
    wheel \
    packaging \
    ninja

# ============================================================
# WAN REQUIREMENTS
# Install everything except Flash Attention first
# ============================================================

RUN grep -v -i "flash_attn\|flash-attn" \
    /opt/Wan2.2/requirements.txt \
    > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir \
    -r /tmp/wan-requirements.txt \
    && rm -f /tmp/wan-requirements.txt

# ============================================================
# FLASH ATTENTION
# ============================================================

RUN MAX_JOBS=4 pip install \
    --no-cache-dir \
    flash-attn \
    --no-build-isolation

# ============================================================
# RUNPOD + IMAGE PIPELINES
# ============================================================

RUN pip install --no-cache-dir \
    runpod \
    requests \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub \
    decord \
    librosa \
    einops \
    peft

# ============================================================
# VERIFY IMPORTANT IMPORTS DURING BUILD
# ============================================================

RUN python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda)"

RUN python -c "import flash_attn; print('Flash Attention:', flash_attn.__version__)"

RUN python -c "import runpod; import diffusers; import transformers; print('Core imports OK')"

# ============================================================
# COPY SERVERLESS WORKER
# ============================================================

COPY . /app

# ============================================================
# START RUNPOD WORKER
# ============================================================

CMD ["python", "-u", "/app/handler.py"]
