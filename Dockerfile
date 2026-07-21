# ============================================================
# STAGE 1: BUILD FLASH ATTENTION
# ============================================================

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel AS builder

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U \
    pip \
    setuptools \
    wheel \
    packaging \
    ninja
    

# Build a Flash Attention wheel that we can copy into
# the smaller runtime image.
RUN mkdir -p /wheels \
    && MAX_JOBS=4 pip wheel \
        --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        flash-attn \
        -w /wheels


# ============================================================
# STAGE 2: FINAL RUNTIME IMAGE
# ============================================================

FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/opt/Wan2.2"

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
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
# PYTHON TOOLS
# ============================================================

RUN pip install --no-cache-dir -U \
    pip \
    setuptools \
    wheel

# ============================================================
# WAN REQUIREMENTS
# Install everything except Flash Attention
# ============================================================

RUN grep -v -i "flash_attn\|flash-attn" \
    /opt/Wan2.2/requirements.txt \
    > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir \
        -r /tmp/wan-requirements.txt \
    && rm -f /tmp/wan-requirements.txt

# ============================================================
# INSTALL PRECOMPILED FLASH ATTENTION FROM BUILDER
# ============================================================

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# ============================================================
# RUNPOD + HANDLER DEPENDENCIES
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
    protobuf \
    sentencepiece
    tokenizers
# ============================================================
# VERIFY INSTALLATION
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
