FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/Wan2.2"

RUN grep -v -i "flash_attn\|flash-attn" /opt/Wan2.2/requirements.txt > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir -r /tmp/wan-requirements.txt

# Runpod serverless runtime (required by handler.py)
RUN pip install --no-cache-dir runpod

# Wan deps seen missing in logs
RUN pip install --no-cache-dir decord librosa einops

# Optional: if you must align torch/torchvision to official cu126 wheels, do this (NO torchaudio):
# RUN pip install --no-cache-dir --upgrade --force-reinstall \
#     torch torchvision \
#     --index-url https://download.pytorch.org/whl/cu126

COPY . .
CMD ["python", "-u", "handler.py"]



