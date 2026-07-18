FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install official Wan 2.2 native inference code
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Install Wan dependencies
RUN pip install --no-cache-dir -r /opt/Wan2.2/requirements.txt

# Install Wan dependencies without flash-attn
RUN grep -v -i "flash_attn\|flash-attn" /opt/Wan2.2/requirements.txt > /tmp/wan-requirements.txt \
    && pip install --no-cache-dir -r /tmp/wan-requirements.txt

RUN pip uninstall -y torchaudio || true
COPY . .

ENV PYTHONPATH="/opt/Wan2.2"

CMD ["python", "-u", "handler.py"]

