FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "handler.py"]

