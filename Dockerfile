FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY ./runpod-worker/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./runpod-worker/ .

CMD ["python", "handler.py"]
