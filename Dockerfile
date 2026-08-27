# torch wheels bundle their own CUDA runtime, so the base image needs no CUDA —
# only the host driver plus `--gpus all`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# weights and outputs are mounted, never baked in (the AWQ build alone is 27.6 GB)
VOLUME ["/models", "/out"]

ENTRYPOINT ["duplex"]
CMD ["--help"]
