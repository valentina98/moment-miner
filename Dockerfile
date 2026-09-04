# Runtime image: works locally (docker run --gpus all) and as a Vast.ai
# instance image. Build with:  docker build --target runtime -t moment-miner .
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY moment_miner ./moment_miner
RUN pip install --no-cache-dir -e ".[asr,siglip,smartcut]"

# Mount archive at /videos (read-only) and a persistent volume at /data.
ENTRYPOINT ["mm", "--data-dir", "/data"]
CMD ["--help"]

# Dev/test image (default build target):  docker build -t moment-miner:dev .
FROM runtime AS dev
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[asr,siglip,smartcut,dev]"
ENTRYPOINT []
CMD ["pytest", "-v"]
