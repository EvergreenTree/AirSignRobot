FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
WORKDIR /opt/airsign
COPY pyproject.toml ./
COPY ebim_phase2 ./ebim_phase2
RUN python -m pip install --no-cache-dir '.[sim,vision]'
COPY config ./config
COPY docs ./docs
COPY scripts ./scripts
COPY README.md run.sh ./
LABEL org.opencontainers.image.title="AirSign EBiM Phase II"
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "ebim_phase2.runtime"]
