# syntax=docker/dockerfile:1.7

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54
FROM ${ISAAC_SIM_IMAGE}

ARG EBIM_REPOSITORY=https://github.com/EBiM-Benchmark/benchmark.git
ARG EBIM_COMMIT=cb5184574f33611f943ff42aae461678ccb538e9
ARG AIRSIGN_REVISION=unavailable
ARG ROOM_ASSET_SHA256=696c71577f1874d815fe29c6a58c65f0f1a0a0fb15c0d8adbb5105209f5ff883
ARG ROBOT_ASSET_COMMIT=c2439d961b652b1eda6122bf530c58cb9559b219
ARG ROBOT_ASSET_SHA256=aa1a833de48cc543c73957461dab82fe0979320b7c0b6a0a113d24b500075e5c

LABEL org.opencontainers.image.title="AirSignRobot — EBiM Task 3"
LABEL org.opencontainers.image.description="AirSign actuator-driven participant workload for the EBiM Assisted Living and Feeding task"
LABEL org.opencontainers.image.source="https://github.com/EvergreenTree/AirSignRobot"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54"
LABEL org.opencontainers.image.revision="${AIRSIGN_REVISION}"
LABEL org.airsignrobot.ebim.revision="${EBIM_COMMIT}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

ENV EBIM_ROOT=/workspace/EBiM_Challenge \
    EBIM_COMMIT=${EBIM_COMMIT} \
    AIRSIGN_REVISION=${AIRSIGN_REVISION} \
    ROOM_ASSET_SHA256=${ROOM_ASSET_SHA256} \
    ROBOT_ASSET_SHA256=${ROBOT_ASSET_SHA256} \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    HOME=/isaac-sim \
    XDG_CACHE_HOME=/isaac-sim/.cache \
    XDG_CONFIG_HOME=/isaac-sim/.nvidia-omniverse/config \
    XDG_DATA_HOME=/isaac-sim/.local/share \
    OMNI_KIT_ALLOW_ROOT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Fetch the exact official development snapshot used during validation. The
# competition Robotiq USD is restored from a separately pinned official commit
# and checksum-verified so the build fails instead of silently using a drifted
# asset.
RUN git init "${EBIM_ROOT}" \
    && git -C "${EBIM_ROOT}" remote add origin "${EBIM_REPOSITORY}" \
    && git -C "${EBIM_ROOT}" fetch --depth=1 origin "${EBIM_COMMIT}" \
    && git -C "${EBIM_ROOT}" -c advice.detachedHead=false checkout --detach "${EBIM_COMMIT}" \
    && test "$(git -C "${EBIM_ROOT}" rev-parse HEAD)" = "${EBIM_COMMIT}" \
    && git -C "${EBIM_ROOT}" lfs install --local \
    && git -C "${EBIM_ROOT}" lfs pull origin \
        --include="assets/robot_room.usd" \
        --exclude="" \
    && printf '%s  %s\n' \
        "${ROOM_ASSET_SHA256}" \
        "${EBIM_ROOT}/assets/robot_room.usd" \
        | sha256sum --check --strict \
    && git -C "${EBIM_ROOT}" fetch --depth=1 origin "${ROBOT_ASSET_COMMIT}" \
    && mkdir -p "${EBIM_ROOT}/task1_isaacsim/assets" \
    && git -C "${EBIM_ROOT}" show \
        "${ROBOT_ASSET_COMMIT}:DEMO/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
        > "${EBIM_ROOT}/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
    && printf '%s  %s\n' \
        "${ROBOT_ASSET_SHA256}" \
        "${EBIM_ROOT}/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd" \
        | sha256sum --check --strict \
    && mkdir -p /opt/airsign \
    && printf '%s\n' "${EBIM_COMMIT}" > /opt/airsign/benchmark-commit \
    && printf '%s\n' "${AIRSIGN_REVISION}" > /opt/airsign/source-revision

ENV ROS_DISTRO=jazzy \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib \
    ROS_HOME=/tmp/isaac_ros_home

COPY participant/ ${EBIM_ROOT}/ebim-track3-solution/
COPY evidence/ /opt/airsign/evidence/
COPY scripts/airsign /usr/local/bin/airsign
COPY README.md LICENSE NOTICE /opt/airsign/

RUN chmod 0755 /usr/local/bin/airsign \
    "${EBIM_ROOT}/ebim-track3-solution/autonomous_probe.py" \
    "${EBIM_ROOT}/ebim-track3-solution/gripper_scene_gate2.py" \
    "${EBIM_ROOT}/ebim-track3-solution/stage1_table_setup.py" \
    "${EBIM_ROOT}/ebim-track3-solution/four_stage_rehearsal.py" \
    "${EBIM_ROOT}/ebim-track3-solution/validate_four_stage_replay.py" \
    && mkdir -p \
        /isaac-sim/.cache \
        /isaac-sim/.nvidia-omniverse/config \
        /isaac-sim/.local/share \
        /tmp/isaac_ros_home \
    && /usr/local/bin/airsign smoke

WORKDIR ${EBIM_ROOT}
ENTRYPOINT ["/usr/local/bin/airsign"]
CMD ["smoke"]
