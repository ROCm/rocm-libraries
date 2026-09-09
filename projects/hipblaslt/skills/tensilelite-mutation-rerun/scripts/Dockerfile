# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Optional environment for manual TensileLite mutation analysis.

FROM rocm/dev-ubuntu-22.04:latest

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        python3-dev \
        python3-pip \
        python3-venv \
        util-linux \
    && python3 -m venv /opt/tl-mut-tools \
    && /opt/tl-mut-tools/bin/python -m pip install \
        --no-cache-dir --upgrade pip "tox>=4,<5" \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm \
    TOX_WORK_DIR=/opt/tl-mut \
    PATH=/opt/tl-mut-tools/bin:/opt/tl-mut/mutation-unit/bin:${PATH}

WORKDIR /work/projects/hipblaslt/tensilelite

CMD ["sleep", "infinity"]
