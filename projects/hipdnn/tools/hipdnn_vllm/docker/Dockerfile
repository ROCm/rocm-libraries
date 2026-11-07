# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# hipdnn-vllm runtime image: ubuntu:24.04 + a venv with the pinned vLLM rocm100
# wheel (which brings TheRock pip ROCm and torch) + one family's hipdnn-vllm
# wheel. The image adds no logic of its own.
#
# Build from a context directory holding the wheel and, under hipdnn_vllm_src/,
# BUILD_PINS.toml, patches/ and scripts/apply_vllm_patches.sh from this tool dir
# (the local vLLM patches, README "Local vLLM patches"):
#   docker build -f docker/Dockerfile \
#     --build-arg WHEEL=hipdnn_vllm-0.1.0+rocm10.0.0.gfx94x-cp312-cp312-linux_x86_64.whl \
#     -t hipdnn-vllm:0.1.0-gfx94X <context dir>
# Run (the user is not root, so it needs the host's GPU device groups):
#   docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host \
#     hipdnn-vllm:... vllm serve <model> --attention-backend HIPDNN
#
# The vLLM, torch and ROCm versions and the vLLM wheel index are read from
# BUILD_PINS.toml, the same file the wheel's requirements come from.

FROM ubuntu:24.04

ARG WHEEL
ARG PIP_VERSION=25.2

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates libnuma1 libgomp1 patch \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv \
 && pip install --no-cache-dir "pip==${PIP_VERSION}"

# BUILD_PINS.toml -> a pip constraints file and the vLLM index URL. Fails when
# the image's Python is not the pinned one.
COPY hipdnn_vllm_src/BUILD_PINS.toml /opt/hipdnn_vllm_src/BUILD_PINS.toml
RUN python -c "import sys, tomllib; \
p = tomllib.load(open('/opt/hipdnn_vllm_src/BUILD_PINS.toml', 'rb'))['pins']; \
v = '%d.%d' % sys.version_info[:2]; \
sys.exit(f'image Python {v} is not the pinned {p[\"python\"]}') if v != p['python'] else None; \
open('/opt/hipdnn_vllm_src/constraints.txt', 'w').write(''.join(f'{k}=={p[k]}\n' for k in ('vllm', 'torch', 'rocm'))); \
open('/opt/hipdnn_vllm_src/vllm_index', 'w').write(p['vllm_index'])" \
 && cat /opt/hipdnn_vllm_src/constraints.txt

RUN pip install --no-cache-dir --extra-index-url "$(cat /opt/hipdnn_vllm_src/vllm_index)" \
      -c /opt/hipdnn_vllm_src/constraints.txt vllm torch "rocm[devel]"

COPY hipdnn_vllm_src /opt/hipdnn_vllm_src
RUN bash /opt/hipdnn_vllm_src/scripts/apply_vllm_patches.sh python

# With its dependencies, under the same constraints: the wheel requires the
# pinned vllm, torch and rocm, so pip only confirms they are installed, and a
# wheel built against other pins fails here instead of at run time.
COPY ${WHEEL} /tmp/
RUN test -n "${WHEEL}" \
 && pip install --no-cache-dir --extra-index-url "$(cat /opt/hipdnn_vllm_src/vllm_index)" \
      -c /opt/hipdnn_vllm_src/constraints.txt "/tmp/${WHEEL}" \
 && rm -f "/tmp/${WHEEL}"

RUN useradd --create-home --shell /bin/bash hipdnn
USER hipdnn
WORKDIR /home/hipdnn

# `python -m hipdnn_vllm doctor` needs a GPU; run it in the container to verify.
CMD ["python", "-m", "hipdnn_vllm", "doctor"]
