#!/bin/bash
# Run one command in the already-imported ROCm PyTorch image.
#
# n01 exposes Docker, while n07 exposes Spur's imported SquashFS image but no
# Docker CLI. On n07, extract the image once to node-local NVMe and launch it
# with rootless runc. This is direct SSH execution; it does not submit a Spur
# job or pull another image.
set -eu

IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
IMAGE_SQSH="/var/spool/spur/images/docker.io+rocm+pytorch+rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0.sqsh"
ROOTFS="${ROCKE_CONTAINER_ROOTFS:-/tmp/rocke-rocm724-rootfs}"
CWD="${ROCKE_CONTAINER_CWD:-/}"

[ "$#" -gt 0 ] || { echo "usage: $0 COMMAND [ARG ...]" >&2; exit 2; }

if command -v docker >/dev/null 2>&1; then
    RENDER_GID="$(getent group render | cut -d: -f3)"
    VIDEO_GID="$(getent group video | cut -d: -f3)"
    exec docker run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --user "$(id -u):$(id -g)" \
        --group-add "${VIDEO_GID}" \
        --group-add "${RENDER_GID}" \
        --ipc=host \
        --network=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v /ossci-storage:/ossci-storage \
        -w "${CWD}" \
        -e HOME=/tmp \
        -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
        -e ROCKE_DENSE_VPAD="${ROCKE_DENSE_VPAD:-32}" \
        -e ROCKE_DENSE_NBUF="${ROCKE_DENSE_NBUF:-2}" \
        -e ROCKE_LLVM_FLAVOR="${ROCKE_LLVM_FLAVOR:-llvm22}" \
        -e ROCKE_DEBUG_LOC="${ROCKE_DEBUG_LOC:-0}" \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONPATH="${PYTHONPATH:-}" \
        -e ROCPROF_TRACE_DECODER_LIB="${ROCPROF_TRACE_DECODER_LIB:-}" \
        "${IMAGE}" "$@"
fi

command -v runc >/dev/null 2>&1 || {
    echo "neither docker nor runc is available" >&2
    exit 1
}
[ -r "${IMAGE_SQSH}" ] || {
    echo "imported image is not readable at ${IMAGE_SQSH}" >&2
    exit 1
}

if [ ! -f "${ROOTFS}/.rocke_extract_complete" ]; then
    rm -rf "${ROOTFS}"
    mkdir -p "${ROOTFS}"
    unsquashfs -processors "$(nproc)" -f -d "${ROOTFS}" "${IMAGE_SQSH}"
    touch "${ROOTFS}/.rocke_extract_complete"
fi

BUNDLE="$(mktemp -d /tmp/rocke-runc-bundle.XXXXXX)"
STATE="$(mktemp -d /tmp/rocke-runc-state.XXXXXX)"
CID="rocke-$$-$(date +%s)"
cleanup() {
    runc --root "${STATE}" delete -f "${CID}" >/dev/null 2>&1 || true
    rm -rf "${BUNDLE}" "${STATE}"
}
trap cleanup EXIT INT TERM

cd "${BUNDLE}"
runc spec --rootless
python3 - "config.json" "${ROOTFS}" "${CWD}" "$@" <<'PY'
import json
import os
import sys

config_path, rootfs, cwd, *argv = sys.argv[1:]
with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

config["root"] = {"path": rootfs, "readonly": True}
config["process"]["terminal"] = False
config["process"]["args"] = argv
config["process"]["cwd"] = cwd

base_env = {
    "HOME": "/tmp",
    "PATH": (
        "/opt/rocm/bin:/opt/rocm/llvm/bin:/opt/venv/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/lib64",
}
for key in (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "ROCKE_DENSE_VPAD",
    "ROCKE_DENSE_NBUF",
    "ROCKE_LLVM_FLAVOR",
    "ROCKE_DEBUG_LOC",
    "ROCPROF_TRACE_DECODER_LIB",
):
    if key in os.environ:
        base_env[key] = os.environ[key]
config["process"]["env"] = [f"{k}={v}" for k, v in base_env.items()]

config["mounts"].extend(
    [
        {
            "destination": "/tmp",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "nodev", "mode=1777", "size=16g"],
        },
        {
            "destination": "/ossci-storage",
            "type": "none",
            "source": "/ossci-storage",
            "options": ["rbind", "rw"],
        },
        {
            "destination": "/dev/kfd",
            "type": "none",
            "source": "/dev/kfd",
            "options": ["bind", "rw"],
        },
        {
            "destination": "/dev/dri",
            "type": "none",
            "source": "/dev/dri",
            "options": ["rbind", "rw"],
        },
    ]
)

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PY

runc --root "${STATE}" run "${CID}"
