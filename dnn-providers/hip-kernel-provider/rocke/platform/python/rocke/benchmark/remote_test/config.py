# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static config for the ROCKE remote-test orchestrator.

Kept as plain Python (no PyYAML dep) so it imports cleanly inside the
rocke package. Override any field on the CLI with --set key=value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


CK_PY_ROOT = (
    Path(__file__).resolve().parents[3]
)  # the Python package root (holds rocke)
REPO_ROOT = CK_PY_ROOT.parents[
    2
]  # repo root (relative; remote-test orchestration only)
LOCAL_STAGE_ROOT = Path("/tmp/rocke_remote")


@dataclass(frozen=True)
class ArchProfile:
    """Per-arch build + slurm dispatch info."""

    arch: str  # e.g. "gfx942"
    example_module: str  # rocke.examples.<arch>.<mod>
    example_args: List[str]  # extra CLI args for the build
    slurm_constraint: str  # sinfo AVAIL_FEATURES token (e.g. "GFX942")
    # gres uses generic gpu:1 + a feature constraint; specific gres names
    # (gfx942-mi300x:1 etc.) vary per node so we keep it generic.
    slurm_gres: str = "gpu:1"
    slurm_time: str = "00:15:00"
    slurm_partition: str = "defq"


@dataclass(frozen=True)
class RemoteHost:
    # `host` is either a real hostname OR an SSH alias defined in the
    # user's ~/.ssh/config. In alias mode `user` is empty and we let
    # ssh_config handle User / IdentityFile / ProxyJump / etc.
    user: str
    host: str
    # Staging dir on the login node (shared FS reachable from compute nodes).
    # `~` is fine — ssh expands it on the remote side.
    stage_root: str
    # Only used when not in alias mode. Empty string => don't pass `-i`.
    ssh_key: str = ""


_REMOTE_HOST = os.environ.get("ROCKE_REMOTE_HOST", "")
# Default to empty so the orchestrator stays portable. With no ROCKE_REMOTE_USER
# set, ROCKE_REMOTE_HOST is treated as an ~/.ssh/config alias (user, key,
# ProxyJump, etc. all come from ssh_config). Set ROCKE_REMOTE_USER to opt into
# explicit user@host mode when pointing at a bare hostname.
_REMOTE_USER = os.environ.get("ROCKE_REMOTE_USER", "")
_ALIAS_MODE = bool(_REMOTE_HOST) and not _REMOTE_USER

# In alias mode, leave ssh_key empty so ssh_config's IdentityFile wins.
# Outside alias mode, allow an explicit override but don't default to a
# site-specific path.
_SSH_KEY = os.environ.get(
    "ROCKE_REMOTE_KEY",
    "" if _ALIAS_MODE else str(Path(os.environ.get("HOME", "")) / ".ssh/id_ed25519"),
)

REMOTE = RemoteHost(
    user=_REMOTE_USER,
    host=_REMOTE_HOST,
    stage_root=os.environ.get("ROCKE_REMOTE_STAGE", "~/rocke_remote"),
    ssh_key=_SSH_KEY,
)


# Docker image used inside the srun step to run the verifier. The compute
# node's docker daemon must already have this image. Empty => skip docker
# and run the verifier directly under bash (only works when /dev/kfd is
# accessible bare). Typical value: f"{user}_rock:latest".
DOCKER_IMAGE = os.environ.get("ROCKE_DOCKER_IMAGE", "")
# Extra `docker run` flags appended after the base flag set, before the
# image name. Space-separated, parsed with shlex.
DOCKER_EXTRA_FLAGS = os.environ.get("ROCKE_DOCKER_EXTRA_FLAGS", "")
# Extra `srun` args appended after the base flags (space-separated, shlex
# parsed). Useful to pin/avoid specific nodes, e.g.
# ROCKE_SLURM_EXTRA="--nodelist=ctr-halo-b48-02" or "--exclude=ctr-halo-b48-01"
# when some matching nodes have their GPU bound to vfio-pci, not amdgpu.
SLURM_EXTRA = os.environ.get("ROCKE_SLURM_EXTRA", "")


ARCHES: Dict[str, ArchProfile] = {
    # $HOME on the login node is only NFS-shared to MARKHAM compute nodes;
    # constrain to MARKHAM so the staged dir is visible inside srun.
    "gfx942": ArchProfile(
        arch="gfx942",
        example_module="rocke.examples.gfx942.gemm_demo",
        example_args=["--m", "256", "--n", "256", "--k", "256"],
        slurm_constraint="GFX942",
    ),
    "gfx1151": ArchProfile(
        arch="gfx1151",
        example_module="rocke.examples.gfx1151.gemm.scripts.02_int4_matmul_nbits_verify",
        example_args=[
            "--m",
            "128",
            "--n",
            "4096",
            "--k",
            "4096",
            "--group-size",
            "32",
            "--family",
            "large_n",
            "--seq-len-tile",
            "64",
        ],
        slurm_constraint="GFX1151",
    ),
    # RDNA4 / Navi 48. MARKHAM nodes (NFS-visible HOME), ROCm 7.2.0. Lane map is
    # hardware-confirmed (Stage 3 probe passed bit-exact), so the default example
    # is now the MatMulNBits large-N verify on a real Qwen3.5-9B shape
    # (attn out_proj: N=4096, K=4096, int4/g32). Swap N/K via example_args to
    # cover the other large-N geometries (12288x4096, 4096x12288, 8192x4096,
    # 1024x4096). N=32 (skinny_n) and N=248320/M=1 (decode_gemv) need the
    # unimplemented families. Re-point example_module to wmma_probe to re-run the
    # lane-map probe.
    "gfx1201": ArchProfile(
        arch="gfx1201",
        example_module="rocke.examples.gfx1201.matmul_nbits_verify",
        example_args=[
            "--m",
            "128",
            "--n",
            "4096",
            "--k",
            "4096",
            "--group-size",
            "32",
            "--family",
            "large_n",
            "--seq-len-tile",
            "64",
        ],
        slurm_constraint="GFX1201",
        # 12h hold: gfx1201 nodes are scarce, so reserve long and reuse via
        # rsync + srun. (The 15min default was killing the allocation early.)
        slurm_time="12:00:00",
    ),
}


def stage_dir(arch: str) -> Path:
    return LOCAL_STAGE_ROOT / arch


def remote_stage_root() -> str:
    # Lazy import avoids a transport <-> config import cycle at module load.
    from . import transport

    return transport.expand_remote(REMOTE.stage_root)


def remote_stage_dir(arch: str) -> str:
    return f"{remote_stage_root()}/{arch}"
