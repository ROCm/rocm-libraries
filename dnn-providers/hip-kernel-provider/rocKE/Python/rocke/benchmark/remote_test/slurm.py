# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Push staged artifacts + a slim copy of the rocke python tree to the
login node, then srun the manifest runner on a compute node that matches
the requested arch constraint.

Output from srun is streamed back over the SSH session.
"""

from __future__ import annotations

import json
import shlex

from . import config, transport


# rocke/__init__ pulls in helpers->analysis->benchmark->... so it's easier
# (and still small) to mirror the whole package minus heavy / unrelated dirs.
_PKG_EXCLUDES = (
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=examples",
    "--exclude=dsl_docs",
    # The remote runs its own /usr/bin/python3 (numpy present; run_manifest's
    # torch launcher falls back to direct HIP timing), so the local virtualenv
    # is never used on the node -- pushing it is pure waste (~15 GB of torch
    # aotriton images) and trips remote path/quota limits during rsync.
    "--exclude=.venv",
)


def push_rocke_tree() -> str:
    """Rsync rocke/ to <stage_root>/ck_pkg/rocke/ on the login node."""
    remote_pkg = f"{config.remote_stage_root()}/ck_pkg"
    transport.ssh_run(["mkdir", "-p", f"{remote_pkg}/rocke"])
    transport.rsync_push(
        config.CK_PY_ROOT / "rocke",
        f"{remote_pkg}/rocke",
        delete=True,
        extra=_PKG_EXCLUDES,
    )
    return remote_pkg


def push_artifacts(arch: str) -> str:
    local = config.stage_dir(arch)
    remote = config.remote_stage_dir(arch)
    transport.rsync_push(local, remote, delete=True, extra=("--exclude=__pycache__",))
    return remote


def _build_srun_argv(
    arch: str, remote_pkg: str, remote_art: str, run_spec: dict
) -> str:
    profile = config.ARCHES[arch]
    shape = run_spec.get("shape", {})
    shape_str = f"{shape.get('m', 0)},{shape.get('n', 0)},{shape.get('k', 0)}"
    hsaco = f"{remote_art}/{run_spec['hsaco']}"
    manifest = f"{remote_art}/{run_spec['manifest']}"
    # Work script that runs inside the container (or bare bash if no image).
    # Preflight: HSA must be able to enumerate a GPU agent. If only the
    # CPU agent shows up, no container-side fix will help — the host's
    # amdgpu DRM driver hasn't bound to the GPU (common when the DKMS
    # driver predates a new silicon's PCI ID, e.g. gfx1151 / Strix Halo
    # PCI 0x1586 requires ROCm 6.4+). Fail with an actionable message
    # instead of letting `hipInit` return the opaque hipError(100).
    preflight = (
        "if command -v rocminfo >/dev/null && "
        "! rocminfo 2>/dev/null | grep -q 'Device Type: *GPU'; then "
        "echo '[remote] FATAL: rocminfo sees no GPU agent on this node.' >&2; "
        "echo '[remote] amdgpu likely not bound to the device (check /sys/class/drm,' >&2; "
        "echo '[remote] lspci -k for the AMD Display controller, and the host driver version).' >&2; "
        "exit 42; fi; "
    )
    work = (
        f"set -e; cd {shlex.quote(remote_pkg)}; "
        f"export PYTHONPATH={shlex.quote(remote_pkg)}:$PYTHONPATH; "
        f"echo '[remote] host='$(hostname)' arch={arch}'; "
        + preflight
        + f"python3 -m rocke.run_manifest {shlex.quote(hsaco)} "
        f"{shlex.quote(manifest)} --shape {shape_str} --verify"
    )
    if config.DOCKER_IMAGE:
        # Plain docker on the compute node. /dev/kfd is required; /dev/dri
        # is optional — compute-only ROCm nodes (e.g. gfx1151 Halo) expose
        # only kfd and `docker run --device /dev/dri` then fails with
        # "no such file or directory". The DRI_FLAGS shell var below is
        # filled in only when /dev/dri actually exists on the host.
        docker = [
            "docker",
            "run",
            "--rm",
            "--network=host",
            "--device=/dev/kfd",
            "$DRI_FLAGS",
            "-v",
            "/var/lib/docker/:/var/lib/docker",
            "--group-add",
            "video",
            "--ipc=host",
            "-v",
            "$HOME:$HOME",
            "-w",
            remote_pkg,
        ]
        if config.DOCKER_EXTRA_FLAGS:
            docker += shlex.split(config.DOCKER_EXTRA_FLAGS)
        docker += [config.DOCKER_IMAGE, "/bin/bash", "-lc", work]
        # Tokens that must reach the remote shell unquoted (so $VAR
        # expansion happens there, not via shlex.quote on this side).
        _unquoted = {"$HOME:$HOME", "$DRI_FLAGS"}
        docker_str = " ".join(a if a in _unquoted else shlex.quote(a) for a in docker)
        # Probe /dev/dri on the compute node and only forward the device
        # flags if it actually exists.
        inner = (
            'DRI_FLAGS=""; '
            '[ -e /dev/dri ] && DRI_FLAGS="--device /dev/dri:/dev/dri:rw '
            '--volume /dev/dri:/dev/dri:rw"; ' + docker_str
        )
    else:
        inner = work
    srun = [
        "srun",
        f"--partition={profile.slurm_partition}",
        f"--constraint={profile.slurm_constraint}",
        f"--gres={profile.slurm_gres}",
        f"--time={profile.slurm_time}",
        "--job-name",
        f"rocke-{arch}",
        "--output",
        f"{remote_art}/srun.out",
        "--error",
        f"{remote_art}/srun.err",
        # Site-specific srun args (e.g. --nodelist / --exclude to steer
        # around nodes whose GPU is bound to vfio-pci instead of amdgpu).
        # Env-driven so nothing site-specific lands in the repo.
        *shlex.split(config.SLURM_EXTRA),
        "bash",
        "-lc",
        inner,
    ]
    return " ".join(shlex.quote(a) for a in srun)


def run_arch(arch: str) -> int:
    profile = config.ARCHES[arch]  # noqa: F841
    local = config.stage_dir(arch)
    spec_path = local / "run_spec.json"
    if not spec_path.exists():
        raise RuntimeError(f"missing {spec_path}; run build first")
    run_spec = json.loads(spec_path.read_text())

    remote_pkg = push_rocke_tree()
    remote_art = push_artifacts(arch)
    srun_cmd = _build_srun_argv(arch, remote_pkg, remote_art, run_spec)
    print(f"[slurm:{arch}] $ {srun_cmd}")
    r = transport.ssh_run(srun_cmd, check=False, stream=True)
    # Always pull the per-job logs back for inspection.
    tail = transport.ssh_run(
        f"tail -n +1 {shlex.quote(remote_art)}/srun.out "
        f"{shlex.quote(remote_art)}/srun.err 2>/dev/null || true",
        capture=True,
        check=False,
    )
    if tail.stdout:
        print(f"[slurm:{arch}] --- remote logs ---\n{tail.stdout}")
    return r.returncode
