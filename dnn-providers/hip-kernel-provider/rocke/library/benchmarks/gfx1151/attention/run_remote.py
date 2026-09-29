# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the offline benchmark in a bounded, exclusive Slurm allocation.

SSH only transports a content-addressed source snapshot and its output. No
packages, kernels, datasets, or references are downloaded during a benchmark.
Site paths and hostnames live in a user-owned configuration outside the repo.
The Slurm job owns execution; disconnecting the client does not lose its output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid


_RUNTIME_SHA = "0b01ec5ccc55ca86059afb20f1f5764fee5edd9d7e7b456167d77dc9f691cdc9"
_IMAGES_SHA = "e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e"


def _snapshot(source, destination):
    roots = (source / "platform/python", source / "library")
    files = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part == "__pycache__" or part.endswith(".egg-info")
                for part in path.parts
            )
            and path.suffix not in (".pyc", ".pyo")
        )
    # The package asset resolver uses these source-tree markers.
    for relative in ("platform/pyproject.toml", "platform/CMakeLists.txt"):
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    digest = hashlib.sha256()
    with tarfile.open(destination, "w") as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(source).as_posix()
            payload = path.read_bytes()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
            member = tarfile.TarInfo("rocke/" + relative)
            member.size = len(payload)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))
    return digest.hexdigest()


def _job_script(config, source, build, digest):
    aot = PurePosixPath(config["aotriton_root"])
    rocm = PurePosixPath(config["rocm_path"])
    compute_root = PurePosixPath(config["compute_root"])
    bridge_source = source / "library/benchmarks/gfx1151/attention/aotriton_bridge.cpp"
    bridge = build / "aotriton_bridge.so"
    environment = {
        "PYTHONPATH": f"{source}/platform/python:{source}/library",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "ROCKE_BACKEND": "python",
        "ROCKE_LLVM_FLAVOR": config["llvm_flavor"],
        "ROCM_PATH": str(rocm),
        "ROCM_HOME": str(rocm),
        "ROCKE_HIP_LIB": str(rocm / "lib/libamdhip64.so"),
        "ROCKE_COMGR_LIB": str(rocm / "lib/libamd_comgr.so"),
        "LD_LIBRARY_PATH": f"{aot}/lib:{rocm}/lib",
    }
    verify = (
        "import hashlib,pathlib; "
        f"root=pathlib.Path({str(compute_root / 'deps')!r}); "
        f"expected={{'aotriton-runtime.tar.gz':{_RUNTIME_SHA!r},'aotriton-gfx115x.tar.gz':{_IMAGES_SHA!r}}}; "
        "actual={name:hashlib.file_digest((root/name).open('rb'),'sha256').hexdigest() for name in expected}; "
        "assert actual == expected, 'AOTriton release archive checksum changed'"
    )
    compile_command = [
        "hipcc",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(bridge_source),
        "-I" + str(aot / "include"),
        "-L" + str(aot / "lib"),
        "-laotriton_v2",
        "-Wl,-rpath," + str(aot / "lib"),
        "-o",
        str(bridge),
    ]
    run_command = [
        config["python"],
        "-u",
        "-m",
        "benchmarks.gfx1151.attention.benchmark_sdpa",
        "--aotriton-shim",
        str(bridge),
        "--source-hash",
        digest,
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "umask 077",
        "unset HSA_OVERRIDE_GFX_VERSION",
    ]
    lines.extend(
        f"export {key}={shlex.quote(value)}" for key, value in environment.items()
    )
    lines.append(shlex.join([config["python"], "-c", verify]))
    lines.append(shlex.join(["mkdir", "-p", str(build)]))
    lines.append(f"if [ ! -f {shlex.quote(str(bridge))} ]; then")
    lines.append("  " + shlex.join(compile_command))
    lines.append("fi")
    lines.append("exec " + shlex.join(run_command))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.site.read_text(encoding="utf-8"))
    source = Path(config["source_root"]).resolve()
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        config["ssh_host"],
    ]
    login_root = PurePosixPath(config["login_root"])
    compute_root = PurePosixPath(config["compute_root"])
    # A run ID separates private logs, not workloads, random seeds, or measurements.
    run_id = uuid.uuid4().hex
    login_run = login_root / "runs" / run_id
    compute_run = compute_root / "runs" / run_id
    with tempfile.TemporaryDirectory(prefix="rocke-sdpa-snapshot-") as temporary:
        archive = Path(temporary) / "source.tar"
        digest = _snapshot(source, archive)
        login_snapshot = login_root / "sources" / digest
        compute_snapshot = compute_root / "sources" / digest
        upload = (
            "umask 077; "
            + shlex.join(["mkdir", "-p", str(login_snapshot), str(login_run)])
            + " && "
            + shlex.join(["tar", "-xf", "-", "-C", str(login_snapshot)])
        )
        with archive.open("rb") as data:
            subprocess.run(ssh + [upload], stdin=data, check=True)
        job = _job_script(
            config, compute_snapshot / "rocke", compute_root / "build" / digest, digest
        )
        command = [
            "sbatch",
            "--parsable",
            "--wait",
            "--exclusive",
            "--nodes=1",
            "--ntasks=1",
            "--partition=" + config["partition"],
            "--nodelist=" + config["node"],
            "--gres=gpu:gfx1151:1",
            "--cpus-per-task=" + str(config["cpus"]),
            "--mem=" + config["memory"],
            "--time=" + config["job_time"],
            "--job-name=rocke-sdpa",
            "--chdir=" + str(compute_snapshot / "rocke"),
            "--output=" + str(compute_run / "output.log"),
            "--error=" + str(compute_run / "output.log"),
        ]
        print(f"ASI source_sha256={digest}", flush=True)
        process = subprocess.Popen(
            ssh + [shlex.join(command)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        job_id = None
        try:
            process.stdin.write(job.encode("utf-8"))
            process.stdin.close()
            for line in process.stdout:
                text = line.decode("utf-8", "replace").strip()
                if re.fullmatch(r"[0-9]+(?:;[^\s]+)?", text):
                    job_id = text.split(";", 1)[0]
                    print(f"ASI slurm_job_id={job_id}", flush=True)
                else:
                    print(text, flush=True)
            status = process.wait()
        except BaseException:
            if job_id:
                subprocess.run(ssh + [shlex.join(["scancel", job_id])], check=False)
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if job_id is None:
            raise RuntimeError(f"Slurm submission failed (exit {status})")
        # The private log is transported only after the Slurm-owned job completes.
        output_status = subprocess.run(
            ssh + [shlex.join(["cat", "--", str(login_run / "output.log")])],
            check=False,
        ).returncode
        return status or output_status


if __name__ == "__main__":
    raise SystemExit(main())
