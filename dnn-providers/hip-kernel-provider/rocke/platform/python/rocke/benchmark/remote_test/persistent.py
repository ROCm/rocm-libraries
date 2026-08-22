# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Persistent node allocation: submit a long-running holder job, run tests
within that allocation using `srun --jobid`, then cancel when done.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path
from typing import Optional

from . import config, slurm, transport


def _state_path(arch: str) -> Path:
    return config.LOCAL_STAGE_ROOT / f"{arch}_session.json"


class PersistentAllocation:
    """Context manager for a persistent slurm allocation.

    The holder job (sbatch + sleep) lives independently of this process, so
    the allocation can be *detached* (state written to a local JSON file) and
    later *reattached* from a separate invocation. This lets us reserve a node
    once and run many experiments against it (rsync artifacts + srun --jobid)
    without re-reserving for each test.
    """

    def __init__(self, arch: str):
        self.arch = arch
        self.profile = config.ARCHES[arch]
        self.jobid: Optional[str] = None
        self.nodename: Optional[str] = None
        self._remote_pkg: Optional[str] = None

    # --- detach / reattach -------------------------------------------------
    def save_state(self) -> None:
        p = _state_path(self.arch)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "arch": self.arch,
                    "jobid": self.jobid,
                    "nodename": self.nodename,
                    "remote_pkg": self._remote_pkg,
                }
            )
        )
        print(f"[alloc:{self.arch}] saved session -> {p}")

    @classmethod
    def reattach(cls, arch: str) -> "PersistentAllocation":
        p = _state_path(arch)
        if not p.exists():
            raise RuntimeError(f"no saved session for {arch} ({p}); run `hold` first")
        data = json.loads(p.read_text())
        self = cls(arch)
        self.jobid = data.get("jobid")
        self.nodename = data.get("nodename")
        self._remote_pkg = data.get("remote_pkg")
        if not self.jobid:
            raise RuntimeError(f"saved session for {arch} has no jobid")
        # Confirm the holder job is still RUNNING on the recorded node.
        sq = transport.ssh_run(
            f"squeue -j {self.jobid} -h -o '%T %N'", capture=True, check=False
        )
        parts = sq.stdout.strip().split(None, 1)
        if sq.returncode != 0 or not parts or parts[0] != "RUNNING":
            raise RuntimeError(
                f"holder job {self.jobid} for {arch} is not RUNNING "
                f"(squeue: {sq.stdout.strip()!r}); run `hold` again"
            )
        if len(parts) == 2 and parts[1]:
            self.nodename = parts[1]
        print(f"[alloc:{self.arch}] reattached job {self.jobid} on {self.nodename}")
        return self

    def clear_state(self) -> None:
        p = _state_path(self.arch)
        if p.exists():
            p.unlink()

    def allocate(self) -> None:
        """Submit a long-running sbatch job to reserve the node."""
        if self.jobid:
            raise RuntimeError(f"allocation already active: job {self.jobid}")

        # Hold the node by sleeping well past any plausible --time limit, so the
        # slurm time limit (profile.slurm_time) is the single source of truth for
        # how long the allocation lives (sleep 7d >> 12h limit).
        sbatch_script = (
            "#!/bin/bash\n"
            f"#SBATCH --partition={self.profile.slurm_partition}\n"
            f"#SBATCH --constraint={self.profile.slurm_constraint}\n"
            f"#SBATCH --gres={self.profile.slurm_gres}\n"
            f"#SBATCH --time={self.profile.slurm_time}\n"
            f"#SBATCH --job-name=rocke-{self.arch}-hold\n"
            f"#SBATCH --output={config.REMOTE.stage_root}/_hold.out\n"
            "echo HOLDER_ON=$(hostname)\n"
            "sleep 604800\n"
        )
        remote_script = f"{config.REMOTE.stage_root}/_holder.sbatch"
        transport.ssh_run(["mkdir", "-p", config.REMOTE.stage_root])
        transport.ssh_run(
            f"cat > {shlex.quote(remote_script)} <<'EOFSBATCH'\n{sbatch_script}EOFSBATCH\n"
        )

        print(f"[alloc:{self.arch}] submitting holder job...")
        r = transport.ssh_run(f"sbatch {shlex.quote(remote_script)}", capture=True)
        # Example: "Submitted batch job 349600"
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        if not m:
            raise RuntimeError(f"sbatch failed:\n{r.stdout}")
        self.jobid = m.group(1)

        # Poll until RUNNING. Nodes are often all busy, so be patient and let
        # the job sit in the queue (PENDING) rather than giving up early.
        wait_s = int(self.profile.__dict__.get("queue_wait_s", 0)) or 1800
        deadline = time.time() + wait_s
        last_state = ""
        while time.time() < deadline:
            time.sleep(5)
            sq = transport.ssh_run(
                f"squeue -j {self.jobid} -h -o '%T %N'",
                capture=True,
                check=False,
            )
            if sq.returncode != 0:
                raise RuntimeError(f"job {self.jobid} vanished from squeue")
            parts = sq.stdout.strip().split(None, 1)
            if not parts:
                continue
            state = parts[0]
            if state == "RUNNING" and len(parts) == 2:
                self.nodename = parts[1]
                print(
                    f"[alloc:{self.arch}] job {self.jobid} RUNNING on {self.nodename}"
                )
                break
            if state != last_state:
                print(
                    f"[alloc:{self.arch}] job {self.jobid} state={state}, waiting "
                    f"(up to {wait_s}s for a node)..."
                )
                last_state = state
        else:
            self.release()
            raise TimeoutError(f"job {self.jobid} did not reach RUNNING in {wait_s}s")

        # Push rocke tree once
        self._remote_pkg = slurm.push_rocke_tree()

    def run_test(
        self, extra_args: list[str] | None = None, env: dict | None = None
    ) -> int:
        """Run one test on the allocated node. Returns exit code."""
        if not self.jobid or not self.nodename:
            raise RuntimeError("no active allocation; call allocate() first")

        # Re-push artifacts
        remote_art = slurm.push_artifacts(self.arch)
        spec_path = config.stage_dir(self.arch) / "run_spec.json"
        run_spec = json.loads(spec_path.read_text())

        shape = run_spec.get("shape", {})
        shape_str = f"{shape.get('m', 0)},{shape.get('n', 0)},{shape.get('k', 0)}"
        hsaco = f"{remote_art}/{run_spec['hsaco']}"
        manifest = f"{remote_art}/{run_spec['manifest']}"

        env_exports = ""
        for k, v in (env or {}).items():
            env_exports += f"export {k}={shlex.quote(str(v))}; "

        inner = (
            f"set -e; cd {shlex.quote(self._remote_pkg)}; "
            f"export PYTHONPATH={shlex.quote(self._remote_pkg)}:$PYTHONPATH; "
            f"{env_exports}"
            f"echo '[remote] host='$(hostname)' arch={self.arch}'; "
            f"python3 -m rocke.run_manifest {shlex.quote(hsaco)} "
            f"{shlex.quote(manifest)} --shape {shape_str} --verify"
        )
        if extra_args:
            inner += " " + " ".join(shlex.quote(a) for a in extra_args)

        cmd = [
            "srun",
            f"--jobid={self.jobid}",
            "--output",
            f"{remote_art}/srun.out",
            "--error",
            f"{remote_art}/srun.err",
            "bash",
            "-lc",
            inner,
        ]
        cmd_str = " ".join(shlex.quote(str(a)) for a in cmd)
        print(f"[run:{self.arch}] $ {cmd_str}")
        r = transport.ssh_run(cmd_str, check=False, stream=True)

        # Fetch logs
        tail = transport.ssh_run(
            f"tail -n +1 {shlex.quote(remote_art)}/srun.out "
            f"{shlex.quote(remote_art)}/srun.err 2>/dev/null || true",
            capture=True,
            check=False,
        )
        if tail.stdout:
            print(f"[run:{self.arch}] --- logs ---\n{tail.stdout}")
        return r.returncode

    def release(self) -> None:
        """Cancel the allocation."""
        if self.jobid:
            print(f"[alloc:{self.arch}] releasing job {self.jobid}")
            transport.ssh_run(f"scancel {self.jobid}", check=False)
            self.jobid = None
            self.nodename = None

    def __enter__(self) -> PersistentAllocation:
        self.allocate()
        return self

    def __exit__(self, *args) -> None:
        self.release()
