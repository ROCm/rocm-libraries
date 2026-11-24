# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which processes hold the GPU: the one contention check for these tools.

Another tenant's kernels share the device and skew every timing, and the only
reliable signal for that is the KFD process list -- ``gpu_busy_percent`` moves
by about one point under a saturating GEMM, so it cannot tell an idle GPU from
a loaded one. ``fp8_gemm_sweep`` records the list in its machine-state
snapshots and ``fp8_trait_ab`` prints it at the start and end of a run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def kfd_processes() -> tuple[list[dict[str, Any]], str]:
    """Processes currently holding the GPU, and where the answer came from.

    ``rocm-smi --showpids`` first (it also sees root-owned processes);
    ``/sys/class/kfd/kfd/proc`` when rocm-smi is absent or fails. Returns
    ``([], "unavailable")`` rather than raising: a missing GPU must not take
    down a host-only run.
    """
    smi = shutil.which("rocm-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--showpids"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout
            procs: list[dict[str, Any]] = []
            for line in out.splitlines():
                parts = line.split()
                # Rows are "<pid> <name> <gpus> [<vram> <sdma> <cu>]"; the
                # header and the banner lines do not start with a bare integer.
                if len(parts) >= 2 and parts[0].isdigit():
                    vram = parts[3] if len(parts) >= 4 else ""
                    procs.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "vram_bytes": int(vram) if vram.isdigit() else None,
                        }
                    )
            return procs, "rocm-smi"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    proc_dir = Path("/sys/class/kfd/kfd/proc")
    if proc_dir.is_dir():
        return (
            [
                {"pid": int(p.name), "name": None, "vram_bytes": None}
                for p in proc_dir.iterdir()
                if p.name.isdigit()
            ],
            "sysfs",
        )
    return [], "unavailable"


def is_ours(pid: int) -> bool:
    """Whether ``pid`` belongs to this run (us, or a worker we forked)."""
    if pid == os.getpid():
        return True
    try:
        # field 6 of /proc/<pid>/stat is the session id; our forked workers
        # share ours. comm can contain spaces, so split on the closing paren.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[3]) == os.getsid(0)
    except (OSError, IndexError, ValueError):
        return False


def foreign_kfd_processes() -> list[dict[str, Any]] | None:
    """The GPU processes that are not this run's, or None when unknown."""
    procs, source = kfd_processes()
    if source == "unavailable":
        return None
    return [p for p in procs if not is_ours(p["pid"])]
