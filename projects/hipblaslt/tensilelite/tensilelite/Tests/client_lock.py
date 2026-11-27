# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which lock file a pytest-xdist worker holds while its client process runs.

On real hardware two clients on one GPU must not run at once, so workers that
share a GPU share a lock. Under an emulator each client process has its own
emulated GPU, so each worker can hold its own lock and clients run in parallel.
"""

import os
from pathlib import Path

# Set by the FFM and rocjitsu emulator setups respectively; either means each
# client process gets its own emulated GPU.
EMULATOR_VARIABLES = ("HSA_MODEL_MEMFILE", "ROCJITSU_RUNTIME_DIR")


def client_lock_path(lock_dir, worker_id, worker_gpu_id, scope=None, environ=os.environ):
    """Return the lock file for this worker's client runs.

    scope is the --client-lock-scope option: "gpu" (one lock per GPU),
    "worker" (one lock per worker) or None (worker under an emulator, else gpu).
    "worker" is refused outside an emulator, where it would let clients race
    on one GPU whenever there are more workers than GPUs.
    """
    lock_dir = Path(lock_dir)
    emulated = any(environ.get(name) for name in EMULATOR_VARIABLES)
    if scope is None:
        scope = "worker" if emulated else "gpu"
    if scope == "worker" and not emulated:
        raise ValueError(
            "--client-lock-scope=worker lets clients race on one GPU; use it only "
            f"under an emulator (one of {', '.join(EMULATOR_VARIABLES)} set)"
        )
    if scope == "gpu" and worker_gpu_id is not None:
        return lock_dir / f"client_execution_gpu{worker_gpu_id}.lock"
    return lock_dir / f"client_execution_{worker_id}.lock"
