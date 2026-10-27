# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Isolated HIP worker, frozen with the old runtime in a qualified bundle.

Source mode exercises the same dispatch and public launch entry as the existing
SDPA test. Replay mode loads the already compiled old HSACO and frozen ABI.
Neither mode computes independent reference answers or adjusts error budgets.
"""

from __future__ import annotations

from pathlib import Path

from reference_common.worker import WorkerOperation, main, run_case

from .architectures import get_architecture
from .contract import Case


def replay_values(metadata: dict, buffers: dict, case: Case) -> dict:
    """Bind frozen WMMA or legacy dense arguments without initializing HIP."""
    abi = metadata.get("abi", "dense-v1")
    if abi in ("wmma-strided-v1", "wmma-strided-v2", "wmma-strided-v3"):
        from .implementations.wmma import replay_values as wmma_replay_values

        values = wmma_replay_values(metadata, buffers)
    elif abi in ("dense-v1", "dense-v2"):
        values = {
            "q_ptr": buffers["q"],
            "k_ptr": buffers["k"],
            "v_ptr": buffers["v"],
            "o_ptr": buffers["out"],
            "scale": case.scale,
        }
        if abi == "dense-v2":
            values.update(metadata["scalars"])
        elif metadata["runtime_shape"]:
            values.update(
                batch=case.batch,
                seqlen_q=case.sequence_length,
                seqlen_kv=case.sequence_length,
            )
    else:
        raise ValueError(f"unsupported SDPA replay ABI: {abi}")
    return values


_OPERATION = WorkerOperation(
    label="SDPA",
    case=Case,
    launch=lambda target, spec, buffers, case: target.launch(spec, buffers, case.scale),
    replay_values=replay_values,
)


def run(request: dict, output: Path) -> None:
    """Execute a single declared case, poisoning every output before each run."""
    run_case(_OPERATION, get_architecture(request["architecture"]), request, output)


if __name__ == "__main__":
    main(run)
