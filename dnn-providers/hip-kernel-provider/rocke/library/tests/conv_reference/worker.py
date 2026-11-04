# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Isolated HIP worker, frozen with the old runtime in a qualified bundle.

Source mode exercises the same dispatch and public launch entry as the existing
convolution test. Replay mode loads the already compiled old HSACO and frozen ABI.
Neither mode computes independent reference answers or adjusts error budgets.
"""

from __future__ import annotations

from pathlib import Path

from reference_common.worker import WorkerOperation, main, run_case

from .architectures import get_architecture
from .contract import Case


def replay_values(metadata: dict, buffers: dict, case: Case) -> dict:
    """Bind live operand buffers to the frozen scalar ABI arguments."""
    values = dict(metadata["scalars"])
    values.update(
        {name: buffers[operand] for name, operand in metadata["bindings"].items()}
    )
    return values


_OPERATION = WorkerOperation(
    label="convolution",
    case=Case,
    launch=lambda target, prepared, buffers, case: target.launch(prepared, buffers),
    replay_values=replay_values,
)


def run(request: dict, output: Path) -> None:
    """Execute a single declared case, poisoning every output before each run."""
    run_case(_OPERATION, get_architecture(request["architecture"]), request, output)


if __name__ == "__main__":
    main(run)
