# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, qualify, and verify a pinned live SDPA reference bundle.

The snapshot and qualification commands are offline maintenance operations.
Verification only executes a locked bundle and current kernels; it never
downloads dependencies, generates independent answers, or promotes a baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from reference_common.cli import Metric, ReferenceOperation, TorchReference
from reference_common.numeric import max_abs_upper

from . import architectures, contract


def _distance(left: np.ndarray, right: np.ndarray, entry: Mapping[str, Any]) -> float:
    return max_abs_upper(left, right)


def _prepare_torch(device: str, architecture: str) -> Callable[..., dict]:
    from .torch_reference import cross_check, validate_device

    validate_device(device, architecture)

    def check(case, inputs, reference, baseline, parameters):
        return cross_check(case, inputs, reference, baseline, device=device)

    return check


OPERATION = ReferenceOperation(
    name="sdpa",
    label="SDPA",
    description=__doc__,
    contract=contract,
    architectures=architectures,
    reference={
        "implementation": "numpy-float64-sdpa-v1",
        "metric": "max-absolute-error",
        "input_generator": "numpy-PCG64-seed-0-f32-normal; regenerated and digest-checked quantized bits",
    },
    metric=Metric(name="max_abs_upper", distance=_distance),
    torch=TorchReference(
        devices=("cpu", "cuda"),
        help="also qualify Torch float64/float32 answers on cpu (default) or ROCm cuda",
        prepare=_prepare_torch,
    ),
    declares_operation=False,
)
qualify = OPERATION.qualify
load_bundle = OPERATION.load_bundle
manifest_key = OPERATION.manifest_key
verify_case = OPERATION.verify_case
main = OPERATION.main
