# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build, qualify, and verify a pinned live convolution reference bundle.

The snapshot and qualification commands are offline maintenance operations.
Verification only executes a locked bundle and current kernels; it never
downloads dependencies, generates independent answers, or promotes a baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from reference_common.cli import Metric, ReferenceOperation, TorchReference

from . import architectures, contract
from .contract import normalized_distance, reference_scale


def _frozen_scale(reference: np.ndarray) -> dict[str, Any]:
    return {"reference_scale": reference_scale(reference)}


def _distance(left: np.ndarray, right: np.ndarray, entry: Mapping[str, Any]) -> float:
    return normalized_distance(left, right, entry["reference_scale"])


def _prepare_torch(device: str, architecture: str) -> Callable[..., dict]:
    import torch  # noqa: F401 - fail before starting qualification if unavailable.

    from .torch_reference import cross_check

    def check(case, inputs, reference, baseline, parameters):
        return cross_check(
            case, inputs, reference, baseline, parameters["reference_scale"]
        )

    return check


OPERATION = ReferenceOperation(
    name="conv",
    label="convolution",
    description=__doc__,
    contract=contract,
    architectures=architectures,
    reference={
        "implementation": "numpy-float64-conv-fwd-rounded-v1",
        "metric": "max-absolute-error/frozen-reference-scale",
        "input_generator": "numpy-PCG64-seed-0-uniform-f32; regenerated and digest-checked quantized bits",
    },
    metric=Metric(
        name="normalized_error_upper", distance=_distance, parameters=_frozen_scale
    ),
    torch=TorchReference(
        devices=("cpu",),
        help="also qualify against CPU Torch float64/float32 conv2d (requires Torch)",
        prepare=_prepare_torch,
    ),
)
qualify = OPERATION.qualify
load_bundle = OPERATION.load_bundle
manifest_key = OPERATION.manifest_key
verify_case = OPERATION.verify_case
main = OPERATION.main
