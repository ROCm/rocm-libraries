# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional offline CPU Torch oracles; never imported by ordinary verification."""

from __future__ import annotations

import numpy as np

from reference_common.numeric import array_digest, decode
from .contract import Case, normalized_distance


def cross_check(
    case: Case,
    inputs: dict[str, np.ndarray],
    reference: np.ndarray,
    baseline: np.ndarray,
    scale: float,
) -> dict:
    """Check NumPy and baseline against Torch float64 and float32 convolution."""
    import torch
    import torch.nn.functional as functional

    results = {}
    with torch.inference_mode():
        for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            a, b = [
                torch.from_numpy(decode(inputs[key], case.dtype))
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device="cpu", dtype=dtype)
                for key in ("a", "b")
            ]
            out = functional.conv2d(
                a,
                b,
                stride=(case.sH, case.sW),
                padding=(case.pH, case.pW),
                dilation=(case.dH, case.dW),
                groups=case.groups,
            )
            # Use Torch's conversion to independently check output quantization.
            storage = torch.float16 if case.dtype == "fp16" else torch.bfloat16
            answer = out.float().to(storage).double().permute(0, 2, 3, 1).numpy()
            if answer.shape != reference.shape or not np.isfinite(answer).all():
                raise ValueError(f"invalid Torch {name} reference for {case.id}")
            disagreement = normalized_distance(answer, reference, scale)
            if disagreement > case.margin:
                raise ValueError(
                    f"NumPy/Torch {name} disagreement exceeds margin for {case.id}"
                )
            results[name] = {
                "reference_digest": array_digest(answer),
                "numpy_distance_upper": disagreement,
                "baseline_error_bound": normalized_distance(baseline, answer, scale),
            }
    return {
        "implementation": "torch-cpu-conv2d-f64-f32-rounded-v1",
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "results": results,
    }
