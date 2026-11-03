# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional offline Torch SDPA checks; ordinary replay never imports Torch."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from reference_common.numeric import (
    array_digest,
    decode,
    file_digest,
    max_abs_upper,
    write_json,
)
from reference_common.targets import processor_from_target_id
from .contract import Case, independent_reference, make_inputs


def validate_device(device: str, architecture: str | None = None) -> str | None:
    """Fail explicitly if the requested Torch device cannot qualify this target."""
    import torch

    if device == "cpu":
        return None
    if device != "cuda":
        raise ValueError("Torch reference device must be cpu or cuda")
    if not torch.version.hip or not torch.cuda.is_available():
        raise ValueError("Torch cuda reference requires a ROCm GPU")
    processor = processor_from_target_id(
        torch.cuda.get_device_properties(0).gcnArchName
    )
    if architecture is not None and processor != architecture:
        raise ValueError(
            f"Torch reference GPU {processor} does not match {architecture}"
        )
    return processor


def cross_check(
    case: Case,
    inputs: dict[str, np.ndarray],
    reference: np.ndarray,
    baseline: np.ndarray | None = None,
    *,
    device: str = "cpu",
) -> dict:
    """Bound reference disagreement on the exact quantized qualification inputs.

    Qualification takes the maximum baseline error across NumPy and the
    enabled Torch answers. The oracle gap is diagnostic; no fixed reserve
    is needed to establish ||N-R_j|| <= D+B < T for each recorded reference.
    """
    import torch
    import torch.nn.functional as functional

    processor = validate_device(device)
    scale = 1.0 / math.sqrt(case.head_dim)
    results = {}
    with torch.inference_mode():
        for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            q, k, v = (
                torch.from_numpy(decode(inputs[key], case.dtype))
                .permute(0, 2, 1, 3)
                .to(device="cuda:0" if device == "cuda" else "cpu", dtype=dtype)
                for key in ("q", "k", "v")
            )
            # Match the original tests' explicit GQA expansion and scale.
            repetitions = case.query_heads // case.kv_heads
            k = k.repeat_interleave(repetitions, dim=1)
            v = v.repeat_interleave(repetitions, dim=1)
            output = functional.scaled_dot_product_attention(
                q, k, v, is_causal=case.causal, scale=scale
            )
            answer = output.permute(0, 2, 1, 3).double().cpu().numpy()
            if answer.shape != reference.shape or not np.isfinite(answer).all():
                raise ValueError(f"invalid Torch {name} reference for {case.id}")
            disagreement = max_abs_upper(answer, reference)
            result = {
                "reference_digest": array_digest(answer),
                "numpy_distance_upper": disagreement,
            }
            if baseline is not None:
                result["baseline_error_bound"] = max_abs_upper(baseline, answer)
            results[name] = result
    return {
        "implementation": "torch-sdpa-f64-f32-default-backend-v1",
        "checker_sha256": file_digest(Path(__file__)),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device": device,
        "device_index": 0 if device == "cuda" else None,
        "processor": processor,
        "backend_policy": "torch-default-selection",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_backend_settings": (
            {
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            }
            if device == "cuda"
            else None
        ),
        "scale": scale,
        "numpy_abi_scale": case.scale,
        "transfer_contract": "max-reference-bound-plus-distance-lt-tolerance-v1",
        "results": results,
    }


def main() -> None:
    """Audit reference agreement without a baseline GPU run or bundle promotion."""
    from .architectures import ARCHITECTURES, get_architecture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHITECTURES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_device(args.device, args.arch)
    results = {}
    for case in get_architecture(args.arch).CASES:
        inputs = make_inputs(case)
        results[case.id] = {
            "input_digests": {
                key: array_digest(value) for key, value in inputs.items()
            },
            "torch_reference": cross_check(
                case, inputs, independent_reference(case, inputs), device=args.device
            ),
        }
    write_json(
        args.output,
        {
            "scope": "reference-agreement-only",
            "architecture": args.arch,
            "cases": results,
        },
    )


if __name__ == "__main__":
    main()
