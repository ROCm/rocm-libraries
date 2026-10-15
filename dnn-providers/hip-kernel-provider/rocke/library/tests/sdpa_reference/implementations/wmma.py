# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared strided FP16 WMMA reference launch and frozen replay ABI.

The gfx1151/gfx1201 adapters exercise the standalone verifier's kernel.
This does not establish automatic unified-attention dispatch coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..contract import Case
from ..architectures.common import validate_library_root

ABI = "wmma-strided-v1"
CASES = tuple(
    Case("fp16", d, 4, kv, False, causal, batch=2, sequence_length=64)
    for d in (64, 128)
    for kv in (4, 1)
    for causal in (False, True)
)


@dataclass
class Prepared:
    """Selected production spec plus the exact runtime geometry to export."""

    architecture: str
    case: Case
    spec: Any
    launcher: Any = None


def prepare(case: Case, library_root: str, architecture: str) -> Prepared:
    """Select the standalone WMMA path without importing another source tree."""
    validate_library_root(library_root)
    if case not in CASES:
        raise ValueError("case is outside the qualified WMMA cohort")
    if architecture in ("gfx1151", "gfx1201"):
        from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

        spec = WmmaFmhaFwdSpec(
            head_size=case.head_dim,
            num_query_heads=case.query_heads,
            num_kv_heads=case.kv_heads,
            dtype=case.dtype,
            mask_mode="causal" if case.causal else "none",
        )
    else:
        raise ValueError(f"unsupported WMMA reference architecture: {architecture}")
    return Prepared(architecture, case, spec)


def signature() -> tuple[dict, ...]:
    """Freeze the ABI declared by the production WMMA builder."""
    return tuple(
        {"name": name, "type": kind, "size_bytes": size}
        for name, kind, size in (
            *((name, "ptr<f16, global>", 8) for name in ("Q", "K", "V", "O")),
            ("scale_log2", "f32", 4),
            *(
                (name, "i32", 4)
                for name in (
                    "seqlen_q",
                    "seqlen_k",
                    "stride_q_token",
                    "stride_q_head",
                    "stride_k_token",
                    "stride_k_head",
                    "stride_v_token",
                    "stride_v_head",
                    "stride_o_token",
                    "stride_o_head",
                )
            ),
        )
    )


def scalars(case: Case, scale: float) -> dict:
    """Convert the SDPA scale to the WMMA exp2 ABI; strides are in elements."""
    values = {
        "scale_log2": scale * math.log2(math.e),
        "seqlen_q": case.sequence_length,
        "seqlen_k": case.sequence_length,
    }
    for name, heads in (
        ("q", case.query_heads),
        ("k", case.kv_heads),
        ("v", case.kv_heads),
        ("o", case.query_heads),
    ):
        values[f"stride_{name}_token"] = heads * case.head_dim
        values[f"stride_{name}_head"] = case.head_dim
    return values


def replay_values(metadata: dict, buffers: dict) -> dict:
    """Bind live pointers to authenticated frozen scalar arguments."""
    return dict(
        metadata["scalars"],
        Q=buffers["q"],
        K=buffers["k"],
        V=buffers["v"],
        O=buffers["out"],
    )


def build(prepared: Prepared):
    """Build through the selected architecture's existing production builder."""
    from kernels.gfx1151.wmma_fmha_fwd import build_wmma_fmha_fwd

    return build_wmma_fmha_fwd(prepared.spec, arch=prepared.architecture)


def grid(prepared: Prepared) -> tuple[int, int, int]:
    """Use the production grid helper, including the nontrivial batch axis."""
    from kernels.gfx1151.wmma_fmha_fwd import wmma_fmha_fwd_grid as helper

    return helper(
        prepared.spec, seqlen_q=prepared.case.sequence_length, batch=prepared.case.batch
    )


def launch(prepared: Prepared, buffers: dict, scale: float) -> None:
    """Compile the selected kernel and launch through the counted HIP launcher."""
    from rocke.helpers import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    if prepared.launcher is None:
        artifact = compile_kernel(
            build(prepared),
            arch=prepared.architecture,
            backend="python",
            capture_ir_text=False,
        )
        prepared.launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=signature(),
        )
    prepared.launcher(
        replay_values({"scalars": scalars(prepared.case, scale)}, buffers),
        config=LaunchConfig(
            grid=grid(prepared), block=(prepared.spec.block_size, 1, 1)
        ),
    )


def exported_kernel(prepared: Prepared) -> tuple[bytes, dict]:
    """Export the executed code object, stride arguments, and launch geometry."""
    launcher = prepared.launcher
    if launcher is None:
        raise RuntimeError("cannot export an unlaunched WMMA reference")
    return launcher._hsaco, {
        "abi": ABI,
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": grid(prepared),
        "block": (prepared.spec.block_size, 1, 1),
        "runtime_shape": True,
        "scalars": scalars(prepared.case, prepared.case.scale),
    }
