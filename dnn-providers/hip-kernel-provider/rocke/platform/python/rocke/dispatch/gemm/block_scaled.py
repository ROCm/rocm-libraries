# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection and sweep surface for native gfx950 block-scaled RCR GEMM."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...core.dtypes import normalize_dtype
from ...instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    stable_json_hash,
)


@dataclass(frozen=True)
class BlockScaledGemmRequest(OperatorRequest):
    M: int
    N: int
    K: int
    arch: str = "gfx950"
    dtype_a: str = "fp8e4m3"
    dtype_b: str = "fp8e4m3"
    dtype: str = "bf16"
    block_k: int = 32
    layout: str = "RCR"
    op: str = "block_scaled_gemm"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self):
        d = asdict(self)
        for key in ("dtype", "dtype_a", "dtype_b"):
            d[key] = normalize_dtype(d[key])
        return d

    def dims(self):
        return {"M": self.M, "N": self.N, "K": self.K}


def _spec(req):
    return BlockScaledGemmSpec(
        "block_scaled_gemm",
        req.M,
        req.N,
        req.K,
        req.dtype_a,
        req.dtype_b,
        req.dtype,
        req.block_k,
    )


def _supports(req):
    if not isinstance(req, BlockScaledGemmRequest):
        return False, "expected BlockScaledGemmRequest"
    if req.op != "block_scaled_gemm" or req.layout != "RCR":
        return False, "expected block_scaled_gemm RCR"
    return is_valid_spec(_spec(req), req.arch)


def _select(req):
    ok, why = _supports(req)
    if not ok:
        raise ValueError(why)
    return _spec(req)


def _binding(result, verify):
    # Host fixture imports are lazy; ordinary selection needs no NumPy runtime.
    import struct

    import numpy as np

    from ...examples.common.scaled_gemm_reference import reference_result
    from ...examples.gfx950.gemm.block_scaled_gemm_verify import (
        _output_dtype,
        _u8_buffer,
        make_inputs,
    )
    from ..core import ProblemBinding

    spec = result.spec
    inputs = make_inputs(spec, "mixed")
    output = np.full((spec.M, spec.N), np.nan, dtype=_output_dtype(spec.dtype_c))
    expected = (
        reference_result(
            *inputs,
            32,
            native=True,
            dtype_a=spec.dtype_a,
            dtype_b=spec.dtype_b,
            dtype_c=spec.dtype_c,
        )
        if verify
        else None
    )

    def make_args(rt):
        ptrs = []
        try:
            for array in (*inputs, output):
                ptr = rt.alloc(array.nbytes)
                ptrs.append(ptr)
                rt.memcpy_h2d(ptr, _u8_buffer(array), array.nbytes)
        except Exception:
            for ptr in ptrs:
                rt.free(ptr)
            raise
        return struct.pack("<QQQQQiii", *ptrs, spec.M, spec.N, spec.K), tuple(ptrs)

    def check(rt, ptrs):
        if not verify:
            return 0.0, 0, output.size
        import ctypes

        host = (ctypes.c_uint8 * output.nbytes)()
        rt.memcpy_d2h(host, ptrs[-1], output.nbytes)
        got = (
            np.frombuffer(bytes(host), dtype=output.dtype)
            .reshape(output.shape)
            .astype(np.float32)
        )
        bad = ~np.isfinite(got) | (got != expected)
        error = np.abs(got.astype(np.float64) - expected.astype(np.float64))
        return float(np.max(error)), int(np.count_nonzero(bad)), output.size

    return ProblemBinding(
        result.grid,
        result.block,
        make_args,
        check,
        2.0 * spec.M * spec.N * spec.K,
        float(sum(x.nbytes for x in (*inputs, output))),
    )


BLOCK_SCALED_GEMM_REGISTRY = CandidateRegistry(
    "block_scaled_gemm",
    dim_vocabulary=("M", "N", "K"),
    require_build=True,
    require_binding=True,
)
BLOCK_SCALED_GEMM_REGISTRY.register(
    KernelCandidate(
        name="gfx950_mfma_scale",
        family="block_scaled_gemm",
        algorithm="mfma_scale",
        spec_id="32x32x64",
        abi_version="block-scaled-rcr/v1",
        priority=0,
        _supports=_supports,
        select_spec=_select,
        signature=block_scaled_gemm_signature,
        grid=lambda spec, req: block_scaled_gemm_grid(spec),
        block=lambda spec: (64, 1, 1),
        sweep_space=lambda req: (_select(req),) if _supports(req)[0] else (),
        capability=Capability(
            arches=("gfx950",), dtypes=("bf16", "fp16"), layouts=("RCR",)
        ),
        build=lambda spec, arch: build_block_scaled_gemm(spec, arch=arch),
        bind=_binding,
    )
)


def _kernel_id(req, candidate, spec):
    return KernelId(
        op=req.op,
        family=candidate.family,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(asdict(spec), n=16),
    )


def registered_block_scaled_gemm_combos(req):
    return BLOCK_SCALED_GEMM_REGISTRY.combos(req)


def block_scaled_gemm_sweep_space(req):
    return BLOCK_SCALED_GEMM_REGISTRY.sweep_space(req)


def dispatch_block_scaled_gemm_all(req):
    return BLOCK_SCALED_GEMM_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_block_scaled_gemm(req):
    candidate = BLOCK_SCALED_GEMM_REGISTRY.select(req)
    spec = candidate.select_spec(req)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=_kernel_id(req, candidate, spec),
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=("gfx950 native K32-scaled MFMA",),
    )
