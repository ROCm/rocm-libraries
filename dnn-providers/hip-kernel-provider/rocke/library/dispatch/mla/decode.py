# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MLA decode candidate, built once per arch.

The decode kernel is one builder for both arches (see
:mod:`kernels.mla.mla_decode`); the arch module picks its spec class and
occupancy and registers what :func:`make_mla_decode_candidate` returns. The
candidate is selected by the request's ``max_seqlen_q`` alone: the caller
states the problem, and this candidate owns the decode regime and its
split-KV count.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from kernels.mla.mla_decode import (
    MlaDecodeSpec,
    mla_decode_block,
    mla_decode_grid,
    mla_decode_signature,
    supports_mla_decode,
)
from rocke.core.ir import KernelDef
from rocke.dispatch.core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    FAMILY,
    MLA_ABI_VERSION,
    MLA_FEATURES,
    MLARequest,
    _request_errors,
    _selector_matches,
    resolve_num_cus,
)

#: Outranks every prefill candidate (10), which also admits ``S_q = 1``.
DECODE_PRIORITY = 5

#: The shipped decode layout: 16 heads per wave x 4 waves, ``BLOCK_H = 64``.
#: 64 heads (one Kimi-K2 layer) is one head group, 128 (DeepSeek-V3) two.
DECODE_NUM_WARPS = 4

# The decode kernels index qa, out and the split workspace with i32 element
# offsets.
_I32_MAX = 2**31 - 1


#: Floor on key tiles per split-KV segment. A segment's partial is a 512-wide
#: fp32 latent per head, about as large as 32 tiles of the bf16 keys it
#: summarizes are per head; much shorter segments would move more workspace
#: than KV.
DECODE_MIN_TILES_PER_SEG = 32


def choose_num_seg(
    *,
    num_seqs: int,
    head_groups: int,
    max_seqlen_k: int,
    block_k: int,
    num_cus: int,
    wgs_per_cu: int,
) -> int:
    """Split-KV segment count for a decode launch.

    Enough segments that ``num_seqs * head_groups * num_seg`` workgroups fill
    ``wgs_per_cu`` workgroups on every CU, but never so many that a segment of
    the longest sequence drops under :data:`DECODE_MIN_TILES_PER_SEG` tiles.
    At least 1. Shorter sequences in the same batch split their own tile range
    the same number of ways, so their segments are shorter still -- or empty.
    A pure function of the request. A bad count costs speed, not correctness,
    once :func:`decode_spec_for` has clamped it to the i32 workspace bound.
    """
    base = max(1, num_seqs * head_groups)
    want = -(-(max(1, num_cus) * wgs_per_cu) // base)
    n_tiles = -(-max(0, max_seqlen_k) // block_k)
    cap = max(1, n_tiles // DECODE_MIN_TILES_PER_SEG)
    return max(1, min(want, cap))


def decode_spec_for(
    req: MLARequest,
    *,
    spec_cls: type[MlaDecodeSpec],
    wgs_per_cu: int,
    fallback_num_cus: int,
) -> MlaDecodeSpec:
    """The decode spec dispatch ships for ``req``, with its split-KV count.

    The bring-up geometry, and the segment count :func:`choose_num_seg` picks
    for this batch, clamped so the split workspace stays i32-addressable.
    ``fallback_num_cus`` stands in when neither the request nor the live
    device gives a CU count; too high a value only splits further than needed.
    """
    spec = spec_cls(
        num_heads=req.num_heads,
        d_rope=req.d_rope,
        r_kv=req.kv_lora_rank,
        block_k=req.page_block_size,
        page_block_size=req.page_block_size,
        num_warps=DECODE_NUM_WARPS,
        dtype=req.dtype.lower(),
    )
    num_seg = choose_num_seg(
        num_seqs=req.num_seqs,
        head_groups=max(1, req.num_heads // spec.block_h),
        max_seqlen_k=req.max_seqlen_k,
        block_k=spec.block_k,
        num_cus=resolve_num_cus(req) or fallback_num_cus,
        wgs_per_cu=wgs_per_cu,
    )
    row_elems = int(req.total_q) * spec.num_heads * spec.r_kv
    num_seg = max(1, min(num_seg, _I32_MAX // max(1, row_elems)))
    return dataclasses.replace(spec, num_seg=num_seg)


def make_mla_decode_candidate(
    *,
    spec_cls: type[MlaDecodeSpec],
    build_fn: Callable[..., KernelDef],
    wgs_per_cu: int,
    fallback_num_cus: int,
) -> KernelCandidate:
    """The weight-absorbed MLA decode kernel on ``spec_cls.ARCH``.

    ``Qa`` in, normalized latent + LSE out; with ``spec.split_kv`` the launch
    is the latent kernel plus :mod:`kernels.mla.mla_decode_combine`.
    ``build_fn`` is the arch's builder, the one whose ``spec`` annotation is
    ``spec_cls``; ``wgs_per_cu`` is the kernel's occupancy on that arch, which
    the split-KV count aims to fill.
    """
    arch = spec_cls.ARCH
    name = f"mla_decode_{arch}"
    spec_id = f"{arch}_mla_decode"

    def spec_for(req: MLARequest) -> MlaDecodeSpec:
        return decode_spec_for(
            req,
            spec_cls=spec_cls,
            wgs_per_cu=wgs_per_cu,
            fallback_num_cus=fallback_num_cus,
        )

    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MLARequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        try:
            spec = spec_for(req)
        except ValueError as exc:
            return False, f"spec rejected: {exc}"
        widest = max(spec.qa_cols, spec.r_kv * spec.num_seg)
        if int(req.total_q) * spec.num_heads * widest > _I32_MAX:
            return False, (
                f"total_q={req.total_q} x num_heads={spec.num_heads} x {widest} "
                "elements overflows the decode kernel's i32 offsets"
            )
        ok, why = supports_mla_decode(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> MlaDecodeSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MLARequest)
        return spec_for(req)

    def grid(spec: MlaDecodeSpec, req: OperatorRequest) -> tuple[int, int, int]:
        assert isinstance(req, MLARequest)
        return mla_decode_grid(spec, num_seqs=req.num_seqs)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="mla_decode_absorb",
        spec_id=spec_id,
        abi_version=MLA_ABI_VERSION,
        priority=DECODE_PRIORITY,
        capability=Capability(
            arches=(arch,),
            dtypes=("bf16",),
            shapes=(
                ShapeRange("max_seqlen_q", max=1),
                ShapeRange("num_heads", multiple_of=16 * DECODE_NUM_WARPS),
                ShapeRange("d_nope", allowed=(128,)),
                ShapeRange("d_rope", allowed=(64,)),
                ShapeRange("d_v", allowed=(128,)),
                ShapeRange("kv_lora_rank", allowed=(512,)),
                ShapeRange("page_block_size", allowed=(16,)),
                ShapeRange("num_seqs", min=1),
            ),
            # At S_q = 1 the bottom-right causal mask admits every key, so a
            # causal and a non-causal request are the same problem: the
            # feature is served, not required.
            supports_features=MLA_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: tuple(mla_decode_signature(spec)),
        grid=grid,
        block=mla_decode_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: build_fn(spec, arch=arch),
    )
    return candidate


__all__ = [
    "DECODE_MIN_TILES_PER_SEG",
    "DECODE_NUM_WARPS",
    "DECODE_PRIORITY",
    "choose_num_seg",
    "decode_spec_for",
    "make_mla_decode_candidate",
]
