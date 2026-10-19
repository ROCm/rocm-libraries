# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MLA decode candidate, built once per arch.

The decode kernel is one builder for both arches (see
:mod:`kernels.mla.mla_decode`); the arch module picks its spec class and
occupancy and registers what :func:`make_mla_decode_candidate` returns. The
candidate is selected by the request's ``max_seqlen_q`` alone: the caller
states the problem, and this candidate owns the decode regime and its
split-KV count.

Input contract: the *absorbed* query ``qa = [Q_nope @ W_UK_nopeᵀ | Q_rope]``
(``[total_q, H, r_kv + d_rope]``) in, the normalized latent (before
``W_UV``) and the LSE out. That is not the prefill kernel's contract, so no
other candidate stands in for decode: a one-query request that decode cannot
serve is refused.

A split-KV selection is two launches -- the latent kernel writing fp32
partials, then the combine. :func:`mla_decode_launch` describes both from the
``DispatchResult``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from kernels.mla.mla_decode import (
    MlaDecodeSpec,
    check_mla_decode_extents,
    mla_decode_block,
    mla_decode_combine_spec,
    mla_decode_grid,
    mla_decode_max_num_seg,
    mla_decode_signature,
    mla_decode_workspace_shapes,
    supports_mla_decode,
)
from kernels.mla.mla_decode_combine import (
    MlaDecodeCombineSpec,
    build_mla_decode_combine,
    mla_decode_combine_block,
    mla_decode_combine_grid,
    mla_decode_combine_signature,
    supports_mla_decode_combine,
)
from rocke.core.ir import KernelDef
from rocke.dispatch.core import (
    Capability,
    DispatchResult,
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
    mla_kernel_id,
    resolve_max_grid_dims,
    resolve_num_cus,
)

#: Ahead of prefill (10). The two never compete -- prefill refuses
#: ``max_seqlen_q == 1`` -- so this only orders the registry listing.
DECODE_PRIORITY = 5

#: The shipped decode layout: 16 heads per wave x 4 waves, ``BLOCK_H = 64``.
#: 64 heads (one Kimi-K2 layer) is one head group, 128 (DeepSeek-V3) two.
DECODE_NUM_WARPS = 4

#: Head counts decode admits: the ones its GPU evidence covers. Any multiple
#: of ``BLOCK_H`` lowers, but only these were run on a device. Head counts
#: below 64 -- tensor-parallel shards such as 16 heads per rank -- need a
#: smaller-``BLOCK_H`` layout first.
# TODO: validate 192, 256, ... on gfx942 and gfx950 (numeric lane + goldens)
# before admitting them.
DECODE_NUM_HEADS = (64, 128)

#: LLVM flavors whose backend keeps the decode kernel out of the AGPR file.
#: llvm20 ignores ``agpr_alloc``: the same IR then spills and runs one
#: workgroup per CU instead of the two the split-KV count is sized for.
DECODE_LLVM_FLAVORS = ("llvm22", "llvm23")

#: Floor on key tiles per split-KV segment: untuned, but bounded by traffic.
#: A 16-key tile is 16 x 576 bf16 = 18 KiB shared by 64 heads, about 288 B
#: per head, while a segment's partial is a 512-wide fp32 row plus
#: ``(m, l)``, about 2 KiB per head, written once and read once by the
#: combine. At 32 tiles a segment reads about 9 KiB of keys per head for
#: about 4 KiB of partial traffic; below about 14 tiles the partials would
#: cost more than the keys they summarize.
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
    An empty segment costs one ``(m, l)`` store and its read in the combine,
    not a partial row. A pure function of the request. A bad count costs
    speed, not correctness, once :func:`decode_spec_for` has clamped it to the
    i32 workspace bound.
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
    for this batch, clamped so the split workspace stays i32-addressable and
    the grid's z axis (one workgroup per segment) stays within the device's.
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
    num_seg = min(
        num_seg,
        mla_decode_max_num_seg(spec, total_q=req.total_q),
        resolve_max_grid_dims(req)[2],
    )
    return dataclasses.replace(spec, num_seg=max(1, num_seg))


def _check_grids(spec: MlaDecodeSpec, req: MLARequest) -> tuple[bool, str]:
    """Whether every launch of ``spec`` fits the device's per-axis grid limit:
    the latent kernel's ``(head groups, sequences, segments)`` and, split, the
    combine's ``(head groups, tokens, 1)``."""
    limits = resolve_max_grid_dims(req)
    grids = [("latent", mla_decode_grid(spec, num_seqs=req.num_seqs))]
    if spec.split_kv:
        grids.append(
            (
                "split-KV combine",
                mla_decode_combine_grid(
                    mla_decode_combine_spec(spec), total_q=req.total_q
                ),
            )
        )
    for kernel, grid in grids:
        for axis, extent, limit in zip("xyz", grid, limits, strict=True):
            if extent > limit:
                return False, (
                    f"the {kernel} grid's {axis} extent {extent} exceeds the "
                    f"device's {limit}"
                )
    return True, ""


def _decode_spec_gate(spec: MlaDecodeSpec, req: MLARequest) -> tuple[bool, str]:
    """The per-spec half of the candidate's support: i32 extents, grid limits,
    the kernel gate, and the combine gate when split. Dispatch's own choice and
    a forced split count (:meth:`MlaDecodeLaunch.with_num_seg`) pass the same
    gate."""
    ok, why = check_mla_decode_extents(spec, total_q=req.total_q)
    if not ok:
        return False, why
    ok, why = _check_grids(spec, req)
    if not ok:
        return False, why
    ok, why = supports_mla_decode(spec, arch=req.arch)
    if not ok:
        return False, why
    if spec.split_kv:
        ok, why = supports_mla_decode_combine(
            mla_decode_combine_spec(spec), arch=req.arch
        )
        if not ok:
            return False, f"split-KV combine: {why}"
    return True, "ok"


def make_mla_decode_candidate(
    *,
    spec_cls: type[MlaDecodeSpec],
    build_fn: Callable[..., KernelDef],
    wgs_per_cu: int,
    fallback_num_cus: int,
) -> KernelCandidate:
    """The weight-absorbed MLA decode kernel on ``spec_cls.ARCH``.

    Serves the absorbed contract (module docstring): ``qa`` in, normalized
    latent + LSE out; with ``spec.split_kv`` the launch is the latent kernel
    plus :mod:`kernels.mla.mla_decode_combine` (see :func:`mla_decode_launch`).
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
        from rocke.core.lower_llvm import _resolve_llvm_flavor
        from rocke.runtime.comgr import ComgrError

        try:
            flavor = _resolve_llvm_flavor()
        except ComgrError as exc:
            return False, (
                f"MLA decode needs one of {DECODE_LLVM_FLAVORS}; set "
                f"ROCKE_LLVM_FLAVOR to choose one offline ({exc})"
            )
        if flavor not in DECODE_LLVM_FLAVORS:
            return False, (
                f"MLA decode needs one of {DECODE_LLVM_FLAVORS} (llvm20 ignores "
                f"agpr_alloc, and the kernel spills), got {flavor}"
            )
        try:
            spec = spec_for(req)
        except ValueError as exc:
            return False, f"spec rejected: {exc}"
        return _decode_spec_gate(spec, req)

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
                ShapeRange("num_heads", allowed=DECODE_NUM_HEADS),
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


@dataclass(frozen=True)
class MlaDecodeLaunch:
    """Both stages of a dispatched decode selection.

    ``result`` is the latent kernel, exactly as dispatch returned it. With
    ``combine_spec`` set (split-KV), the latent kernel writes the fp32
    workspace of :meth:`workspace_shapes` and the combine merges it into
    ``out`` / ``lse``; its launch geometry and signature are the methods
    below. Without it the latent kernel writes ``out`` / ``lse`` itself.
    """

    result: DispatchResult

    def __post_init__(self) -> None:
        if not isinstance(self.result.spec, MlaDecodeSpec):
            raise ValueError(
                f"{self.result.candidate.name} is not an MLA decode selection; "
                "mla_decode_launch describes decode results only"
            )

    @property
    def spec(self) -> MlaDecodeSpec:
        """The latent kernel's spec, as dispatched."""
        return self.result.spec

    @property
    def combine_spec(self) -> MlaDecodeCombineSpec | None:
        """The split-KV combine, derived from :attr:`spec`; ``None`` single-pass."""
        return mla_decode_combine_spec(self.spec) if self.spec.split_kv else None

    def _combine(self) -> MlaDecodeCombineSpec:
        cs = self.combine_spec
        if cs is None:
            raise ValueError(f"{self.spec.kernel_name()} is single-pass: no combine")
        return cs

    def workspace_shapes(self) -> dict[str, tuple[int, ...]]:
        """fp32 workspace, keyed by argument name; empty without a split."""
        return mla_decode_workspace_shapes(
            self.spec, total_q=self.result.request.total_q
        )

    def combine_grid(self) -> tuple[int, int, int]:
        """Combine grid: one workgroup per ``num_warps`` heads per token."""
        return mla_decode_combine_grid(
            self._combine(), total_q=self.result.request.total_q
        )

    def combine_block(self) -> tuple[int, int, int]:
        """Combine workgroup shape."""
        return mla_decode_combine_block(self._combine())

    def combine_signature(self) -> tuple[dict, ...]:
        """Combine arguments: ``out``, ``lse``, the workspace, ``num_seg``."""
        return tuple(mla_decode_combine_signature(self._combine()))

    def build_combine(self) -> KernelDef:
        """The combine kernel for the dispatched arch (split-KV only)."""
        return build_mla_decode_combine(self._combine(), arch=self.result.request.arch)

    def kernel_names(self) -> tuple[str, ...]:
        """The latent kernel's name, then the combine's when split."""
        names = (self.spec.kernel_name(),)
        if self.combine_spec is not None:
            names += (self.combine_spec.kernel_name(),)
        return names

    def with_num_seg(self, num_seg: int) -> MlaDecodeLaunch:
        """This launch with the split count forced (parity sweeps), gated as
        dispatch gates its own choice; the result's identity follows the spec."""
        req, cand = self.result.request, self.result.candidate
        spec = dataclasses.replace(self.spec, num_seg=int(num_seg))
        ok, why = _decode_spec_gate(spec, req)
        if not ok:
            raise ValueError(f"num_seg={num_seg}: {why}")
        return MlaDecodeLaunch(
            dataclasses.replace(
                self.result,
                spec=spec,
                kernel_id=mla_kernel_id(req, cand, spec),
                grid=tuple(cand.grid(spec, req)),
                signature=tuple(cand.signature(spec)),
                explanation=self.result.explanation + (f"num_seg forced to {num_seg}",),
            )
        )


def mla_decode_launch(result: DispatchResult) -> MlaDecodeLaunch:
    """Describe the launches of a decode ``result`` from ``dispatch_mla``.

    Raises :class:`ValueError` when ``result`` selected another kernel family
    (a prefill result is one launch with a different contract).
    """
    return MlaDecodeLaunch(result)


__all__ = [
    "DECODE_LLVM_FLAVORS",
    "DECODE_MIN_TILES_PER_SEG",
    "DECODE_NUM_HEADS",
    "DECODE_NUM_WARPS",
    "DECODE_PRIORITY",
    "MlaDecodeLaunch",
    "choose_num_seg",
    "decode_spec_for",
    "make_mla_decode_candidate",
    "mla_decode_launch",
]
