# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipTensor contractions (and plain permutes) as task-queue monokernel ops.

A hipTensor contraction ``D[m.., n..] = sum_k A[m.., k..] B[n.., k..]`` with
generalized packed column-major tensors (first mode fastest) becomes, for the
rocKE RCR GEMM bodies (K contiguous):

* ``pA``: permute A to ``Ap[M, K]`` (K fastest; mode list K modes + M modes),
* ``pB``: permute B to ``Bp[N, K]`` (K fastest),
* ``gemm``: ``E[n, m] = sum_k Bp[n, k] Ap[m, k]`` (rocKE M_r = N, N_r = M),
  i.e. ``E`` holds D's M modes fastest, then its N modes,
* ``pD``: permute ``E`` to D's mode order (skipped when D already is
  ``[M modes..., N modes...]`` in that order, e.g. the canonical rows).

M modes are taken in D's order, N modes in D's order, K modes in A's order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.dispatch.gemm.common import GemmRequest
from rocke.dispatch.gemm.fp16_rcr import dispatch_gemm_fp16, dispatch_gemm_fp16_all
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)
from rocke.instances.common.permute_nd import PermuteSpec, build_permute, permute_grid

from .task_queue import QueueOp

__all__ = ["Contraction", "contraction_ops", "gemm_spec_for", "permute_op"]


@dataclass(frozen=True)
class Contraction:
    name: str
    modes_a: Tuple[str, ...]
    modes_b: Tuple[str, ...]
    modes_d: Tuple[str, ...]
    extent: Dict[str, int]

    @property
    def m_modes(self) -> Tuple[str, ...]:
        return tuple(x for x in self.modes_d if x in self.modes_a)

    @property
    def n_modes(self) -> Tuple[str, ...]:
        return tuple(x for x in self.modes_d if x in self.modes_b)

    @property
    def k_modes(self) -> Tuple[str, ...]:
        return tuple(
            x for x in self.modes_a if x in self.modes_b and x not in self.modes_d
        )

    def size(self, modes) -> int:
        out = 1
        for x in modes:
            out *= self.extent[x]
        return out

    @property
    def MNK(self) -> Tuple[int, int, int]:
        return self.size(self.m_modes), self.size(self.n_modes), self.size(self.k_modes)


def _np_shape(modes: Sequence[str], ext: Dict[str, int]) -> Tuple[int, ...]:
    """numpy (row-major) shape of a col-major hipTensor with ``modes``."""
    return tuple(ext[x] for x in reversed(modes))


def _perm(src_modes: Sequence[str], dst_modes: Sequence[str]) -> Tuple[int, ...]:
    """permute_nd ``perm`` taking numpy axes of src (reversed modes) to dst."""
    sx = list(reversed(src_modes))
    return tuple(sx.index(x) for x in reversed(dst_modes))


def _build_permute_any_rank(spec: PermuteSpec, arch: str):
    """``build_permute`` above its rank cap (interleaved rank-5 / rank-6
    hipTensor operands are rank 10 / 12 after coalescing). The cap is the
    host-args array size mirrored by the C++ instance builder's fixed arrays;
    the Python body is a generic transform-DAG offset chain with no rank
    limit. The resulting IR uses mirrored ops only; only the instance
    *builder* is Python-side, like the whole task-queue monokernel."""
    from rocke.instances.common import permute_nd as _pn

    if spec.rank <= _pn.MAX_RANK:
        return build_permute(spec, arch)
    old = _pn.MAX_RANK
    _pn.MAX_RANK = max(old, spec.rank)
    try:
        return build_permute(spec, arch)
    finally:
        _pn.MAX_RANK = old


def _with_rank(spec: PermuteSpec, fn):
    from rocke.instances.common import permute_nd as _pn

    old = _pn.MAX_RANK
    _pn.MAX_RANK = max(old, spec.rank)
    try:
        return fn()
    finally:
        _pn.MAX_RANK = old


def coalesce_modes(src: Sequence[str], dst: Sequence[str], ext: Dict[str, int]):
    """Merge runs of modes that are adjacent, in the same order, in both
    ``src`` and ``dst`` (col-major, first fastest). Returns new (src, dst, ext)."""
    groups: List[List[str]] = []
    for x in src:
        if groups:
            prev = groups[-1][-1]
            i = list(dst).index(prev)
            if i + 1 < len(dst) and dst[i + 1] == x:
                groups[-1].append(x)
                continue
        groups.append([x])
    name = {g[0]: "+".join(g) for g in groups}
    gext = {}
    for g in groups:
        e = 1
        for x in g:
            e *= ext[x]
        gext[name[g[0]]] = e
    nsrc = tuple(name[g[0]] for g in groups)
    lead = {x: g[0] for g in groups for x in g}
    ndst = tuple(name[x] for x in dst if lead[x] == x)
    return nsrc, ndst, gext


def permute_op(
    name: str,
    src: str,
    dst: str,
    src_modes: Sequence[str],
    dst_modes: Sequence[str],
    ext: Dict[str, int],
    *,
    block_size: int,
    elems_per_ticket: int,
    deps: Sequence[str] = (),
    body: str = "gather",
    max_unroll: int = 16,
    arch: str = "gfx950",
) -> QueueOp:
    src_modes, dst_modes, ext = coalesce_modes(src_modes, dst_modes, ext)
    spec = PermuteSpec(
        x_shape=_np_shape(src_modes, ext),
        perm=_perm(src_modes, dst_modes),
        block_size=block_size,
    )
    if body == "auto":
        from .transpose_lds import transpose_lds_ok

        body = (
            "tlds"
            if transpose_lds_ok(spec.x_shape, spec.perm, block_size)
            else "gather"
        )
    if body == "tlds":
        from .transpose_lds import TILE, TransposeLdsSpec, build_transpose_lds

        R, C = spec.x_shape
        kdef = build_transpose_lds(
            TransposeLdsSpec(rows=R, cols=C, block_size=block_size), arch=arch
        )
        gx = (R // TILE) * (C // TILE)
        per_blk = TILE * TILE
    elif body == "gather":
        from .permute_gather import (
            PermuteGatherSpec,
            build_permute_gather,
            permute_gather_grid,
        )

        u = max(1, min(max_unroll, elems_per_ticket // (block_size * spec.vec_width)))
        while u > 1 and block_size * u * spec.vec_width > 2 * spec.total_elements:
            u //= 2
        kdef = _with_rank(
            spec,
            lambda: build_permute_gather(
                PermuteGatherSpec(permute=spec, unroll=u), arch=arch
            ),
        )
        gx = permute_gather_grid(spec, u)[0]
        per_blk = block_size * u * spec.vec_width
    elif body == "nd":
        kdef = _build_permute_any_rank(spec, arch)
        gx = permute_grid(spec)[0]
        per_blk = block_size * spec.vec_width
    else:
        raise ValueError(body)
    chunk = max(1, elems_per_ticket // per_blk)
    return QueueOp(
        name=name,
        kind="permute",
        kdef=kdef,
        grid=(gx, 1),
        bind={"X": ("buf", src), "Y": ("buf", dst)},
        deps=tuple(deps),
        chunk=chunk,
    )


def _ring_small_spec(arch: str) -> UniversalGemmSpec:
    """The fp16 dispatcher's ``cdna_dtl_ring_small`` body in its base geometry."""
    from rocke.dispatch.gemm import fp16_rcr
    from rocke.dispatch.gemm.common import dtl_ring_small_tile_trait

    tile, trait = dtl_ring_small_tile_trait(arch)
    return fp16_rcr._make_spec(
        name="universal_gemm_fp16_cdna_dtl_ring_small",
        arch=arch,
        tile=tile,
        trait=trait,
    )


def gemm_spec_for(
    M_r: int,
    N_r: int,
    K: int,
    *,
    block_size: int,
    arch: str = "gfx950",
    forbid_attrs=("agpr_alloc", "mfma_agpr_form"),
    force: Optional[UniversalGemmSpec] = None,
) -> UniversalGemmSpec:
    """The dispatcher's pick for the rocKE RCR problem if it fits the plan's
    block size (and no AGPR-form kernel attrs), else the first eligible
    candidate that does."""
    if force is not None:
        return force
    req = GemmRequest(M=M_r, N=N_r, K=K, arch=arch)
    picks = [dispatch_gemm_fp16(req)] + list(dispatch_gemm_fp16_all(req))
    for r in picks:
        s = r.spec
        if s.block_size != block_size:
            continue
        if getattr(s.trait, "acc_agpr", False) and "agpr_alloc" in forbid_attrs:
            continue
        ok, _ = is_valid_spec(s, arch)
        if ok and launch_shape_error(s, M_r, N_r, K) is None:
            return s
    # no eligible candidate at this block size (e.g. K % 64 == 32: the
    # dispatcher only has 256-thread K32 bodies): the block-size pick of the
    # K-rounded problem with a K32 tile (same ring traits).
    kr = -(-K // 64) * 64
    if kr != K:
        base = gemm_spec_for(
            M_r, N_r, kr, block_size=block_size, arch=arch, forbid_attrs=forbid_attrs
        )
        s = replace(base, tile=replace(base.tile, tile_k=32))
        ok, why = is_valid_spec(s, arch)
        if ok and launch_shape_error(s, M_r, N_r, K) is None:
            return s
    # last resort: the dispatcher's base 512-thread DTL small-ring body
    # (``dtl_ring_small_tile_trait`` without a problem: 128x128x64, 2x4 warps,
    # 2-stage ring with pad and exact DTL bounds), with a K32 tile when
    # K % 64 == 32
    if block_size == 512:
        base = _ring_small_spec(arch)
        for s in (base, replace(base, tile=replace(base.tile, tile_k=32))):
            ok, _ = is_valid_spec(s, arch)
            if (
                ok
                and launch_shape_error(s, M_r, N_r, K) is None
                and K % s.tile.tile_k == 0
            ):
                return s
    raise ValueError(f"no {block_size}-thread GEMM candidate for {M_r}x{N_r}x{K}")


def gemm_grid(spec: UniversalGemmSpec, M_r: int, N_r: int) -> Tuple[int, int]:
    t = spec.tile
    return (-(-N_r // t.tile_n), -(-M_r // t.tile_m))


def _out_map(c: "Contraction", spec: UniversalGemmSpec):
    """GEMM output C[n, m] (flat n * M + m) -> D (col-major over c.modes_d)."""
    stride, acc = {}, 1
    for x in c.modes_d:
        stride[x] = acc
        acc *= c.extent[x]
    md = [(c.extent[x], stride[x]) for x in c.m_modes]
    nd = [(c.extent[x], stride[x]) for x in c.n_modes]
    M = c.size(c.m_modes)
    return ("C", M, spec.tile.tile_m, spec.tile.tile_n, md, nd)


def contraction_ops(
    c: Contraction,
    *,
    a: str,
    b: str,
    d: str,
    block_size: int = 512,
    elems_per_ticket: int = 1 << 16,
    gemm_spec: Optional[UniversalGemmSpec] = None,
    permute_body: str = "gather",
    fuse_out: str = "off",
    arch: str = "gfx950",
) -> Tuple[List[QueueOp], List[QueueOp], List[QueueOp], Dict[str, int]]:
    """(input permutes, [gemm], output permutes, scratch tensors {name: elems}).

    ``fuse_out``: ``off`` = output permute as its own op (E scratch);
    ``on`` = non-canonical outputs are written by the GEMM epilogue directly
    in D's layout (store-index rewrite, no E, no output-permute op);
    ``all`` = also canonical outputs go through the (identity) rewrite, so
    every GEMM of one spec shares one inlined body.
    """
    if fuse_out not in ("off", "on", "all"):
        raise ValueError(fuse_out)
    M, N, K = c.MNK
    canon_out = tuple(c.modes_d) == tuple(c.m_modes) + tuple(c.n_modes)
    fused = fuse_out == "all" or (fuse_out == "on" and not canon_out)
    e = d if (canon_out or fused) else f"{c.name}_E"
    # identity permutes are elided: an operand already K-fastest ([K modes, M modes]) is read in place
    a_in_place = tuple(c.modes_a) == tuple(c.k_modes) + tuple(c.m_modes)
    b_in_place = tuple(c.modes_b) == tuple(c.k_modes) + tuple(c.n_modes)
    ap = a if a_in_place else f"{c.name}_Ap"
    bp = b if b_in_place else f"{c.name}_Bp"
    scratch = {}
    if not a_in_place:
        scratch[ap] = M * K
    if not b_in_place:
        scratch[bp] = N * K
    if not (canon_out or fused):
        scratch[e] = M * N
    pre = []
    if not a_in_place:
        pre.append(
            permute_op(
                f"{c.name}.pA",
                a,
                ap,
                c.modes_a,
                c.k_modes + c.m_modes,
                c.extent,
                block_size=block_size,
                elems_per_ticket=elems_per_ticket,
                body=permute_body,
                arch=arch,
            )
        )
    if not b_in_place:
        pre.append(
            permute_op(
                f"{c.name}.pB",
                b,
                bp,
                c.modes_b,
                c.k_modes + c.n_modes,
                c.extent,
                block_size=block_size,
                elems_per_ticket=elems_per_ticket,
                body=permute_body,
                arch=arch,
            )
        )
    spec = gemm_spec_for(N, M, K, block_size=block_size, arch=arch, force=gemm_spec)
    if fused and spec.trait.chiplet_swizzle:
        # the epilogue rewrite's per-ticket LDS tables are built for the tile the
        # ticket names; the chiplet swizzle remaps that tile inside the body (a
        # launch-order heuristic with no meaning under the ticket queue anyway)
        spec = replace(spec, trait=replace(spec.trait, chiplet_swizzle=False))
    g = QueueOp(
        name=f"{c.name}.gemm",
        kind="gemm",
        kdef=build_universal_gemm(spec, arch),
        grid=gemm_grid(spec, N, M),
        bind={
            "A": ("buf", bp),
            "B": ("buf", ap),
            "C": ("buf", e),
            "M": ("i32", N),
            "N": ("i32", M),
            "K": ("i32", K),
        },
        deps=tuple(p.name for p in pre),
        body_key=spec.kernel_name() + ("_om" if fused else ""),
        out_map=_out_map(c, spec) if fused else None,
    )
    post = []
    if not (canon_out or fused):
        post.append(
            permute_op(
                f"{c.name}.pD",
                e,
                d,
                c.m_modes + c.n_modes,
                c.modes_d,
                c.extent,
                block_size=block_size,
                elems_per_ticket=elems_per_ticket,
                deps=(g.name,),
                body=permute_body,
                arch=arch,
            )
        )
    return pre, [g], post, scratch
