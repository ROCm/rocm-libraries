# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multi-axis tiled N-D permute: 16-byte accesses on both sides.

Computes ``Y = X.permute(*perm).contiguous()`` for a row-major ``X`` (the
same semantics as :mod:`rocke.instances.common.permute_nd`), for 2-, 4- and
8-byte elements (f16 / bf16 / f32 / f64 and their integer twins; a permute
only moves bits, so the kernel moves raw carrier words and is bit-exact for
every pattern, NaN payloads included).

Why a new body
--------------

``permute_nd`` maps one thread to one output vector. When the permutation
moves the innermost axis it falls back to scalar loads, and even when the
innermost axis is kept, a short inner run (e.g. 16 or 32 elements) makes each
wave read dozens of tiny runs that sit megabytes apart in the input. Both
patterns waste DRAM bandwidth. This body instead tiles over the axes that
make the *input* contiguous and the axes that make the *output* contiguous at
the same time, so each CTA reads a few long input runs and writes a few long
output runs:

* ``tile[a]`` is the per-axis tile extent (it divides ``x_shape[a]``). Axes
  with ``tile[a] == 1`` are iterated by the grid only. The planner
  (:func:`plan_permute_tiled`) puts the innermost input axes (input runs) and
  the innermost output axes (output runs) into the tile.
* Phase 1 reads the tile in input order with 16-byte loads (``vec`` elements
  along the input's innermost axis ``ai``).
* When the output's innermost axis ``ao`` differs from ``ai`` (a "moving"
  permute), each thread loads a ``vec x vec`` block (``vec`` loads along
  ``ao``) and transposes it in registers (``vector.extract`` /
  ``vector.pack``, which the AMDGPU backend lowers to ``v_perm_b32`` /
  ``v_alignbit`` byte permutes for 16-bit elements and plain register moves
  for 32/64-bit ones), so every LDS write is a 16-byte vector along ``ao``.
* ``stage="lds"``: the vectors go to an LDS tile laid out in output order,
  with an XOR swizzle of the 16-byte chunk index (``swizzle_k``,
  ``swizzle_shift``) that the planner picks with a static bank model; after
  one barrier, phase 2 reads 16-byte vectors in output order and writes them
  with 16-byte stores, so both global sides are fully coalesced.
  ``stage="reg"``: no LDS, the (transposed) vectors are stored straight to
  global memory (the in-register transpose body generalised to N-D).
* The tile base is folded into 64-bit pointers once per CTA
  (``global_ptr_add``), so tensors beyond 2^31 elements are addressed
  correctly; in-tile offsets stay 32-bit and are bounded by the validator.
* ``grid_order`` picks which tile coordinate is fastest across consecutive
  CTAs (input order or output order); ``nt`` adds the streaming
  (``!nontemporal``) hint to the loads, the stores or both (gfx942 / gfx950).
* ``tiles_per_cta > 1`` runs a persistent loop over that many consecutive
  tiles per CTA.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from ...core.ir import F64, I16, I32, I64, IRBuilder, KernelDef, PtrType, TemporalHint
from ...helpers.spec import SignatureBuilder, kernel_name_join

DType = Literal["f16", "bf16", "i16", "f32", "i32", "f64", "i64"]
MAX_RANK = 16
ELEM_BYTES = {"f16": 2, "bf16": 2, "i16": 2, "f32": 4, "i32": 4, "f64": 8, "i64": 8}
_CARRIER = {2: I16, 4: I32, 8: F64}
_CARRIER_SIG = {2: "i16", 4: "i32", 8: "f64"}
_BLOCK_SIZES = (64, 128, 256, 512, 1024)
# Upper bound on vectors a thread holds in phase 1 (each is 16 bytes = 4 VGPRs).
MAX_VECTORS_PER_THREAD = 64
_I32_MAX = 2**31 - 1


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------


def canonicalize_permute(
    x_shape: Sequence[int], perm: Sequence[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Drop extent-1 axes and merge input axes that stay adjacent in the output.

    Returns ``(shape, perm)`` of the smallest-rank row-major permute that moves
    every element to the same flat output position. An identity permute
    canonicalises to a rank-1 ``(N,)`` / ``(0,)``.
    """
    n = len(x_shape)
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm {list(perm)} is not a permutation of range({n})")
    keep = [a for a in range(n) if int(x_shape[a]) != 1]
    if not keep:
        return (1,), (0,)
    renum = {a: i for i, a in enumerate(keep)}
    shape = [int(x_shape[a]) for a in keep]
    p = [renum[a] for a in perm if a in renum]
    groups: List[List[int]] = []
    for a in p:
        if groups and groups[-1][-1] + 1 == a:
            groups[-1].append(a)
        else:
            groups.append([a])
    order = sorted(range(len(groups)), key=lambda g: groups[g][0])
    new_index = {g: i for i, g in enumerate(order)}
    c_shape = tuple(_prod(shape[a] for a in groups[g]) for g in order)
    c_perm = tuple(new_index[g] for g in range(len(groups)))
    return c_shape, c_perm


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermuteTiledSpec:
    """One tiled-permute kernel configuration (shape-specialised)."""

    x_shape: Tuple[int, ...]
    perm: Tuple[int, ...]
    tile: Tuple[int, ...]
    dtype: DType = "f16"
    block_size: int = 256
    stage: Literal["lds", "reg", "gather"] = "lds"
    swizzle_k: int = 0
    """XOR group of the LDS 16-byte chunk index (1 / 0 = no swizzle)."""
    swizzle_shift: int = 0
    """Chunk-index bit the swizzle key is taken from (``key = (c >> shift) & (k-1)``)."""
    grid_order: Literal["in", "out", "both"] = "in"
    """Tile walk across consecutive CTAs: ``in`` / ``out`` follow the input /
    output axis order; ``both`` makes the fastest grid axis of the input order
    and the fastest grid axis of the output order the two fastest walk axes, so
    a short window of consecutive CTAs completes whole contiguous rows on both
    sides (inner-preserving permutes with short inner runs)."""
    nt: Literal["none", "load", "store", "both"] = "none"
    tiles_per_cta: int = 1
    tail: Literal["clamp", "guard"] = "clamp"
    skew: Literal["none", "diag", "hash"] = "none"
    """Tile-grid skew against memory-channel camping: the grid axis that is
    fastest in the *other* side's order is shifted by the fastest grid axis of
    the walk (``diag``) or by the sum of all other tile coordinates
    (``hash``), so concurrently running CTAs spread over that side's low
    address bits. A shift per fixed other coordinates, so still a bijection."""
    """Threads past the last unit of a phase: ``clamp`` redoes the last unit
    (identical data, no branch); ``guard`` skips it under a branch."""
    name: str = "rocke_permute_tiled"

    # --- derived geometry ---------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.x_shape)

    @property
    def elem_bytes(self) -> int:
        return ELEM_BYTES[self.dtype]

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return tuple(self.x_shape[a] for a in self.perm)

    @property
    def ai(self) -> int:
        return self.rank - 1

    @property
    def ao(self) -> int:
        return self.perm[-1]

    @property
    def moving(self) -> bool:
        return self.ai != self.ao

    @property
    def run_in(self) -> int:
        """Contiguous input elements per tile run (input-innermost axes, up to the first partial one)."""
        r = 1
        for a in range(self.rank - 1, -1, -1):
            r *= self.tile[a]
            if self.tile[a] < self.x_shape[a]:
                break
        return r

    @property
    def run_out(self) -> int:
        """Contiguous output elements per tile run (output-innermost axes, up to the first partial one)."""
        r = 1
        for d in range(self.rank - 1, -1, -1):
            a = self.perm[d]
            r *= self.tile[a]
            if self.tile[a] < self.x_shape[a]:
                break
        return r

    @property
    def vec(self) -> int:
        """Elements per 16-byte access (smaller when the tile does not divide).

        ``lds`` / ``reg``: ``vec`` must divide the tile extent of the input's
        innermost axis (and of the output's innermost axis when it moves).
        ``gather``: ``vec`` must divide the whole contiguous runs on both sides,
        so axes with odd extents still get 16-byte accesses.
        """
        v = 16 // self.elem_bytes
        if self.stage == "gather":
            while v > 1 and (self.run_in % v or self.run_out % v):
                v //= 2
            return v
        while v > 1 and (
            self.tile[self.ai] % v or (self.moving and self.tile[self.ao] % v)
        ):
            v //= 2
        return v

    @property
    def tile_elems(self) -> int:
        return _prod(self.tile)

    @property
    def n_tiles(self) -> int:
        return _prod(self.x_shape[a] // self.tile[a] for a in range(self.rank))

    @property
    def n_ctas(self) -> int:
        return -(-self.n_tiles // self.tiles_per_cta)

    @property
    def phase1_units(self) -> int:
        v = self.vec
        if self.stage == "gather":
            return self.tile_elems // v
        return self.tile_elems // (v * v if self.moving else v)

    @property
    def phase2_units(self) -> int:
        return self.tile_elems // self.vec

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            "s" + "x".join(str(d) for d in self.x_shape),
            "p" + "".join(str(d) for d in self.perm),
            "t" + "x".join(str(d) for d in self.tile),
            f"b{self.block_size}",
            self.stage,
            f"z{self.swizzle_k}s{self.swizzle_shift}" if self.swizzle_k > 1 else "",
            {"in": "", "out": "go", "both": "gb"}[self.grid_order],
            f"nt{self.nt}" if self.nt != "none" else "",
            f"tc{self.tiles_per_cta}" if self.tiles_per_cta > 1 else "",
            "tg" if self.tail == "guard" else "",
            f"sk{self.skew}" if self.skew != "none" else "",
        )


def _strides(shape: Sequence[int]) -> List[int]:
    st = [1] * len(shape)
    for a in range(len(shape) - 2, -1, -1):
        st[a] = st[a + 1] * int(shape[a + 1])
    return st


def _geometry(spec: PermuteTiledSpec):
    n = spec.rank
    s = list(spec.x_shape)
    p = list(spec.perm)
    pinv = [0] * n
    for d, a in enumerate(p):
        pinv[a] = d
    st_in = _strides(s)
    st_out_pos = _strides([s[a] for a in p])
    st_out = [st_out_pos[pinv[a]] for a in range(n)]
    t = list(spec.tile)
    # tile-local strides of the output-ordered LDS tile
    so = [0] * n
    acc = 1
    for d in range(n - 1, -1, -1):
        a = p[d]
        so[a] = acc
        acc *= t[a]
    return s, p, pinv, st_in, st_out, t, so


def is_valid_spec(spec: PermuteTiledSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """``(ok, reason)`` for a tiled-permute spec on ``arch``."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.dtype not in ELEM_BYTES:
        return False, f"unsupported dtype {spec.dtype!r}"
    n = spec.rank
    if n < 1 or n > MAX_RANK:
        return False, f"rank {n} outside 1..{MAX_RANK}"
    if sorted(spec.perm) != list(range(n)):
        return False, f"perm {list(spec.perm)} is not a permutation of range({n})"
    if len(spec.tile) != n:
        return False, f"tile rank {len(spec.tile)} != x_shape rank {n}"
    for a in range(n):
        if spec.x_shape[a] <= 0:
            return False, f"x_shape must be positive, got {list(spec.x_shape)}"
        if spec.tile[a] <= 0 or spec.x_shape[a] % spec.tile[a]:
            return False, f"tile[{a}]={spec.tile[a]} does not divide x_shape[{a}]={spec.x_shape[a]}"
    if spec.stage not in ("lds", "reg", "gather"):
        return False, f"unknown stage {spec.stage!r}"
    if spec.grid_order not in ("in", "out", "both"):
        return False, f"unknown grid_order {spec.grid_order!r}"
    if spec.nt not in ("none", "load", "store", "both"):
        return False, f"unknown nt {spec.nt!r}"
    if spec.nt != "none" and arch not in ("gfx942", "gfx950"):
        return False, f"streaming hint not validated on {arch}"
    bs = spec.block_size
    legal = tuple(x for x in _BLOCK_SIZES if x <= target.max_threads_per_block)
    if bs not in legal:
        return False, f"block_size {bs} not in {legal} on {arch}"
    if bs % target.wave_size:
        return False, f"block_size {bs} not a multiple of wave{target.wave_size}"
    u1 = spec.phase1_units
    v = spec.vec
    held = (-(-u1 // bs)) * (v if (spec.moving and spec.stage != "gather") else 1)
    if spec.stage == "gather":
        held = max(held, -(-spec.phase2_units // bs))
    if held > MAX_VECTORS_PER_THREAD:
        return False, f"{held} vectors per thread > {MAX_VECTORS_PER_THREAD} (register budget)"
    if spec.stage in ("lds", "gather") and not target.fits_lds(spec.tile_elems * spec.elem_bytes):
        return False, f"LDS tile {spec.tile_elems * spec.elem_bytes} B exceeds {arch} capacity"
    k = spec.swizzle_k
    if k not in (0, 1) and spec.stage == "gather":
        return False, "the gather stage has no LDS swizzle"
    if k not in (0, 1) and spec.stage == "lds":
        if not _is_pow2(k):
            return False, f"swizzle_k {k} must be a power of two"
        chunks = spec.tile_elems // v
        if (1 << spec.swizzle_shift) < k or chunks % (1 << spec.swizzle_shift):
            return False, (
                f"swizzle (k={k}, shift={spec.swizzle_shift}) is not a bijection on "
                f"{chunks} chunks"
            )
    # in-tile offsets stay 32-bit
    s, p, pinv, st_in, st_out, t, so = _geometry(spec)
    span_in = sum((t[a] - 1) * st_in[a] for a in range(n)) + 1
    span_out = sum((t[a] - 1) * st_out[a] for a in range(n)) + 1
    if max(span_in, span_out) > _I32_MAX:
        return False, "in-tile offset span exceeds 2^31 elements"
    if spec.skew not in ("none", "diag", "hash"):
        return False, f"unknown skew {spec.skew!r}"
    if spec.tail not in ("clamp", "guard"):
        return False, f"unknown tail {spec.tail!r}"
    if spec.tiles_per_cta < 1:
        return False, "tiles_per_cta must be >= 1"
    if spec.n_ctas > _I32_MAX:
        return False, "grid exceeds 2^31 CTAs"
    return True, "ok"


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


def _divc(b: IRBuilder, x, d: int):
    if d == 1:
        return x
    if _is_pow2(d):
        return b.lshr(x, b.const_i32(d.bit_length() - 1))
    return b.div(x, b.const_i32(d))


def _modc(b: IRBuilder, x, d: int):
    if d == 1:
        return b.const_i32(0)
    if _is_pow2(d):
        return b.land(x, b.const_i32(d - 1))
    return b.mod(x, b.const_i32(d))


def _unflatten(b: IRBuilder, x, dims: Sequence[Tuple[int, int]]) -> Dict[int, object]:
    """Split ``x`` over ``dims`` = [(axis, extent), ...] (slowest first)."""
    out: Dict[int, object] = {}
    rem = x
    live = [(a, e) for a, e in dims if e > 1]
    for i, (a, e) in enumerate(reversed(live)):
        last = i == len(live) - 1
        out[a] = rem if last else _modc(b, rem, e)
        if not last:
            rem = _divc(b, rem, e)
    for a, e in dims:
        if e <= 1:
            out[a] = None
    return out


def _lin(b: IRBuilder, coords: Dict[int, object], weights: Dict[int, int], const: int = 0):
    acc = None
    for a, w in weights.items():
        c = coords.get(a)
        if c is None or w == 0:
            continue
        term = c if w == 1 else b.mul(c, b.const_i32(w))
        acc = term if acc is None else b.add(acc, term)
    if acc is None:
        return b.const_i32(const)
    return b.add(acc, b.const_i32(const)) if const else acc


def _swz(b: IRBuilder, chunk, k: int, shift: int):
    if k <= 1:
        return chunk
    key = b.land(b.lshr(chunk, b.const_i32(shift)), b.const_i32(k - 1))
    return b.xor(chunk, key)


def grid_walk(spec: PermuteTiledSpec) -> List[int]:
    """Grid axes from slowest to fastest across consecutive CTAs."""
    n = spec.rank
    if spec.grid_order == "in":
        return list(range(n))
    if spec.grid_order == "out":
        return list(spec.perm)
    g = [spec.x_shape[a] // spec.tile[a] for a in range(n)]
    fi = next((a for a in range(n - 1, -1, -1) if g[a] > 1), None)
    fo = next((a for a in reversed(spec.perm) if g[a] > 1), None)
    fast = [a for a in (fo, fi) if a is not None]
    fast = list(dict.fromkeys(fast))
    return [a for a in range(n) if a not in fast] + fast


def skew_axes(spec: PermuteTiledSpec):
    """``(walk_axis, skewed_axis)`` of the grid skew, or ``None`` when it is a no-op."""
    n = spec.rank
    g = [spec.x_shape[a] // spec.tile[a] for a in range(n)]
    walk = grid_walk(spec)
    other = list(spec.perm) if spec.grid_order != "out" else list(range(n))
    fi = next((a for a in reversed(walk) if g[a] > 1), None)
    fo = next((a for a in reversed(other) if g[a] > 1), None)
    if spec.skew == "none" or fi is None or fo is None or fi == fo:
        return None
    return fi, fo


def _tile_coords(b: IRBuilder, spec: PermuteTiledSpec, tile_id, g, order):
    tc = _unflatten(b, tile_id, [(a, g[a]) for a in order])
    ax = skew_axes(spec)
    if ax is None:
        return tc
    fi, fo = ax
    if spec.skew == "diag":
        shift = tc[fi]
    else:
        shift = None
        for a in range(spec.rank):
            if a != fo and tc[a] is not None:
                shift = tc[a] if shift is None else b.add(shift, tc[a])
    tc = dict(tc)
    tc[fo] = _modc(b, b.add(tc[fo], shift), g[fo])
    return tc


def build_permute_tiled(spec: PermuteTiledSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR. Signature ``(X: ptr<carrier>, Y: ptr<carrier>)``; grid ``(n_ctas, 1, 1)``."""
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid permute_tiled spec: {why}")
    n = spec.rank
    es = spec.elem_bytes
    ct = _CARRIER[es]
    V = spec.vec
    BS = spec.block_size
    s, p, pinv, st_in, st_out, t, so = _geometry(spec)
    ai, ao, moving = spec.ai, spec.ao, spec.moving
    g = [s[a] // t[a] for a in range(n)]
    hint_ld = TemporalHint.STREAMING if spec.nt in ("load", "both") else TemporalHint.DEFAULT
    hint_st = TemporalHint.STREAMING if spec.nt in ("store", "both") else TemporalHint.DEFAULT

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS
    X = b.param("X", PtrType(ct, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(ct, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()

    lds = None
    if spec.stage in ("lds", "gather"):
        lds = b.smem_alloc(ct, [spec.tile_elems], name_hint="perm_tile")
    if spec.stage == "gather":
        return _build_gather(b, spec, X, Y, tid, lds, ct, hint_ld, hint_st)

    # phase-1 unit geometry (input order, ai fastest)
    bt = list(t)
    bt[ai] = t[ai] // V
    if moving:
        bt[ao] = t[ao] // V
    p1_dims = [(a, bt[a]) for a in range(n)]
    w_in = {a: st_in[a] * (V if (a == ai or (moving and a == ao)) else 1) for a in range(n)}
    w_out = {a: st_out[a] * (V if (a == ai or (moving and a == ao)) else 1) for a in range(n)}
    w_lds = {a: so[a] * (V if (a == ai or (moving and a == ao)) else 1) for a in range(n)}
    # phase-2 unit geometry (output order, ao fastest, units of V elements)
    p2_ext = list(t)
    p2_ext[ao] = t[ao] // V
    p2_dims = [(p[d], p2_ext[p[d]]) for d in range(n)]
    w2_out = {a: st_out[a] * (V if a == ao else 1) for a in range(n)}
    u1n, u2n = spec.phase1_units, spec.phase2_units
    it1 = -(-u1n // BS)
    it2 = -(-u2n // BS)

    def unit(it, total):
        """``(u, guard)``: guard is the in-range predicate of a guarded tail iteration, else None."""
        u = tid if it == 0 else b.add(tid, b.const_i32(it * BS))
        if (it + 1) * BS > total:
            if spec.tail == "guard":
                return u, b.cmp_lt(u, b.const_i32(total))
            # threads past the last unit redo the last one (identical data, benign)
            u = b.smin(u, b.const_i32(total - 1))
        return u, None
    order = grid_walk(spec)

    def one_tile(tile_id):
        tc = _tile_coords(b, spec, tile_id, g, order)
        in_base = None
        out_base = None
        for a in range(n):
            if tc[a] is None:
                continue
            c64 = b.sext(tc[a], I64)
            ti = b.mul(c64, b.const_i64(t[a] * st_in[a] * es))
            to = b.mul(c64, b.const_i64(t[a] * st_out[a] * es))
            in_base = ti if in_base is None else b.add(in_base, ti)
            out_base = to if out_base is None else b.add(out_base, to)
        Xt = X if in_base is None else b.global_ptr_add(X, in_base)
        Yt = Y if out_base is None else b.global_ptr_add(Y, out_base)

        # ---- phase 1: loads (all iterations first) ----
        def p1_load(u):
            cb = _unflatten(b, u, p1_dims)
            off = _lin(b, cb, w_in)
            if moving:
                rows = [
                    b.global_load_vN(
                        Xt, off if j == 0 else b.add(off, b.const_i32(j * st_in[ao])),
                        ct, V, temporal_hint=hint_ld,
                    ) if V > 1 else b.global_load(
                        Xt, off if j == 0 else b.add(off, b.const_i32(j * st_in[ao])), ct
                    )
                    for j in range(V)
                ]
            else:
                rows = [
                    b.global_load_vN(Xt, off, ct, V, temporal_hint=hint_ld)
                    if V > 1 else b.global_load(Xt, off, ct)
                ]
            return cb, rows

        # ---- in-register transpose + LDS / direct store ----
        def p1_store(cb, rows):
            if moving and V > 1:
                vecs = [
                    b.vec_pack([b.vec_extract(rows[j], i) for j in range(V)], ct)
                    for i in range(V)
                ]
                steps = [(i * so[ai], i * st_out[ai]) for i in range(V)]
            elif moving:
                vecs = rows  # V == 1: scalar elements, one per j along ao
                steps = [(0, 0)]
                vecs = [rows[0]]
            else:
                vecs = rows
                steps = [(0, 0)]
            if spec.stage == "lds":
                base = _lin(b, cb, w_lds)
                for vv, (dl, _) in zip(vecs, steps):
                    e = base if dl == 0 else b.add(base, b.const_i32(dl))
                    if V > 1:
                        ch = _swz(b, _divc(b, e, V), spec.swizzle_k, spec.swizzle_shift)
                        b.smem_store_vN(lds, [b.mul(ch, b.const_i32(V))], vv, V)
                    else:
                        b.smem_store_vN(lds, [e], vv, 1)
            else:
                base = _lin(b, cb, w_out)
                for vv, (_, do) in zip(vecs, steps):
                    e = base if do == 0 else b.add(base, b.const_i32(do))
                    if V > 1:
                        b.global_store_vN(Yt, e, vv, V, temporal_hint=hint_st)
                    else:
                        b.global_store(Yt, e, vv)

        regs, tails = [], []
        for it in range(it1):
            u, guard = unit(it, u1n)
            if guard is None:
                regs.append(p1_load(u))
            else:
                tails.append((u, guard))
        for cb, rows in regs:
            p1_store(cb, rows)
        for u, guard in tails:
            with b.scf_if(guard):
                p1_store(*p1_load(u))

        if spec.stage == "lds":
            b.sync()

            def p2_load(u):
                ch = _swz(b, u, spec.swizzle_k, spec.swizzle_shift)
                if V > 1:
                    return b.smem_load_vN(lds, b.mul(ch, b.const_i32(V)), dtype=ct, n=V)
                return b.vec_extract(b.smem_load_vN(lds, ch, dtype=ct, n=1), 0)

            def p2_store(u, v):
                c2 = _unflatten(b, u, p2_dims)
                off = _lin(b, c2, w2_out)
                if V > 1:
                    b.global_store_vN(Yt, off, v, V, temporal_hint=hint_st)
                else:
                    b.global_store(Yt, off, v)

            vals, tails2 = [], []
            for it in range(it2):
                u, guard = unit(it, u2n)
                if guard is None:
                    vals.append((u, p2_load(u)))
                else:
                    tails2.append((u, guard))
            for u, v in vals:
                p2_store(u, v)
            for u, guard in tails2:
                with b.scf_if(guard):
                    p2_store(u, p2_load(u))

    bid = b.block_id_x()
    if spec.tiles_per_cta == 1:
        one_tile(bid)
    else:
        tpc = spec.tiles_per_cta
        first = b.mul(bid, b.const_i32(tpc))
        for k in range(tpc):
            tile_id = first if k == 0 else b.add(first, b.const_i32(k))
            if spec.n_tiles % tpc:
                with b.scf_if(b.cmp_lt(tile_id, b.const_i32(spec.n_tiles))):
                    one_tile(tile_id)
            else:
                one_tile(tile_id)
            if spec.stage == "lds" and k + 1 < tpc:
                b.sync()
    return b.kernel


def _build_gather(b, spec, X, Y, tid, lds, ct, hint_ld, hint_st):
    """``stage="gather"``: 16-byte copies of the tile's input runs into an
    input-ordered LDS tile, one barrier, then each thread gathers ``vec``
    output-consecutive elements from LDS and writes them with one 16-byte
    store. Works whenever both run lengths are multiples of ``vec``, whatever
    the individual axis extents (odd modes, short innermost axes)."""
    n = spec.rank
    es = spec.elem_bytes
    V = spec.vec
    BS = spec.block_size
    s, p, pinv, st_in, st_out, t, so = _geometry(spec)
    g = [s[a] // t[a] for a in range(n)]
    si = [0] * n  # input-ordered tile-local strides
    acc = 1
    for a in range(n - 1, -1, -1):
        si[a] = acc
        acc *= t[a]
    u1n, u2n = spec.phase1_units, spec.phase2_units
    it1, it2 = -(-u1n // BS), -(-u2n // BS)
    order = grid_walk(spec)
    in_dims = [(a, t[a]) for a in range(n)]
    out_dims = [(p[d], t[p[d]]) for d in range(n)]

    def unit(it, total):
        u = tid if it == 0 else b.add(tid, b.const_i32(it * BS))
        if (it + 1) * BS > total:
            if spec.tail == "guard":
                return u, b.cmp_lt(u, b.const_i32(total))
            u = b.smin(u, b.const_i32(total - 1))
        return u, None

    def one_tile(tile_id):
        tc = _tile_coords(b, spec, tile_id, g, order)
        in_base = out_base = None
        for a in range(n):
            if tc[a] is None:
                continue
            c64 = b.sext(tc[a], I64)
            ti = b.mul(c64, b.const_i64(t[a] * st_in[a] * es))
            to = b.mul(c64, b.const_i64(t[a] * st_out[a] * es))
            in_base = ti if in_base is None else b.add(in_base, ti)
            out_base = to if out_base is None else b.add(out_base, to)
        Xt = X if in_base is None else b.global_ptr_add(X, in_base)
        Yt = Y if out_base is None else b.global_ptr_add(Y, out_base)

        def g_load(u):
            e = b.mul(u, b.const_i32(V)) if V > 1 else u
            c = _unflatten(b, e, in_dims)
            off = _lin(b, c, {a: st_in[a] for a in range(n)})
            v = (b.global_load_vN(Xt, off, ct, V, temporal_hint=hint_ld) if V > 1
                 else b.global_load(Xt, off, ct))
            return e, v

        loads, tails = [], []
        for it in range(it1):
            u, guard = unit(it, u1n)
            if guard is None:
                loads.append(g_load(u))
            else:
                tails.append((u, guard))
        for e, v in loads:
            b.smem_store_vN(lds, [e], v, V if V > 1 else 1)
        for u, guard in tails:
            with b.scf_if(guard):
                e, v = g_load(u)
                b.smem_store_vN(lds, [e], v, V if V > 1 else 1)
        b.sync()

        def g_gather(u):
            q0 = b.mul(u, b.const_i32(V)) if V > 1 else u
            elems = []
            base_off = None
            for j in range(V):
                q = q0 if j == 0 else b.add(q0, b.const_i32(j))
                c = _unflatten(b, q, out_dims)
                if j == 0:
                    base_off = _lin(b, c, {a: st_out[a] for a in range(n)})
                li = _lin(b, c, {a: si[a] for a in range(n)})
                elems.append(b.vec_extract(b.smem_load_vN(lds, li, dtype=ct, n=1), 0))
            return base_off, elems

        def g_store(off, elems):
            if V > 1:
                b.global_store_vN(Yt, off, b.vec_pack(elems, ct), V, temporal_hint=hint_st)
            else:
                b.global_store(Yt, off, elems[0])

        outs, tails2 = [], []
        for it in range(it2):
            u, guard = unit(it, u2n)
            if guard is None:
                outs.append(g_gather(u))
            else:
                tails2.append((u, guard))
        for off, elems in outs:
            g_store(off, elems)
        for u, guard in tails2:
            with b.scf_if(guard):
                g_store(*g_gather(u))

    bid = b.block_id_x()
    tpc = spec.tiles_per_cta
    if tpc == 1:
        one_tile(bid)
    else:
        first = b.mul(bid, b.const_i32(tpc))
        for k in range(tpc):
            tile_id = first if k == 0 else b.add(first, b.const_i32(k))
            if spec.n_tiles % tpc:
                with b.scf_if(b.cmp_lt(tile_id, b.const_i32(spec.n_tiles))):
                    one_tile(tile_id)
            else:
                one_tile(tile_id)
            if k + 1 < tpc:
                b.sync()
    return b.kernel


def permute_tiled_grid(spec: PermuteTiledSpec) -> Tuple[int, int, int]:
    return (spec.n_ctas, 1, 1)


def permute_tiled_signature(spec: PermuteTiledSpec):
    c = _CARRIER_SIG[spec.elem_bytes]
    return SignatureBuilder().ptr("X", c).ptr("Y", c).build()


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def lds_bank_cost(spec: PermuteTiledSpec, banks: int = 64, wave: int = 64) -> int:
    """Static LDS bank-conflict cost of one wave's accesses (both phases).

    Model: a 16-byte access occupies 4 consecutive 4-byte banks; the lanes of
    a wave are served in groups of ``banks // 4``; a group costs the largest
    number of its lanes that map to the same 16-byte bank slot. Returns the
    summed cost over the first wave's phase-1 stores and phase-2 loads (lower
    is better; a conflict-free wave costs one cycle per group).
    """
    if spec.stage != "lds":
        return 0
    n = spec.rank
    V = spec.vec
    s, p, pinv, st_in, st_out, t, so = _geometry(spec)
    ai, ao, moving = spec.ai, spec.ao, spec.moving
    bt = list(t)
    bt[ai] = t[ai] // V
    if moving:
        bt[ao] = t[ao] // V
    slots = max(1, banks * 4 // max(16, V * spec.elem_bytes))
    group = slots

    def swz(c):
        k, sh = spec.swizzle_k, spec.swizzle_shift
        return c ^ ((c >> sh) & (k - 1)) if k > 1 else c

    def unflat(u, dims):
        out = {}
        for a, e in reversed(dims):
            out[a] = u % e
            u //= e
        return out

    cost = 0
    lanes = range(min(wave, spec.block_size))
    nvec = V if moving else 1
    for i in range(nvec):
        addrs = []
        for u in lanes:
            cb = unflat(u, [(a, bt[a]) for a in range(n)])
            e = 0
            for a in range(n):
                mul = V if (a == ai or (moving and a == ao)) else 1
                e += cb[a] * mul * so[a]
            e += i * so[ai] if moving else 0
            addrs.append(swz(e // V) if V > 1 else e)
        for g0 in range(0, len(addrs), group):
            grp = addrs[g0:g0 + group]
            hist: Dict[int, int] = {}
            for c in grp:
                hist[c % slots] = hist.get(c % slots, 0) + 1
            cost += max(hist.values())
    addrs = [swz(u) for u in lanes]
    for g0 in range(0, len(addrs), group):
        grp = addrs[g0:g0 + group]
        hist = {}
        for c in grp:
            hist[c % slots] = hist.get(c % slots, 0) + 1
        cost += max(hist.values())
    return cost


def best_swizzle(spec: PermuteTiledSpec, banks: int = 64, wave: int = 64) -> Tuple[int, int]:
    """The (k, shift) swizzle with the lowest static bank cost (ties: no swizzle)."""
    import dataclasses

    best = (lds_bank_cost(dataclasses.replace(spec, swizzle_k=0, swizzle_shift=0), banks, wave), 0, 0)
    chunks = spec.tile_elems // spec.vec
    for k in (2, 4, 8, 16):
        for sh in range(k.bit_length() - 1, max(chunks.bit_length(), 1)):
            if chunks % (1 << sh):
                continue
            c = lds_bank_cost(dataclasses.replace(spec, swizzle_k=k, swizzle_shift=sh), banks, wave)
            if c < best[0]:
                best = (c, k, sh)
    return best[1], best[2]


def _axis_chunks(extent: int, want: int) -> int:
    """Largest divisor of ``extent`` that is <= ``want`` (at least 1)."""
    best = 1
    for d in range(1, min(extent, want) + 1):
        if extent % d == 0:
            best = d
    return best


def default_tile(
    shape: Sequence[int], perm: Sequence[int], dtype: str, run_bytes: int = 256,
    max_tile_bytes: int = 32768,
) -> Tuple[int, ...]:
    """Pick tile extents: input runs and output runs of about ``run_bytes``.

    Walks the input axes innermost-first, then the output axes innermost-first,
    adding each axis (whole, or the divisor chunk that completes the run) until
    the run reaches ``run_bytes`` on each side; then trims to ``max_tile_bytes``
    by halving the outermost chunked axis where possible.
    """
    n = len(shape)
    es = ELEM_BYTES[dtype]
    want = max(1, run_bytes // es)
    t = [1] * n

    def grow(axes):
        run = 1
        for a in axes:
            if run >= want:
                break
            need = -(-want // run)
            if t[a] >= need:
                run *= t[a]
                continue
            c = _axis_chunks(shape[a], max(need, t[a]))
            if c < need and c < shape[a]:
                # the next divisor up keeps the run whole
                for d in range(need, shape[a] + 1):
                    if shape[a] % d == 0:
                        c = d
                        break
            t[a] = max(t[a], c)
            run *= t[a]

    grow(list(range(n - 1, -1, -1)))
    grow([perm[d] for d in range(n - 1, -1, -1)])
    v = 16 // es
    for a in (n - 1, perm[-1]):
        if t[a] % v and shape[a] % v == 0:
            t[a] = _axis_chunks(shape[a], max(v, t[a] - t[a] % v + v)) if shape[a] >= v else t[a]
    moving = perm[-1] != n - 1
    min_elems = 64 * v * (v if moving else 1)

    def next_div(a):
        for d in range(t[a] + 1, shape[a] + 1):
            if shape[a] % d == 0:
                return d
        return t[a]

    grow_order = list(range(n - 1, -1, -1)) + [perm[d] for d in range(n - 1, -1, -1)]
    while _prod(t) < min_elems:
        for a in grow_order:
            if t[a] < shape[a]:
                t[a] = next_div(a)
                break
        else:
            break
    while _prod(t) * es > max_tile_bytes:
        cand = [a for a in range(n) if t[a] > 1 and t[a] % 2 == 0 and a not in (n - 1, perm[-1])]
        if not cand:
            # the innermost axes: shrink to a smaller divisor that keeps vec dividing it
            cand = [a for a in range(n) if any(shape[a] % d == 0 and d % v == 0 for d in range(v, t[a]))]
            if not cand:
                break
            a = max(cand, key=lambda x: t[x])
            t[a] = max(d for d in range(v, t[a]) if shape[a] % d == 0 and d % v == 0)
            continue
        a = max(cand, key=lambda x: t[x])
        t[a] //= 2
    return tuple(t)


def _shrink_tile(shape, perm, tile, dtype, max_tile_bytes):
    """Shrink a tile to ``max_tile_bytes`` from the side whose contiguous run is longest.

    Each step lowers the outermost tiled axis of the longer run (input or
    output order) to its next smaller divisor, keeping ``vec`` dividing the
    innermost axes; the shorter run is never cut while the longer one can be.
    """
    n = len(shape)
    es = ELEM_BYTES[dtype]
    v = 16 // es
    t = list(tile)

    def run(order):
        r, last = 1, None
        for a in order:
            if t[a] == 1:
                break
            r *= t[a]
            last = a
            if t[a] < shape[a]:
                break
        return r, last

    def prev_div(a):
        for d in range(t[a] - 1, 0, -1):
            if shape[a] % d == 0 and (a not in (n - 1, perm[-1]) or d % v == 0 or d == 1):
                return d
        return t[a]

    in_order = list(range(n - 1, -1, -1))
    out_order = [perm[d] for d in range(n - 1, -1, -1)]
    for _ in range(256):
        if _prod(t) * es <= max_tile_bytes:
            break
        rin, ain = run(in_order)
        rout, aout = run(out_order)
        # axes outside both runs first (they only add rows), then the longer run's outer axis
        outside = [a for a in range(n) if t[a] > 1 and a not in in_order[:in_order.index(ain) + 1]
                   and a not in out_order[:out_order.index(aout) + 1]] if ain is not None and aout is not None else []
        order = outside + ([ain, aout] if rin >= rout else [aout, ain])
        for a in order:
            if a is None:
                continue
            d = prev_div(a)
            if d < t[a]:
                t[a] = d
                break
        else:
            break
    return tuple(t)


#: Product-route cost model (relative time; lower is better). Chosen on gfx950 by
#: the regret of its pick over measured tuning observations of a generality set
#: (GETT operand packs and random rank-2..6 permutes, 2 / 4 / 8-byte elements;
#: held-out problems checked separately): per-side run term ``1 + run_a / run``
#: (bytes, capped), access width ``(16 / width) ** width_exp``, stage factors,
#: tile-size factors, idle-thread share, and a parallelism term
#: ``max(1, waves_target / waves) ** waves_exp`` that keeps small problems from
#: running on too few CTAs.
PLAN_COST = {
    "run_a": 48.0, "run_cap": 2048, "width_exp": 0.4, "gather": 1.3, "reg": 1.2, "small_tile": 4096,
    "small_f": 1.15, "mid_tile": 32768, "mid_f": 1.05, "big_tile": 65536, "big_f": 1.2, "idle": 0.1,
    "waves_target": 2048, "waves_exp": 0.5, "tpc": 1.05,
}


def _plan_cost(spec: "PermuteTiledSpec", th: Optional[Dict[str, float]] = None) -> float:
    """Relative time estimate of a planned spec under :data:`PLAN_COST`."""
    th = PLAN_COST if th is None else th
    es = spec.elem_bytes
    rin = min(spec.run_in * es, th["run_cap"])
    rout = min(spec.run_out * es, th["run_cap"])
    c = 0.5 * ((1 + th["run_a"] / rin) + (1 + th["run_a"] / rout))
    c *= (16.0 / (spec.vec * es)) ** th["width_exp"]
    if spec.stage == "gather":
        c *= th["gather"]
    if spec.stage == "reg":
        c *= th["reg"]
    tb = spec.tile_elems * es
    if tb < th["small_tile"]:
        c *= th["small_f"]
    elif tb > th["big_tile"]:
        c *= th["big_f"]
    elif tb > th["mid_tile"]:
        c *= th["mid_f"]
    u, bs = spec.phase1_units, spec.block_size
    c *= 1 + th["idle"] * ((-(-u // bs)) * bs - u) / max(u, 1)
    waves = spec.n_ctas * max(1, bs // 64)
    c *= max(1.0, th["waves_target"] / waves) ** th["waves_exp"]
    if spec.tiles_per_cta > 1:
        c *= th["tpc"]
    return c


@functools.lru_cache(maxsize=4096)
def _auto_spec_cached(x_shape, perm, dtype, arch, grid_order, nt, tiles_per_cta, block_size):
    return auto_spec(x_shape, perm, dtype, arch, grid_order, nt, tiles_per_cta, block_size)


def auto_spec(x_shape, perm, dtype: str, arch: str, grid_order: Optional[str], nt: str,
              tiles_per_cta: int = 1, block_size: Optional[int] = None) -> "PermuteTiledSpec":
    """Product-route choice: enumerate run / tile-size targets, the three
    stages and (unless given) the three walk orders; keep the valid specs and
    return the lowest :func:`_plan_cost`."""
    shape, cperm = canonicalize_permute(x_shape, perm)
    seen, best = set(), None
    orders = (grid_order,) if grid_order is not None else ("out", "both", "in")
    for rb in (64, 128, 256, 512, 1024, 2048, 4096):
        for mt in (4096, 8192, 16384, 32768, 65536, 131072):
            for stage in ("lds", "gather", "reg"):
                for go in orders:
                    for bs in ((block_size,) if block_size else (None, 256, 512)):
                        try:
                            sp = plan_permute_tiled(x_shape, perm, dtype, arch, run_bytes=rb, max_tile_bytes=mt,
                                                    stage=stage, grid_order=go, nt=nt, tiles_per_cta=tiles_per_cta,
                                                    block_size=bs, swizzle=(0, 0), use_tuned=False, auto=False)
                        except ValueError:
                            continue
                        if sp in seen:
                            continue
                        seen.add(sp)
                        c = _plan_cost(sp)
                        if best is None or c < best[0] - 1e-12:
                            best = (c, sp)
    if best is None:
        raise ValueError(f"no valid permute_tiled config for {list(x_shape)} perm {list(perm)}")
    sp = best[1]
    if sp.stage == "lds" and sp.vec > 1:
        from ...core.arch import ArchTarget

        k, sh = best_swizzle(sp, banks=64 if arch == "gfx950" else 32,
                             wave=ArchTarget.from_gfx(arch).wave_size)
        sp = dataclasses.replace(sp, swizzle_k=k, swizzle_shift=sh)
    return sp


#: Planner defaults (gfx950 design-of-experiments over the hipTensor permute rows,
#: f16 / bf16 / f32 / f64): input and output runs of 512 bytes, tiles up to
#: 32 KiB, consecutive CTAs walking the output order, streaming hints on both
#: sides where the target validates them.
DEFAULT_RUN_BYTES = 512
DEFAULT_MAX_TILE_BYTES = 32768
DEFAULT_GRID_ORDER = "out"

#: Searched configurations served for exact canonical problems:
#: ``{arch: {(elem_bytes, canonical shape, canonical perm): knobs}}``. Only a
#: plan call without knob overrides consults it.
TUNED_CONFIGS: Dict[str, Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Dict[str, object]]] = {
    # hipTensor permute rows (doc and inner-identity controls, ranks 2-6), 2 / 4 / 8-byte
    # elements: search-engine winners confirmed in interleaved timing against the defaults
    "gfx950": {
        (2, (2, 4, 4, 6, 4, 8, 4, 8, 6, 10), (1, 3, 5, 7, 9, 0, 2, 4, 6, 8)): {"tile": (1, 1, 1, 1, 4, 1, 4, 4, 6, 10), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 4, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (4, 8, 6, 12, 8, 10, 8, 14), (1, 3, 5, 7, 0, 2, 4, 6)): {"tile": (1, 1, 1, 1, 8, 5, 8, 14), "block_size": 256, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (36864, 8, 4, 10), (0, 2, 1, 3)): {"tile": (8, 8, 4, 10), "block_size": 256, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (16, 24, 20, 32, 24, 40), (1, 3, 5, 0, 2, 4)): {"tile": (1, 1, 20, 1, 24, 40), "block_size": 64, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 4, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (128, 128, 96, 192), (1, 3, 0, 2)): {"tile": (1, 1, 96, 192), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (320, 224, 40, 72), (1, 3, 0, 2)): {"tile": (8, 1, 40, 72), "block_size": 64, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 5, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (6, 4, 4, 4, 2, 15360), (5, 4, 3, 2, 1, 0)): {"tile": (6, 4, 4, 1, 1, 64), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 5, "grid_order": "in", "nt": "store", "tiles_per_cta": 2},
        (2, (2, 2, 2, 4, 3, 4, 4, 6, 4, 8, 4, 10), (1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10)): {"tile": (1, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 10), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 5, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (2, 4, 4, 8, 4, 6, 4, 8, 6, 10), (1, 3, 5, 7, 9, 0, 2, 4, 6, 8)): {"tile": (1, 1, 2, 1, 4, 1, 4, 4, 6, 10), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 4, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (480, 12, 8, 14, 8, 18), (0, 2, 4, 1, 3, 5)): {"tile": (1, 2, 1, 14, 8, 18), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "both", "nt": "store", "tiles_per_cta": 1},
        (2, (4, 8, 4, 6, 6, 8, 6, 10), (1, 3, 5, 7, 0, 2, 4, 6)): {"tile": (1, 1, 2, 1, 6, 4, 6, 10), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (16, 20, 20, 28, 24, 36), (1, 3, 5, 0, 2, 4)): {"tile": (1, 1, 20, 1, 24, 36), "block_size": 256, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "both", "nt": "store", "tiles_per_cta": 1},
        (2, (4, 4, 4, 3, 2, 2, 15360), (6, 5, 4, 3, 2, 1, 0)): {"tile": (4, 4, 4, 1, 1, 1, 64), "block_size": 64, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (2, 3, 2, 4, 3, 4, 4, 4, 4, 8, 4, 10), (1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10)): {"tile": (1, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 10), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 5, "grid_order": "out", "nt": "store", "tiles_per_cta": 1},
        (2, (4, 8, 6, 10, 8, 12, 8, 16), (1, 3, 5, 7, 0, 2, 4, 6)): {"tile": (1, 1, 1, 1, 8, 4, 8, 16), "block_size": 256, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (98304, 10, 6, 12), (0, 2, 1, 3)): {"tile": (2, 10, 6, 12), "block_size": 256, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "both", "nt": "both", "tiles_per_cta": 1},
        (2, (24576, 8, 6, 10), (0, 2, 1, 3)): {"tile": (6, 8, 6, 10), "block_size": 512, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "both", "tiles_per_cta": 2},
        (2, (46080, 16, 8, 20), (0, 2, 1, 3)): {"tile": (1, 16, 8, 20), "block_size": 256, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (4, 6, 4, 8, 6, 8, 6, 12), (1, 3, 5, 7, 0, 2, 4, 6)): {"tile": (1, 1, 2, 1, 6, 4, 6, 12), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "store", "tiles_per_cta": 1},
        (2, (384, 12, 12, 20), (0, 2, 1, 3)): {"tile": (1, 3, 12, 20), "block_size": 64, "stage": "reg", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 4},
        (2, (320, 40, 40, 224), (1, 3, 0, 2)): {"tile": (1, 1, 40, 112), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "load", "tiles_per_cta": 1},
        (2, (16, 32, 32, 32, 16, 16), (0, 3, 1, 5, 2, 4)): {"tile": (1, 1, 8, 2, 16, 16), "block_size": 64, "stage": "reg", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "store", "tiles_per_cta": 1},
        (2, (16, 32, 32, 32, 16, 16), (4, 3, 2, 1, 0, 5)): {"tile": (8, 8, 1, 8, 8, 16), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 9, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (32, 64, 64, 64, 32), (0, 4, 1, 3, 2)): {"tile": (1, 1, 64, 8, 32), "block_size": 64, "stage": "lds", "swizzle_k": 4, "swizzle_shift": 9, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (2, (32, 64, 64, 64, 32), (3, 2, 1, 0, 4)): {"tile": (32, 1, 1, 32, 32), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 5, "grid_order": "both", "nt": "both", "tiles_per_cta": 1},
        (2, (64, 256, 112, 112), (2, 1, 0, 3)): {"tile": (4, 1, 4, 112), "block_size": 256, "stage": "lds", "swizzle_k": 8, "swizzle_shift": 3, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (64, 256, 12544), (0, 2, 1)): {"tile": (1, 128, 256), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 7, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (2, (512, 1024, 1024), (1, 0, 2)): {"tile": (1, 1, 1024), "block_size": 128, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (8192, 8192), (1, 0)): {"tile": (256, 256), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 8, "grid_order": "in", "nt": "both", "tiles_per_cta": 4},
        (2, (524288, 1024), (1, 0)): {"tile": (64, 128), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (2, (67108864,), (0,)): {"tile": (512,), "block_size": 64, "stage": "reg", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (4, (16, 32, 32, 32, 16, 16), (0, 3, 1, 5, 2, 4)): {"tile": (1, 1, 16, 1, 16, 16), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 6, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (4, (16, 32, 32, 32, 16, 16), (4, 3, 2, 1, 0, 5)): {"tile": (4, 2, 1, 2, 4, 16), "block_size": 256, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 4, "grid_order": "out", "nt": "both", "tiles_per_cta": 2},
        (4, (32, 64, 64, 64, 32), (0, 4, 1, 3, 2)): {"tile": (1, 1, 64, 8, 32), "block_size": 256, "stage": "lds", "swizzle_k": 8, "swizzle_shift": 9, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (4, (32, 64, 64, 64, 32), (3, 2, 1, 0, 4)): {"tile": (2, 1, 1, 4, 32), "block_size": 64, "stage": "reg", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (4, (64, 256, 112, 112), (2, 1, 0, 3)): {"tile": (2, 1, 2, 112), "block_size": 128, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (4, (64, 256, 12544), (0, 2, 1)): {"tile": (1, 128, 128), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 7, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (4, (512, 1024, 1024), (1, 0, 2)): {"tile": (1, 1, 256), "block_size": 64, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (4, (8192, 8192), (1, 0)): {"tile": (128, 256), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 7, "grid_order": "out", "nt": "both", "tiles_per_cta": 4},
        (4, (524288, 1024), (1, 0)): {"tile": (64, 128), "block_size": 64, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 6, "grid_order": "in", "nt": "load", "tiles_per_cta": 1},
        (4, (67108864,), (0,)): {"tile": (256,), "block_size": 64, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (8, (16, 32, 32, 32, 16, 16), (0, 3, 1, 5, 2, 4)): {"tile": (1, 1, 8, 1, 16, 16), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 6, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (8, (16, 32, 32, 32, 16, 16), (4, 3, 2, 1, 0, 5)): {"tile": (4, 2, 1, 2, 8, 16), "block_size": 512, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (8, (32, 64, 64, 64, 32), (0, 4, 1, 3, 2)): {"tile": (1, 1, 64, 2, 32), "block_size": 64, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 7, "grid_order": "out", "nt": "none", "tiles_per_cta": 1},
        (8, (32, 64, 64, 64, 32), (3, 2, 1, 0, 4)): {"tile": (1, 1, 1, 4, 32), "block_size": 64, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
        (8, (64, 256, 112, 112), (2, 1, 0, 3)): {"tile": (2, 1, 2, 112), "block_size": 256, "stage": "reg", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (8, (64, 256, 12544), (0, 2, 1)): {"tile": (1, 64, 128), "block_size": 64, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 6, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (8, (512, 1024, 1024), (1, 0, 2)): {"tile": (1, 1, 128), "block_size": 64, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "out", "nt": "both", "tiles_per_cta": 1},
        (8, (8192, 8192), (1, 0)): {"tile": (128, 128), "block_size": 512, "stage": "lds", "swizzle_k": 16, "swizzle_shift": 7, "grid_order": "out", "nt": "both", "tiles_per_cta": 2},
        (8, (524288, 1024), (1, 0)): {"tile": (64, 128), "block_size": 256, "stage": "gather", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "both", "nt": "both", "tiles_per_cta": 1},
        (8, (67108864,), (0,)): {"tile": (128,), "block_size": 64, "stage": "lds", "swizzle_k": 0, "swizzle_shift": 0, "grid_order": "in", "nt": "both", "tiles_per_cta": 1},
    },
}


def _default_nt(arch: str) -> str:
    from ...core.ir import STREAMING_ARCHS

    return "both" if arch in STREAMING_ARCHS else "none"


def tuned_spec(x_shape, perm, dtype: str, arch: str) -> Optional[PermuteTiledSpec]:
    """The tuned-table spec for this problem on ``arch`` (``None`` if absent or invalid)."""
    shape, cperm = canonicalize_permute(x_shape, perm)
    knobs = TUNED_CONFIGS.get(arch, {}).get((ELEM_BYTES[dtype], tuple(shape), tuple(cperm)))
    if knobs is None:
        return None
    kw = dict(knobs)
    kw["tile"] = tuple(kw["tile"])
    sp = PermuteTiledSpec(x_shape=tuple(shape), perm=tuple(cperm), dtype=dtype, **kw)
    return sp if is_valid_spec(sp, arch)[0] else None


def _grow_runs(shape, perm, tile, dtype, max_steps: int = 64):
    """Grow tile extents until both contiguous runs hold a whole number of 16-byte vectors."""
    n = len(shape)
    v = 16 // ELEM_BYTES[dtype]
    t = list(tile)

    def run(order):
        r, part = 1, None
        for a in order:
            r *= t[a]
            if t[a] < shape[a]:
                part = a
                break
        return r, part

    def nxt(a):
        for d in range(t[a] + 1, shape[a] + 1):
            if shape[a] % d == 0:
                return d
        return t[a]

    for _ in range(max_steps):
        changed = False
        for order in (list(range(n - 1, -1, -1)), [perm[d] for d in range(n - 1, -1, -1)]):
            r, part = run(order)
            if r % v and part is not None:
                t[part] = nxt(part)
                changed = True
        if not changed:
            break
    return tuple(t)


def plan_permute_tiled(
    x_shape: Sequence[int], perm: Sequence[int], dtype: str = "f16", arch: str = "gfx950",
    *, run_bytes: Optional[int] = None, max_tile_bytes: Optional[int] = None,
    block_size: Optional[int] = None, stage: Optional[str] = None, grid_order: Optional[str] = None,
    nt: Optional[str] = None, tiles_per_cta: int = 1, swizzle: Optional[Tuple[int, int]] = None,
    tile: Optional[Sequence[int]] = None, use_tuned: bool = True, auto: bool = True,
) -> PermuteTiledSpec:
    """Canonicalise ``(x_shape, perm)`` and return a valid spec (the product pick).

    The kernel runs on the canonical (axis-merged) problem; the caller passes
    the original contiguous buffers (same flat layout). Without knob overrides
    an entry of :data:`TUNED_CONFIGS` wins; otherwise the defaults above (and
    the overrides) build the spec. Raises ``ValueError`` when no valid
    configuration exists for the knobs given.
    """
    import dataclasses

    from ...core.arch import ArchTarget

    overridden = any(v is not None for v in (run_bytes, max_tile_bytes, block_size, grid_order, nt,
                                             swizzle, tile, stage)) or tiles_per_cta != 1
    if use_tuned and not overridden:
        sp = tuned_spec(x_shape, perm, dtype, arch)
        if sp is not None:
            return sp
    if auto and run_bytes is None and max_tile_bytes is None and tile is None and stage is None \
            and swizzle is None:
        # the product route: cost-model choice of tile and stage
        return _auto_spec_cached(tuple(x_shape), tuple(perm), dtype, arch, grid_order,
                                 _default_nt(arch) if nt is None else nt, tiles_per_cta, block_size)
    tile_given, stage_given = tile, stage
    run_bytes = DEFAULT_RUN_BYTES if run_bytes is None else run_bytes
    max_tile_bytes = DEFAULT_MAX_TILE_BYTES if max_tile_bytes is None else max_tile_bytes
    grid_order = DEFAULT_GRID_ORDER if grid_order is None else grid_order
    nt = _default_nt(arch) if nt is None else nt
    shape, cperm = canonicalize_permute(x_shape, perm)
    if stage is None:
        # 16-byte accesses need vec | the innermost tile extents (lds / reg);
        # odd extents go to the gather stage, which only needs vec | the runs
        probe = PermuteTiledSpec(x_shape=tuple(shape), perm=tuple(cperm),
                                 tile=tuple(tile or default_tile(shape, cperm, dtype, run_bytes,
                                                                 max_tile_bytes)), dtype=dtype)
        stage = "lds"
        if probe.vec * probe.elem_bytes < 16:
            try:
                g = plan_permute_tiled(x_shape, perm, dtype, arch, run_bytes=run_bytes,
                                       max_tile_bytes=max_tile_bytes, block_size=block_size,
                                       stage="gather", grid_order=grid_order, nt=nt,
                                       tiles_per_cta=tiles_per_cta, tile=tile, use_tuned=False)
                if g.vec > probe.vec:
                    return g
            except ValueError:
                pass
    if tile is None:
        tile = default_tile(shape, cperm, dtype, run_bytes, max_tile_bytes)
        if stage == "gather":
            tile = _grow_runs(shape, cperm, tile, dtype)
    spec = PermuteTiledSpec(
        x_shape=tuple(shape), perm=tuple(cperm), tile=tuple(tile), dtype=dtype,
        block_size=256, stage=stage, grid_order=grid_order, nt=nt,
        tiles_per_cta=tiles_per_cta,
    )
    target = ArchTarget.from_gfx(arch)
    if block_size:
        sizes = [block_size]
    else:
        u = spec.phase1_units
        sizes = sorted((256, 128, 512, 64, 1024),
                       key=lambda x: ((-u) % x != 0, (-(-u // x)) * x - u, abs(x - 256)))
    last = "no block size"
    for bs in sizes:
        cand = dataclasses.replace(spec, block_size=bs)
        if swizzle is None and stage == "lds" and cand.vec > 1:
            k, sh = best_swizzle(cand, banks=64 if arch == "gfx950" else 32, wave=target.wave_size)
            cand = dataclasses.replace(cand, swizzle_k=k, swizzle_shift=sh)
        elif swizzle is not None:
            cand = dataclasses.replace(cand, swizzle_k=swizzle[0], swizzle_shift=swizzle[1])
        ok, why = is_valid_spec(cand, arch)
        if ok:
            return cand
        last = why
    if tile_given is None and max_tile_bytes > 1024 and ("per thread" in last or "exceeds" in last):
        # a forced block size holds too many vectors per thread: retry with half the tile
        return plan_permute_tiled(x_shape, perm, dtype, arch, run_bytes=run_bytes, max_tile_bytes=max_tile_bytes // 2,
                                  block_size=block_size, stage=stage_given, grid_order=grid_order, nt=nt,
                                  tiles_per_cta=tiles_per_cta, swizzle=swizzle, use_tuned=False, auto=False)
    raise ValueError(f"no valid permute_tiled config for {list(x_shape)} perm {list(perm)}: {last}")


__all__ = [
    "ELEM_BYTES",
    "MAX_RANK",
    "PermuteTiledSpec",
    "best_swizzle",
    "build_permute_tiled",
    "canonicalize_permute",
    "default_tile",
    "is_valid_spec",
    "lds_bank_cost",
    "permute_tiled_grid",
    "permute_tiled_signature",
    "plan_permute_tiled",
    "tuned_spec",
    "TUNED_CONFIGS",
]
