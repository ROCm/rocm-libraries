# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LayerNorm2D backward kernel instance builder (AOT-generic, runtime shape).

Backward counterpart of :mod:`rocke.instances.common.layernorm2d`. Given the
forward's saved per-row statistics ``mean[m]`` / ``rstd[m]`` (the forward's
``Mean`` / ``InvStd`` outputs) and the upstream gradient ``dY``, computes,
for an ``(M, N)`` activation tensor:

    xhat[m,n] = (X[m,n] - mean[m]) * rstd[m]
    dX[m,n]   = rstd[m] * (dY*gamma)[m,n]
                - a[m] * X[m,n] - b[m]
    dgamma[n] = sum_m dY[m,n] * xhat[m,n]
    dbeta[n]  = sum_m dY[m,n]

with ``a = rstd^3 / N * (sum_n dY*gamma*X - mean * sum_n dY*gamma)`` and
``b = rstd / N * sum_n dY*gamma - a * mean`` -- the same algebra (and the
same ``(dY*gamma, dY*gamma*X)`` row reduction) as MIOpen's
``MIOpenLayerNorm.cpp::layernormbwd``, so the two agree on rounding shape.

The backward is three kernels, the same split MIOpen's
``LayernormBackward`` solver launches, selected by
:attr:`LayerNorm2DBwdSpec.part`:

``"dx"``
    One CTA per row: a runtime-trip-count sweep accumulates
    ``sum(dY*gamma)`` / ``sum(dY*gamma*X)`` per thread, one twin-channel
    block reduction folds them, and a second sweep writes ``dX``. Exactly
    the forward's dynamic shape (:func:`~rocke.helpers.sweep.sweep_row_chunks_dynamic`
    / :func:`~rocke.helpers.sweep.pass2_row_chunks_dynamic` + a
    ``vec``-wide masked scalar tail), so any runtime ``N`` is served.
``"dwdb"``
    Column reduction over rows. Grid ``(ceil(N / (block_size*vec)), P)``:
    each thread owns ``vec`` consecutive columns and walks a contiguous
    ``ceil(M / P)`` row slice, ``unroll`` rows per loop trip (all loads of a
    trip issued before any accumulate, so the trip's loads are in flight
    together). ``P`` (the runtime row-split count) is a kernarg: with
    ``P == 1`` the thread writes ``dgamma`` / ``dbeta`` directly; otherwise
    it writes its f32 partials to the workspace ``WS[P][2][N]``.
``"dwdb_reduce"``
    Folds the ``P`` workspace partials into ``dgamma`` / ``dbeta`` (only
    launched when ``P > 1``). Grid ``ceil(2N / (block_size*vec))``.

All three are AOT-generic: ``M``, ``N`` and ``P`` are runtime kernargs, so
one compiled binary per tuning config serves every shape. ``dwdb`` /
``dwdb_reduce`` vector-load ``vec`` columns at a time, so the dispatcher
must only admit them for ``N % vec == 0``; ``dx`` has a masked tail and
takes any ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from ...helpers.reduction import (
    block_lds_reduce_pair,
    block_lds_reduce_pair_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import pass2_row_chunks_dynamic, sweep_row_chunks_dynamic
from ...helpers.tensor_view import make_global_view, make_lds_view, make_tile_window


DType = Literal["f16", "bf16"]
Part = Literal["dx", "dwdb", "dwdb_reduce"]

PARTS: Tuple[str, ...] = ("dx", "dwdb", "dwdb_reduce")
UNROLLS: Tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class LayerNorm2DBwdSpec:
    """One LayerNorm2D backward kernel (one of the three :data:`PARTS`).

    ``fast_reduce`` (``dx`` only) swaps the barriered LDS-tree twin
    reduction for :func:`~rocke.helpers.reduction.block_lds_reduce_pair_with_wave_prologue`
    (wave-XOR first, one cross-warp barrier). ``unroll`` (``dwdb`` /
    ``dwdb_reduce`` only) is the number of rows (partials) consumed per
    loop trip; ``dx`` ignores it and keeps the default.
    """

    part: Part = "dx"
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    wave_size: int = 64
    fast_reduce: bool = False
    unroll: int = 4
    name: str = "rocke_layernorm2d_bwd"

    @property
    def dynamic(self) -> bool:
        """Always AOT-generic (runtime ``M`` / ``N`` / ``P``); mirrors the
        forward spec's ``dynamic`` field for shape-free artifact caching."""
        return True

    def kernel_name(self) -> str:
        return kernel_name_join(
            f"{self.name}_{self.part}",
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            f"u{self.unroll}" if self.part != "dx" else "",
            flags={"fr": self.fast_reduce},
        )


def is_valid_spec(spec: LayerNorm2DBwdSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one LayerNorm2D backward kernel on ``arch``.

    Shape-free (every part is AOT-generic): only the I/O knob grid, the
    per-arch thread / LDS caps and the per-part knob applicability are
    checked. ``dx`` needs two ``block_size``-word f32 LDS buffers for its
    twin reduction; the column-reduction parts use no LDS.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.part not in PARTS:
        return False, f"unknown part {spec.part!r}; one of {PARTS}"

    ok, why = validate_io(
        IOSpecRule(dtype=spec.dtype, block_size=spec.block_size, vec=spec.vec)
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    if spec.unroll not in UNROLLS:
        return False, f"unroll {spec.unroll} not in {UNROLLS}"

    if spec.part == "dx":
        if spec.unroll != LayerNorm2DBwdSpec.unroll:
            return False, "unroll only applies to the dwdb / dwdb_reduce parts"
        if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
            return False, (
                f"fast_reduce requires block_size {spec.block_size} divisible by "
                f"wave_size {spec.wave_size}"
            )
        bytes_lds = 2 * spec.block_size * 4
        if not target.fits_lds(bytes_lds):
            return False, (
                f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
            )
    elif spec.fast_reduce:
        return False, "fast_reduce only applies to the dx part"

    return True, ""


def build_layernorm2d_bwd(spec: LayerNorm2DBwdSpec) -> KernelDef:
    """Build the IR for one LayerNorm2D backward kernel (see the module
    docstring for each part's signature and grid)."""
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid layernorm2d_bwd spec: {why}")
    if spec.part == "dx":
        return _build_dx(spec)
    if spec.part == "dwdb":
        return _build_dwdb(spec)
    return _build_dwdb_reduce(spec)


def _build_dx(spec: LayerNorm2DBwdSpec) -> KernelDef:
    """``(DY, X, Gamma, Mean, Rstd, DX, M, N)``; grid ``(M, 1, 1)``."""
    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    cap = BS * VEC  # only sizes the (unused for addressing) view shapes

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    DY = b.param("DY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Rstd = b.param("Rstd", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DX = b.param("DX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    _ = b.param("M", I32)  # noqa: F841 - row count; already dynamic via grid size
    cols = b.param("N", I32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    # Issue the per-row statistic loads first: nothing in pass 1 depends
    # on them, so their latency hides behind the sweep.
    mean = load_scalar_as_f32(b, Mean, row, dtype=spec.dtype)
    rstd = load_scalar_as_f32(b, Rstd, row, dtype=spec.dtype)

    dy_view = make_global_view(DY, shape=(1, cap), dtype=io_ty, strides=(cols, 1))
    x_view = make_global_view(X, shape=(1, cap), dtype=io_ty, strides=(cols, 1))
    dx_view = make_global_view(DX, shape=(1, cap), dtype=io_ty, strides=(cols, 1))
    g_view = make_global_view(Gamma, shape=(cap,), dtype=io_ty)
    dy_tile = make_tile_window(dy_view, lengths=(1, 1), origin=(row, b.const_i32(0)))
    x_tile = make_tile_window(x_view, lengths=(1, 1), origin=(row, b.const_i32(0)))
    dx_tile = make_tile_window(dx_view, lengths=(1, 1), origin=(row, b.const_i32(0)))

    lds_a = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_dyg").base
    lds_c = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_dygx").base

    # --- Pass 1: sum(dY*gamma), sum(dY*gamma*X) ---------------------------
    def pass1_body(n_off, dy_scalars, iter_vars):
        s_dyg, s_dygx = iter_vars
        x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        dyg = [b.fmul(dy_scalars[i], gv[i]) for i in range(VEC)]
        dygx = [b.fmul(dyg[i], x_scalars[i]) for i in range(VEC)]
        s_dyg = b.fadd(s_dyg, tree_reduce(b, b.fadd, dyg))
        s_dygx = b.fadd(s_dygx, tree_reduce(b, b.fadd, dygx))
        return [s_dyg, s_dygx]

    (s_dyg, s_dygx), main_elems, _tail = sweep_row_chunks_dynamic(
        b,
        dy_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        length=cols,
        iter_args=[("s_dyg", b.const_f32(0.0)), ("s_dygx", b.const_f32(0.0))],
        body=pass1_body,
    )

    # Masked scalar tail: clamp the index in-bounds, zero the masked
    # lane's dY*gamma (so its dY*gamma*X term is zero too).
    for tc in range(VEC):
        idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
        valid = b.cmp_lt(idx_col, cols)
        idx_safe = b.smin(idx_col, b.sub(cols, b.const_i32(1)))
        flat = b.add(b.mul(row, cols), idx_safe)
        dy_raw = load_scalar_as_f32(b, DY, flat, dtype=spec.dtype)
        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
        g_raw = load_scalar_as_f32(b, Gamma, idx_safe, dtype=spec.dtype)
        dyg = b.select(valid, b.fmul(dy_raw, g_raw), b.const_f32(0.0))
        s_dyg = b.fadd(s_dyg, dyg)
        s_dygx = b.fadd(s_dygx, b.fmul(dyg, x_raw))

    if spec.fast_reduce:
        t_dyg, t_dygx = block_lds_reduce_pair_with_wave_prologue(
            b, s_dyg, s_dygx, lds_a, lds_c, tid, block_size=BS,
            combine_a="sum", combine_c="sum", wave_size=spec.wave_size,
        )
    else:
        t_dyg, t_dygx = block_lds_reduce_pair(
            b, s_dyg, s_dygx, lds_a, lds_c, tid, block_size=BS,
            combine_a="sum", combine_c="sum",
        )

    inv_n = b.fdiv(b.const_f32(1.0), b.sitofp_f32(cols))
    rstd3 = b.fmul(b.fmul(rstd, rstd), rstd)
    coef_a = b.fmul(b.fmul(rstd3, inv_n), b.fsub(t_dygx, b.fmul(t_dyg, mean)))
    coef_b = b.fsub(b.fmul(b.fmul(rstd, t_dyg), inv_n), b.fmul(coef_a, mean))

    def dx_of(dy, g, x):
        return b.fsub(b.fsub(b.fmul(rstd, b.fmul(dy, g)), b.fmul(coef_a, x)), coef_b)

    # --- Pass 2: dX -------------------------------------------------------
    def pass2_body(n_off):
        dys = dy_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        xs = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        return [dx_of(dys[i], gv[i], xs[i]) for i in range(VEC)]

    trip_count = b.div(cols, b.const_i32(BS * VEC))
    pass2_row_chunks_dynamic(
        b,
        dx_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        main_elems_per_row=main_elems,
        trip_count=trip_count,
        body=pass2_body,
    )

    for tc in range(VEC):
        idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
        valid = b.cmp_lt(idx_col, cols)
        with b.scf_if(valid):
            flat = b.add(b.mul(row, cols), idx_col)
            dy_raw = load_scalar_as_f32(b, DY, flat, dtype=spec.dtype)
            x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
            g_raw = load_scalar_as_f32(b, Gamma, idx_col, dtype=spec.dtype)
            store_scalar_from_f32(
                b, DX, flat, dx_of(dy_raw, g_raw, x_raw), dtype=spec.dtype
            )

    return b.kernel


def _dwdb_accumulate(b: IRBuilder, accs, dys, xs, mean, rstd, vec: int):
    """Fold one row's ``vec`` columns into the ``[dw..., db...]`` accumulators."""
    out = list(accs)
    for i in range(vec):
        xhat = b.fmul(b.fsub(xs[i], mean), rstd)
        out[i] = b.fma(dys[i], xhat, out[i])
        out[vec + i] = b.fadd(out[vec + i], dys[i])
    return out


def _build_dwdb(spec: LayerNorm2DBwdSpec) -> KernelDef:
    """``(DY, X, Mean, Rstd, DW, DB, WS, M, N, P)``; grid
    ``(ceil(N / (block_size*vec)), P, 1)``."""
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, U = spec.block_size, spec.vec, spec.unroll

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    DY = b.param("DY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Rstd = b.param("Rstd", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DW = b.param("DW", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    DB = b.param("DB", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    WS = b.param("WS", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    rows = b.param("M", I32)
    cols = b.param("N", I32)
    splits = b.param("P", I32)

    tid = b.thread_id_x()
    bx = b.block_id_x()
    by = b.block_id_y()

    c_one = b.const_i32(1)
    c0 = b.mul(b.add(b.mul(bx, b.const_i32(BS)), tid), b.const_i32(VEC))
    valid = b.cmp_lt(c0, cols)
    c_safe = b.select(valid, c0, b.const_i32(0))

    # This CTA's contiguous row slice [r0, r1).
    rows_per_split = b.div(b.add(rows, b.sub(splits, c_one)), splits)
    r0 = b.mul(by, rows_per_split)
    r1 = b.smin(b.add(r0, rows_per_split), rows)
    n_rows = b.smax(b.sub(r1, r0), b.const_i32(0))
    trip = b.div(n_rows, b.const_i32(U))

    def load_row(r):
        flat = b.add(b.mul(r, cols), c_safe)
        dys = load_vec_as_f32(b, DY, flat, dtype=spec.dtype, n=VEC)
        xs = load_vec_as_f32(b, X, flat, dtype=spec.dtype, n=VEC)
        mean = load_scalar_as_f32(b, Mean, r, dtype=spec.dtype)
        rstd = load_scalar_as_f32(b, Rstd, r, dtype=spec.dtype)
        return dys, xs, mean, rstd

    iter_args = [(f"dw{i}", b.const_f32(0.0)) for i in range(VEC)] + [
        (f"db{i}", b.const_f32(0.0)) for i in range(VEC)
    ]
    main = b.scf_for_iter(b.const_i32(0), trip, c_one, iter_args, iv_name="rk")
    with main as (k, iter_vars):
        r_base = b.add(r0, b.mul(k, b.const_i32(U)))
        loaded = [load_row(b.add(r_base, b.const_i32(u))) for u in range(U)]
        accs = list(iter_vars)
        for dys, xs, mean, rstd in loaded:
            accs = _dwdb_accumulate(b, accs, dys, xs, mean, rstd, VEC)
        b.scf_yield(*accs)

    rem_lo = b.add(r0, b.mul(trip, b.const_i32(U)))
    rem_args = [(f"rdw{i}", main.results[i]) for i in range(VEC)] + [
        (f"rdb{i}", main.results[VEC + i]) for i in range(VEC)
    ]
    rem = b.scf_for_iter(rem_lo, r1, c_one, rem_args, iv_name="rr")
    with rem as (r, iter_vars):
        dys, xs, mean, rstd = load_row(r)
        b.scf_yield(*_dwdb_accumulate(b, list(iter_vars), dys, xs, mean, rstd, VEC))

    dw = list(rem.results[:VEC])
    db = list(rem.results[VEC:])
    with b.scf_if(valid):
        with b.scf_if_else(b.cmp_eq(splits, c_one)) as (then_ctx, else_ctx):
            with then_ctx:
                store_vec(b, DW, c0, pack_f32_to(b, dw, dtype=spec.dtype), n=VEC)
                store_vec(b, DB, c0, pack_f32_to(b, db, dtype=spec.dtype), n=VEC)
            with else_ctx:
                ws_base = b.mul(b.mul(by, b.const_i32(2)), cols)
                b.global_store_vN(WS, b.add(ws_base, c0), b.vec_pack(dw, F32), VEC)
                b.global_store_vN(
                    WS, b.add(b.add(ws_base, cols), c0), b.vec_pack(db, F32), VEC
                )

    return b.kernel


def _build_dwdb_reduce(spec: LayerNorm2DBwdSpec) -> KernelDef:
    """``(WS, DW, DB, N, P)``; grid ``(ceil(2N / (block_size*vec)), 1, 1)``."""
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, U = spec.block_size, spec.vec, spec.unroll

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    WS = b.param("WS", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    DW = b.param("DW", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    DB = b.param("DB", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    cols = b.param("N", I32)
    splits = b.param("P", I32)

    tid = b.thread_id_x()
    bx = b.block_id_x()

    c_one = b.const_i32(1)
    width = b.mul(cols, b.const_i32(2))  # one partial = [dw(N), db(N)]
    e0 = b.mul(b.add(b.mul(bx, b.const_i32(BS)), tid), b.const_i32(VEC))
    valid = b.cmp_lt(e0, width)
    e_safe = b.select(valid, e0, b.const_i32(0))
    trip = b.div(splits, b.const_i32(U))

    def load_partial(p):
        v = b.global_load_vN(WS, b.add(b.mul(p, width), e_safe), F32, VEC)
        return [b.vec_extract(v, i) for i in range(VEC)]

    def accumulate(accs, vals):
        return [b.fadd(accs[i], vals[i]) for i in range(VEC)]

    iter_args = [(f"s{i}", b.const_f32(0.0)) for i in range(VEC)]
    main = b.scf_for_iter(b.const_i32(0), trip, c_one, iter_args, iv_name="pk")
    with main as (k, iter_vars):
        p_base = b.mul(k, b.const_i32(U))
        loaded = [load_partial(b.add(p_base, b.const_i32(u))) for u in range(U)]
        accs = list(iter_vars)
        for vals in loaded:
            accs = accumulate(accs, vals)
        b.scf_yield(*accs)

    rem_lo = b.mul(trip, b.const_i32(U))
    rem_args = [(f"rs{i}", main.results[i]) for i in range(VEC)]
    rem = b.scf_for_iter(rem_lo, splits, c_one, rem_args, iv_name="pr")
    with rem as (p, iter_vars):
        b.scf_yield(*accumulate(list(iter_vars), load_partial(p)))

    total = list(rem.results)
    with b.scf_if(valid):
        with b.scf_if_else(b.cmp_lt(e0, cols)) as (then_ctx, else_ctx):
            with then_ctx:
                store_vec(b, DW, e0, pack_f32_to(b, total, dtype=spec.dtype), n=VEC)
            with else_ctx:
                store_vec(
                    b, DB, b.sub(e0, cols), pack_f32_to(b, total, dtype=spec.dtype),
                    n=VEC,
                )

    return b.kernel


# --------------------------------------------------------------------- #
# Host-side launch geometry
# --------------------------------------------------------------------- #


def layernorm2d_bwd_grid(
    spec: LayerNorm2DBwdSpec, *, rows: int, cols: int, splits: int = 1
) -> Tuple[int, int, int]:
    """Launch grid of ``spec.part`` for an ``(rows, cols)`` problem."""
    per_cta = spec.block_size * spec.vec
    if spec.part == "dx":
        return (int(rows), 1, 1)
    if spec.part == "dwdb":
        return (-(-int(cols) // per_cta), int(splits), 1)
    return (-(-2 * int(cols) // per_cta), 1, 1)


def layernorm2d_bwd_workspace_bytes(*, cols: int, splits: int) -> int:
    """Bytes of the f32 ``WS[P][2][N]`` partials buffer (0 when ``P == 1``)."""
    return 0 if splits <= 1 else int(splits) * 2 * int(cols) * 4


def default_row_splits(
    *, rows: int, cols: int, block_size: int, vec: int, num_cus: int
) -> int:
    """Default runtime ``P`` for the ``dwdb`` part.

    Enough CTAs to give every CU about four (MIOpen's
    ``get_reqd_work_item_cnt`` target), but no more splits than
    ``sqrt(rows)`` (MIOpen's ``get_parallelism_size`` cap), so each CTA
    still walks a row slice long enough to amortise its partial write and
    the workspace stays small next to ``dY`` / ``X``. Powers of two only.
    """
    cols_ctas = max(1, -(-int(cols) // (block_size * vec)))
    target = max(1, 4 * int(num_cus) // cols_ctas)
    splits = 1
    while splits * 2 <= target and (splits * 2) ** 2 <= rows:
        splits *= 2
    return splits


def layernorm2d_bwd_signature(spec: LayerNorm2DBwdSpec):
    sb = SignatureBuilder()
    if spec.part == "dx":
        for name in ("DY", "X", "Gamma", "Mean", "Rstd", "DX"):
            sb.ptr(name, spec.dtype)
        return sb.scalar("M", "i32").scalar("N", "i32").build()
    if spec.part == "dwdb":
        for name in ("DY", "X", "Mean", "Rstd", "DW", "DB"):
            sb.ptr(name, spec.dtype)
        sb.ptr("WS", "f32")
        return sb.scalar("M", "i32").scalar("N", "i32").scalar("P", "i32").build()
    sb.ptr("WS", "f32").ptr("DW", spec.dtype).ptr("DB", spec.dtype)
    return sb.scalar("N", "i32").scalar("P", "i32").build()
