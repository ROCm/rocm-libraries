# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BatchNorm2D forward kernel instance builder (own family, NCHW).

For an ``(N, C, H, W)`` activation tensor, computes, per channel ``c``:

    mean[c]       = sum_{n,h,w}(X[n,c,h,w]) / (N*H*W)
    var[c]        = sum_{n,h,w}((X[n,c,h,w] - mean[c])^2) / (N*H*W)  (Welford)
    inv_std[c]    = 1 / sqrt(var[c] + eps)
    Y[n,c,h,w]    = (X[n,c,h,w] - mean[c]) * inv_std[c] * gamma[c] + beta[c]

``training=True`` computes ``mean``/``var`` from the batch (as above) and
updates ``RunningMean``/``RunningVar`` with the usual EMA:

    running = (1 - momentum) * running + momentum * batch_stat

``training=False`` skips the reduction entirely and normalises with the
``RunningMean``/``RunningVar`` the caller supplies -- the inference path is a
pure elementwise stream, no LDS, no sync.

Why this is its own dispatch family and not a third ``kind`` in
``rocke.dispatch.families.norm``: ``norm.py``'s ``layernorm2d`` / ``rmsnorm2d``
reduce *within one row* (one CTA per row, independent rows). BatchNorm reduces
*across* ``N*H*W`` *per channel* -- the natural unit of work is "one CTA per
channel", a different grid shape and a different dim vocabulary
(``n,c,h,w`` vs ``rows,cols``). Grafting that onto ``norm.py`` would mean
diverging its ``dims()``/``_grid()``/candidate shape for one kind, which is
sharing the file, not the abstraction.

Layout: **NCHW**, not NHWC (the convention conv/pooling use in this repo).
With one CTA per channel sweeping all of that channel's ``N*H*W`` elements,
NCHW puts each ``(n, c)`` slice's ``H*W`` elements contiguously in memory --
exactly what :func:`rocke.helpers.sweep.sweep_row_chunks` wants for coalesced
vec-wide loads. In NHWC the same elements are strided by ``C``, which would
force scalar/gather loads for a single-channel CTA.

Reduction strategy -- an N-loop of per-row sweeps, no new reduction helper.
There is no cross-CTA reduction primitive in this codebase (by design: every
existing block reduction is intra-CTA), so the "one CTA owns all of this
channel's data" design avoids ever needing one. In NCHW, channel ``c``'s data
for sample ``n`` is the contiguous row ``n*C + c`` of a packed ``(N*C, H*W)``
view (see :func:`rocke.helpers.tensor_view.make_naive_tensor_view_packed`).
The kernel therefore does, at Python *build* time (``N`` is a small compile-time
count -- a typical batch size, not a loop the hardware runs):

    for n in range(N):
        sweep the (n, c) row's H*W elements, accumulating into
        the SAME per-thread sum_p / sumsq_p as every other n

and only reduces once, across the workgroup, after the N-loop -- exactly
:func:`rocke.helpers.reduction.welford_block_reduce_stable`, same as
``layernorm2d``, just fed a sum of ``N`` rows' partials instead of one row's.
No new function is needed in ``helpers/reduction.py`` or ``helpers/sweep.py``:
``sweep_row_chunks`` already sweeps one row given a :class:`TileWindow`; this
file just builds ``N`` of those windows (one per sample) and calls it ``N``
times into shared accumulators.

Performance shape:
  - One CTA per channel, ``block_size`` threads
  - Each thread's per-channel reduction footprint is
    ``elems_per_thread = (N*H*W) // block_size``; the single-pass (cached)
    path applies the same VGPR cap as ``layernorm2d``
    (:data:`rocke.helpers.reduction.REGISTER_TILE_MAX_ELEMS_PER_THREAD`) to
    that *aggregate* count, falling back to a streaming two-pass kernel
    (re-reads X from HBM) past it
  - Gamma / Beta / the inference-mode running mean / var are per-channel
    *scalars* -- constant for the whole CTA -- so they are loaded exactly
    once per kernel invocation, not once per swept chunk (unlike
    ``layernorm2d``'s per-column Gamma/Beta, which vary along the row)
  - RunningMean / RunningVar are always f32 buffers (the standard choice for
    accumulated statistics, independent of the storage dtype of X/Y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    row_norm_needs_two_pass,
    tree_reduce,
    welford_block_reduce_stable,
    welford_block_reduce_stable_wave_prologue,
)
from ...helpers.spec import IOSpecRule, kernel_name_join, validate_io
from ...helpers.sweep import (
    pass2_row_chunks,
    pass2_row_chunks_dynamic,
    sweep_row_chunks,
    sweep_row_chunks_dynamic,
)
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class BatchNorm2DSpec:
    """One BatchNorm2D forward instance."""

    n: int
    c: int
    h: int
    w: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    training: bool = True
    save_mean_invstd: bool = False
    channels_per_block: int = 1
    """Default-off lever: how many channels one CTA reduces, run sequentially
    (not split across threads). ``=1`` (the default) reproduces the original
    one-CTA-per-channel emission byte-for-byte. ``>1`` trades a bigger CTA
    body for fewer, bigger CTAs -- the fix for shapes whose grid (``C``
    CTAs) is small enough that launch overhead dominates total runtime (see
    the module docstring's "Grid lever" section for the measurements behind
    this knob)."""
    n_groups: int = 1
    """Default-off lever: split one CTA's ``block_size`` threads into
    ``n_groups`` sub-groups of ``block_size // n_groups`` threads each,
    where each sub-group sweeps a *different* ``n`` sample's ``H*W`` row
    concurrently instead of every thread sweeping the same row one sample
    at a time. ``=1`` (the default) reproduces the original per-sample
    N-loop emission byte-for-byte. ``>1`` mirrors MIOpen's spatial-batchnorm
    thread mapping (threads split across both the spatial and batch
    dimensions, not just spatial) -- the fix for shapes whose ``H*W`` is
    small enough that it structurally caps ``block_size`` well below what
    the hardware can usefully run.

    Supports both ``training=False`` (pure elementwise read of the given
    running stats, no reduction) and ``training=True``: each thread
    accumulates its own Welford partial over its assigned slice of
    elements, then merges with the same ``welford_block_reduce_stable``/
    ``_wave_prologue`` call the ``n_groups == 1`` path uses (that merge
    only needs each of the ``block_size`` threads to hand in a partial, it
    does not care how the slice was chosen). The ``training=True`` path
    never register-caches ``X`` across its two passes (unlike
    ``n_groups == 1``'s cached fast path) -- see ``build_batchnorm2d``'s
    matching comment for why that scope cut doesn't cost much for the
    small-``H*W`` shapes this lever targets.

    ``H*W`` need not divide ``hw_threads = block_size // n_groups`` exactly
    -- a masked scalar tail (the same clamp+select/``scf_if`` idiom the
    ``n_groups == 1`` path uses for its own tail) covers the remainder, so
    this lever reaches shapes too small for any power-of-two ``hw_threads``
    to tile exactly (e.g. 7x7=49), not just shapes that happen to divide
    evenly."""
    dynamic: bool = False
    """Default-off lever: build the AOT-generic variant instead. ``n/c/h/w``
    become real runtime kernargs (read and used, not ``noqa: F841``-dead)
    and the per-sample sweep becomes a genuine runtime-trip-count loop
    (``scf.for``) rather than a Python-unrolled ``for n in range(N)``, so
    one compiled binary serves any runtime ``N/H/W`` (``C`` only ever
    mattered for addressing and grid sizing, both already runtime-safe).
    Requires ``channels_per_block == 1`` and ``n_groups == 1`` -- the
    masked/multi-group-tail machinery those levers would need to stay
    shape-generic is a follow-up, not implemented here (see
    :func:`is_valid_spec`). See :func:`_build_batchnorm2d_dynamic`."""
    fast_reduce: bool = False
    """Default-off lever, training-only: replace the flat ``block_size``-wide
    LDS Welford tree (:func:`welford_block_reduce_stable`, 9 barriers / 3 LDS
    buffers for ``block_size=256``) with a wave-shuffle-first reduction
    (:func:`welford_block_reduce_stable_wave_prologue`): a butterfly-shuffle
    merge inside each wavefront (no LDS, no barrier) followed by a small
    cross-warp LDS merge (``num_warps`` slots, 1 barrier). Mirrors MIOpen's
    ``shfl_interleaved_reduction_welford`` + ``lds_reduce2_welford`` spatial-
    batchnorm reduction shape. ``=False`` (the default) reproduces the
    original flat-tree emission byte-for-byte. Rejected when ``block_size``
    is not a multiple of ``wave_size`` (see ``is_valid_spec``)."""
    wave_size: int = 64
    name: str = "rocke_batchnorm2d_fwd"

    @property
    def hw(self) -> int:
        return self.h * self.w

    @property
    def n_per_channel(self) -> int:
        """Total elements one CTA reduces: every sample's H*W pixels."""
        return self.n * self.h * self.w

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_channel // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.n}C{self.c}H{self.h}W{self.w}",
            f"b{self.block_size}",
            f"v{self.vec}",
            f"cpb{self.channels_per_block}" if self.channels_per_block != 1 else "",
            f"ng{self.n_groups}" if self.n_groups != 1 else "",
            flags={
                "train": self.training,
                "smv": self.save_mean_invstd,
                "fr": self.fast_reduce,
                "dyn": self.dynamic,
            },
        )


def is_valid_spec(spec: BatchNorm2DSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one BatchNorm2D config on ``arch``.

    Two distinct divisibility facts matter here, unlike the single-row
    ``layernorm2d``/``rmsnorm2d`` family:

    * ``H*W`` (one sample's per-channel pixel count -- the length of ONE
      ``sweep_row_chunks`` call) must be divisible by ``block_size * vec``,
      since that is the sweep ``sweep_row_chunks`` itself requires.
    * ``elems_per_thread = (N*H*W) // block_size`` (the AGGREGATE per-thread
      count across all ``N`` sweeps) is what the VGPR cache-budget cap
      (:data:`rocke.helpers.reduction.REGISTER_TILE_MAX_ELEMS_PER_THREAD`)
      bounds -- past it the kernel streams two-pass instead of caching.

    Training mode additionally needs the three f32 Welford LDS buffers;
    inference mode does no reduction at all, so it skips that budget check
    entirely (and is valid at LDS pressures that would reject training).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.n <= 0 or spec.c <= 0 or spec.h <= 0 or spec.w <= 0:
        return False, "n, c, h, w must all be positive"

    if spec.save_mean_invstd and not spec.training:
        return False, "save_mean_invstd requires training=True (inference has no batch mean/var to save)"

    if spec.channels_per_block <= 0:
        return False, "channels_per_block must be positive"
    if spec.c % spec.channels_per_block != 0:
        return False, (
            f"c {spec.c} not divisible by channels_per_block "
            f"{spec.channels_per_block}"
        )

    if spec.n_groups <= 0:
        return False, "n_groups must be positive"
    if spec.block_size % spec.n_groups != 0:
        return False, (
            f"block_size {spec.block_size} not divisible by n_groups "
            f"{spec.n_groups}"
        )
    if spec.n % spec.n_groups != 0:
        return False, f"n {spec.n} not divisible by n_groups {spec.n_groups}"

    if spec.dynamic and (spec.channels_per_block != 1 or spec.n_groups != 1):
        return False, (
            "dynamic=True requires channels_per_block == 1 and n_groups == 1 "
            "(their masked-tail/multi-group machinery is not yet shape-generic)"
        )

    if spec.fast_reduce and not spec.training:
        return False, "fast_reduce requires training=True (inference has no reduction to speed up)"
    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, (
            f"fast_reduce requires block_size {spec.block_size} divisible by "
            f"wave_size {spec.wave_size}"
        )

    # ``block_size`` is the real launch parameter (one CTA's thread count),
    # so it alone -- not ``hw_threads`` -- is what must come from the
    # hardware-meaningful ``allowed_block_sizes`` set; dtype/vec are
    # independent of ``n_groups``.
    ok, why = validate_io(
        IOSpecRule(dtype=spec.dtype, block_size=spec.block_size, vec=spec.vec)
    )
    if not ok:
        return False, why

    # ``n_groups == 1``'s row sweep (the common case) handles an ``HW`` not
    # divisible by ``block_size * vec`` with an in-kernel scalar tail (see
    # ``build_batchnorm2d``'s ``tail_chunks`` block) -- no extra check
    # needed here. ``n_groups > 1`` (the spatial-split lever) reuses the
    # SAME masked-tail idiom scoped to ``hw_threads = block_size //
    # n_groups`` instead of ``block_size`` (see ``build_batchnorm2d``'s
    # matching ``tail_chunks_ng`` block), so ``hw_threads`` is a
    # kernel-internal sub-group thread count, not a real launch parameter
    # -- it does not need to be in ``allowed_block_sizes``, and ``HW``
    # does not need to divide it exactly either.

    if not spec.dynamic:
        # The dynamic path always streams X twice (no per-row/per-sample
        # register cache -- a runtime trip count can't back one sized at
        # build time), so it carries no VGPR cap and ``n/h/w`` are capacity
        # placeholders, not the real shape, so their exact-divisibility
        # relationship to ``elems_per_thread`` doesn't apply either.
        cap = (
            None
            if row_norm_needs_two_pass(spec.elems_per_thread)
            else REGISTER_TILE_MAX_ELEMS_PER_THREAD
        )
        if cap is not None and spec.elems_per_thread > cap:
            return False, (
                f"elems_per_thread {spec.elems_per_thread} > {cap}; "
                "pick a larger block_size or a multi-pass kernel"
            )

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    if spec.training:
        # Three f32 Welford reduction buffers (mean + M2 + count),
        # ``block_size`` words each. Inference does no reduction, so it
        # allocates none of this and is exempt from the budget check.
        bytes_lds = 3 * spec.block_size * 4
        if not target.fits_lds(bytes_lds):
            return False, (
                f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
            )

    return True, ""


def build_batchnorm2d(spec: BatchNorm2DSpec) -> KernelDef:
    """Build the IR for one BatchNorm2D forward instance.

    Kernel signature:
      ``(X: ptr, Gamma: ptr, Beta: ptr, Y: ptr,
         RunningMean: ptr<f32>, RunningVar: ptr<f32>,
         [Mean: ptr, InvStd: ptr,]
         N: i32, C: i32, H: i32, W: i32, eps: f32, momentum: f32)``

    Grid layout: ``grid_x = C`` (one CTA per channel), ``block = (block_size, 1, 1)``.
    ``N``/``C``/``H``/``W`` are accepted as runtime params for ABI/manifest
    symmetry with the rest of the DSL's conventions (mirroring ``layernorm2d``'s
    unused ``M``), but the loop bounds baked into the IR are the compile-time
    ``spec.n`` / ``spec.h`` / ``spec.w`` -- the kernel is specialised to one
    shape, same as every other small-op instance in this repo.

    ``spec.dynamic=True`` builds the AOT-generic variant instead (see
    :func:`_build_batchnorm2d_dynamic`): one compiled binary serves any
    runtime ``N/H/W`` (``channels_per_block``/``n_groups`` must both be 1).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d spec: {why}")

    if spec.dynamic:
        return _build_batchnorm2d_dynamic(spec)

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    N, C, HW = spec.n, spec.c, spec.hw
    two_pass = row_norm_needs_two_pass(spec.elems_per_thread)

    # NG==1 row sweep shape: the vectorized ``sweep_row_chunks``/
    # ``pass2_row_chunks`` main loop only covers whole ``block_size*vec``
    # chunks. When ``HW`` is not evenly divisible by that (common real-model
    # shapes: 56x56=3136, 24x24=576, 14x14=196), the remaining
    # ``tail_per_row`` elements (``0 <= tail_per_row < BS*VEC``) are handled
    # by a scalar tail below -- ``tail_chunks`` scalar (vec=1) passes of
    # ``BS`` threads each, the last one lane-masked via ``cmp_lt``. This is
    # a strict Python-build-time superset: when ``HW % (BS*VEC) == 0``,
    # ``tail_chunks == 0`` and every code path below collapses back to the
    # original byte-identical emission (see ``is_valid_spec``'s matching
    # NG==1 divisibility relaxation).
    chunks_main = HW // (BS * VEC)
    ept_row_main = chunks_main * VEC  # per-row vectorized elems/thread
    main_elems_per_row = ept_row_main * BS
    tail_per_row = HW - main_elems_per_row
    tail_chunks = (tail_per_row + BS - 1) // BS if tail_per_row else 0
    ept_row = ept_row_main  # kept as the historical name used below

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Beta = b.param(
        "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    RunningMean = b.param("RunningMean", PtrType(F32, "global"), noalias=True)
    RunningVar = b.param("RunningVar", PtrType(F32, "global"), noalias=True)
    if spec.save_mean_invstd:
        Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True)
        InvStd = b.param(
            "InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    _ = b.param("N", I32)  # noqa: F841 - ABI symmetry; baked as spec.n
    _ = b.param("C", I32)  # noqa: F841 - ABI symmetry; baked as spec.c
    _ = b.param("H", I32)  # noqa: F841 - ABI symmetry; baked as spec.h
    _ = b.param("W", I32)  # noqa: F841 - ABI symmetry; baked as spec.w
    eps = b.param("eps", F32)
    momentum = b.param("momentum", F32)

    tid = b.thread_id_x()
    base_channel = b.block_id_x()

    # Packed (N*C, H*W) view: sample n / channel c's H*W pixels are the
    # contiguous row (n*C + c). See the module docstring for why this NCHW
    # row-per-(n,c) layout is what lets one CTA per channel use the ordinary
    # row-sweep helpers instead of a gather.
    x_view = make_naive_tensor_view_packed(X, shape=(N * C, HW), dtype=io_ty)
    y_view = make_naive_tensor_view_packed(Y, shape=(N * C, HW), dtype=io_ty)

    # LDS scratch for the Welford block merge (training only -- see
    # is_valid_spec's LDS-budget note). Allocated once; every channel this
    # CTA owns (see ``channels_per_block`` below) reuses the same buffer,
    # one channel fully resolved at a time.
    lds_mean = lds_m2 = lds_count = None
    if spec.training:
        lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
        lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
        lds_count = make_lds_view(
            b, dtype=F32, shape=(BS,), name_hint="lds_count"
        ).base

    CPB = spec.channels_per_block
    NG = spec.n_groups

    # ``channels_per_block`` (default 1, byte-identical to the original
    # single-channel-per-CTA emission): a Python-build-time loop over the
    # CPB channels this CTA owns, run sequentially with the full block of
    # threads on each one in turn -- not a thread-split, so none of the
    # per-channel reduction/affine logic below changes shape. Only the grid
    # (fewer, bigger CTAs) and this outer loop are new. Channels after the
    # first reuse the same lds_mean/lds_m2/lds_count buffers the Welford
    # merge above just finished reading every lane out of (that merge's own
    # trailing ``b.sync()`` makes the final LDS *values* visible, but a
    # lagging lane could still be mid-read of the *old* contents when a
    # fast lane starts the next channel's first LDS *write* -- the extra
    # sync below is what actually orders "every lane done reading channel
    # cb's LDS" before "any lane starts writing channel cb+1's LDS").
    for cb in range(CPB):
        if CPB == 1:
            channel = base_channel
        else:
            channel = b.add(b.mul(base_channel, b.const_i32(CPB)), b.const_i32(cb))
            if spec.training and cb > 0:
                b.sync()

        if NG == 1:
            # --- Pass 1 (training only): N-loop of per-sample row sweeps --
            #
            # Each sweep_row_chunks call reduces one (n, c) row's H*W
            # elements into the SAME sum_p / sumsq_p accumulators every
            # other n also feeds, so after the loop they hold the sum /
            # sum-of-squares over all N*H*W elements this channel owns --
            # exactly what welford_block_reduce_stable below expects, with
            # count = spec.elems_per_thread (N*H*W/block_size).
            sum_p = b.const_f32(0.0)
            sumsq_p = b.const_f32(0.0)
            tail_count_p = b.const_f32(0.0) if tail_chunks else None

            def pass1_body(_n_off, x_scalars):
                nonlocal sum_p, sumsq_p
                sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
                sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
                sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))

            row_values: List = []
            cached_by_n: List[List] = []
            x_tiles_by_n: List = []
            for n in range(N):
                row_n = b.add(b.const_i32(n * C), channel)
                row_values.append(row_n)
                x_tile_n = make_tile_window(
                    x_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                )
                x_tiles_by_n.append(x_tile_n)
                if spec.training:
                    res = sweep_row_chunks(
                        b,
                        x_tile_n,
                        tid=tid,
                        block_size=BS,
                        vec=VEC,
                        elems_per_thread=ept_row,
                        body=pass1_body,
                        cache=not two_pass,
                    )
                    cached_by_n.append(res.cached)

                    # Scalar tail (see the ``tail_chunks`` note above
                    # ``build_batchnorm2d``'s start): branch-free, masked via
                    # ``select`` rather than an ``scf_if`` guard, since every
                    # lane's contribution to a sum/sumsq/count reduction is
                    # naturally zero-able -- unlike pass 2's store, which
                    # cannot mask (see the pass-2 tail below). The load index
                    # is clamped with ``smin`` to stay inside this row's
                    # ``HW`` span (not just inside the tensor) before the
                    # mask is applied to its *value*, so an out-of-range lane
                    # still issues an in-bounds load, just a discarded one.
                    for tc in range(tail_chunks):
                        idx_col = b.add(
                            b.const_i32(main_elems_per_row + tc * BS), tid
                        )
                        valid = b.cmp_lt(idx_col, b.const_i32(HW))
                        idx_safe = b.smin(idx_col, b.const_i32(HW - 1))
                        flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_safe)
                        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                        x_masked = b.select(valid, x_raw, b.const_f32(0.0))
                        valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                        sum_p = b.fadd(sum_p, x_masked)
                        sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
                        tail_count_p = b.fadd(tail_count_p, valid_f)

            if spec.training:
                if tail_chunks:
                    count_val = b.fadd(
                        b.const_f32(float(N * ept_row_main)), tail_count_p
                    )
                    mean_p = b.fdiv(sum_p, count_val)
                else:
                    # Byte-identical to the pre-tail-support emission: two
                    # independent ``b.const_f32(count_p)`` calls (one folded
                    # into ``inv_count_p``, one re-emitted inline at the
                    # Welford call below), not a single value reused --
                    # deduplicating them changes the op count/value
                    # numbering the C++ mirror must match byte-for-byte.
                    count_p = float(spec.elems_per_thread)
                    inv_count_p = b.const_f32(1.0 / count_p)
                    mean_p = b.fmul(sum_p, inv_count_p)
                m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

                if spec.fast_reduce:
                    mean, var = welford_block_reduce_stable_wave_prologue(
                        b,
                        mean_p,
                        m2_p,
                        count_val if tail_chunks else b.const_f32(count_p),
                        lds_mean,
                        lds_m2,
                        lds_count,
                        tid,
                        block_size=BS,
                        wave_size=spec.wave_size,
                    )
                else:
                    mean, var = welford_block_reduce_stable(
                        b,
                        mean_p,
                        m2_p,
                        count_val if tail_chunks else b.const_f32(count_p),
                        lds_mean,
                        lds_m2,
                        lds_count,
                        tid,
                        block_size=BS,
                    )

                with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
                    old_mean = b.global_load_f32(RunningMean, channel)
                    old_var = b.global_load_f32(RunningVar, channel)
                    one_minus_mom = b.fsub(b.const_f32(1.0), momentum)
                    new_running_mean = b.fadd(
                        b.fmul(one_minus_mom, old_mean), b.fmul(momentum, mean)
                    )
                    new_running_var = b.fadd(
                        b.fmul(one_minus_mom, old_var), b.fmul(momentum, var)
                    )
                    b.global_store(RunningMean, channel, new_running_mean)
                    b.global_store(RunningVar, channel, new_running_var)
                    if spec.save_mean_invstd:
                        inv_std_save = b.rsqrt(b.fadd(var, eps))
                        store_scalar_from_f32(b, Mean, channel, mean, dtype=spec.dtype)
                        store_scalar_from_f32(
                            b, InvStd, channel, inv_std_save, dtype=spec.dtype
                        )
            else:
                # Inference: no reduction, no LDS, no sync -- just read the
                # running stats this channel was given.
                mean = b.global_load_f32(RunningMean, channel)
                var = b.global_load_f32(RunningVar, channel)

            inv_std = b.rsqrt(b.fadd(var, eps))

            # Gamma / Beta are per-channel scalars -- constant for the whole
            # CTA -- so load them once here, not once per swept chunk
            # (contrast layernorm2d's per-column Gamma/Beta, which vary
            # along the row).
            gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
            beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
            scale = b.fmul(inv_std, gamma_v)

            # --- Pass 2: N-loop of per-sample row writes ----------------
            for n in range(N):
                row_n = row_values[n]
                y_tile_n = make_tile_window(
                    y_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                )
                x_tile_n = x_tiles_by_n[n]
                cached_n = cached_by_n[n] if (spec.training and not two_pass) else ()

                def pass2_body(n_off, _k, x_scalars, _x_tile_n=x_tile_n):
                    if two_pass or not spec.training:
                        x_scalars = _x_tile_n.load_vec_as_f32(
                            b, b.const_i32(0), n_off, n=VEC
                        )
                    return [
                        b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
                        for i in range(VEC)
                    ]

                pass2_row_chunks(
                    b,
                    y_tile_n,
                    tid=tid,
                    block_size=BS,
                    vec=VEC,
                    elems_per_thread=ept_row,
                    body=pass2_body,
                    cached_f32=cached_n,
                )

                # Scalar tail write-back. Unlike pass 1's masked reduction,
                # a store cannot be branch-free: clamping an out-of-range
                # lane's index to write a value at an in-range location
                # would make multiple lanes race to store (correct value
                # vs. stale/garbage) into the SAME element. So this uses a
                # real ``scf_if`` guard, matching elementwise.py's scalar-
                # tail idiom, and always reloads ``X`` directly (the tail is
                # at most ``BS*VEC - 1`` elements -- not worth caching).
                for tc in range(tail_chunks):
                    idx_col = b.add(b.const_i32(main_elems_per_row + tc * BS), tid)
                    valid = b.cmp_lt(idx_col, b.const_i32(HW))
                    with b.scf_if(valid):
                        flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_col)
                        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                        y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                        store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)
        else:
            # --- n_groups > 1: N and H*W split across the CTA's threads
            # simultaneously (mirrors MIOpen's spatial batchnorm thread
            # mapping: `chwid = lid%hw`, `lidihw = lid/hw`, looping
            # `nid = n*segihw + lidihw`). Training and inference are kept
            # as two separate emission orders below (not a shared
            # preamble) so the pre-existing inference emission stays
            # byte-for-byte identical to before this training support was
            # added -- only the ``if spec.training`` arm is new.
            hw_threads = BS // NG
            # Same tail shape as NG==1's chunks_main/tail_chunks (see the
            # comment above build_batchnorm2d's start), scoped to
            # hw_threads instead of BS: the vectorized main sweep covers
            # whole hw_threads*vec chunks of one og's row, and the
            # remaining tail_per_row_ng elements (0 <= tail_per_row_ng <
            # hw_threads*vec) are handled by a masked scalar tail below.
            # When HW % (hw_threads*vec) == 0 (the old exact-divisibility
            # case), tail_chunks_ng == 0 and every path below collapses
            # back to the original emission.
            chunks_main_ng = HW // (hw_threads * VEC)
            ept_row_ng = chunks_main_ng * VEC  # per-thread vectorized elems/row
            main_elems_per_row_ng = ept_row_ng * hw_threads
            tail_per_row_ng = HW - main_elems_per_row_ng
            tail_chunks_ng = (
                (tail_per_row_ng + hw_threads - 1) // hw_threads
                if tail_per_row_ng
                else 0
            )

            if spec.training:
                # Training: each thread accumulates Welford partials over
                # its assigned (og, tid_within_row) slice of elements,
                # then merges with the SAME welford_block_reduce_stable/
                # _wave_prologue call the NG==1 path uses over the full
                # tid/block_size=BS -- that merge only needs each of the
                # BS threads to hand in its own (mean, M2, count) triple,
                # it does not care how a thread's slice was chosen, so no
                # new reduction helper or two-tier (sub-group then
                # cross-group) merge is needed. Unlike NG==1's training
                # path this never caches X in registers across pass 1/
                # pass 2 (``cache=False`` below): the shapes this lever
                # targets (HW small enough that block_size structurally
                # caps at NG==1) have so little data that a second HBM
                # pass is cheap -- threading a per-og cached-tile array
                # through the merge would be disproportionate to what
                # this fixes.
                c_hw_threads = b.const_i32(hw_threads)
                lidihw = b.div(tid, c_hw_threads)
                tid_within_row = b.mod(tid, c_hw_threads)

                og_rows: List = []
                og_x_tiles: List = []
                for og in range(N // NG):
                    n_runtime = b.add(b.const_i32(og * NG), lidihw)
                    row_n = b.add(b.mul(n_runtime, b.const_i32(C)), channel)
                    og_rows.append(row_n)
                    og_x_tiles.append(
                        make_tile_window(
                            x_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                        )
                    )

                sum_p = b.const_f32(0.0)
                sumsq_p = b.const_f32(0.0)
                tail_count_p = b.const_f32(0.0) if tail_chunks_ng else None

                def pass1_body_ng(_n_off, x_scalars):
                    nonlocal sum_p, sumsq_p
                    sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
                    sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
                    sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))

                for row_n, x_tile_n in zip(og_rows, og_x_tiles):
                    sweep_row_chunks(
                        b,
                        x_tile_n,
                        tid=tid_within_row,
                        block_size=hw_threads,
                        vec=VEC,
                        elems_per_thread=ept_row_ng,
                        body=pass1_body_ng,
                        cache=False,
                    )

                    # Masked scalar tail, same idiom as NG==1's pass-1
                    # tail (see the comment above that block): clamp +
                    # select, not scf_if, since a reduction contribution
                    # is naturally zero-able.
                    for tc in range(tail_chunks_ng):
                        idx_col = b.add(
                            b.const_i32(main_elems_per_row_ng + tc * hw_threads),
                            tid_within_row,
                        )
                        valid = b.cmp_lt(idx_col, b.const_i32(HW))
                        idx_safe = b.smin(idx_col, b.const_i32(HW - 1))
                        flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_safe)
                        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                        x_masked = b.select(valid, x_raw, b.const_f32(0.0))
                        valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                        sum_p = b.fadd(sum_p, x_masked)
                        sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
                        tail_count_p = b.fadd(tail_count_p, valid_f)

                if tail_chunks_ng:
                    count_val = b.fadd(
                        b.const_f32(float((N // NG) * ept_row_ng)), tail_count_p
                    )
                    # A thread whose tid_within_row falls entirely outside
                    # every og's valid HW range (possible when hw_threads
                    # alone exceeds HW, e.g. 7x7=49 < hw_threads=64) gets a
                    # real count_val of 0 -- sum_p is correctly 0 for it too
                    # (every tail contribution was masked to 0), but
                    # sum_p/count_val would be a literal 0/0 = NaN, and NaN
                    # poisons the block merge below even at weight 0 (its
                    # delta term is NaN * 0 = NaN, not 0). Dividing by
                    # max(count_val, 1) instead keeps every real (count>=1)
                    # thread's mean_p untouched and gives a 0-count thread a
                    # harmless, finite mean_p=0/1=0 -- the merge still uses
                    # the REAL (possibly-zero) count_val as that thread's
                    # weight, so it contributes nothing either way.
                    count_safe = b.fmax(count_val, b.const_f32(1.0))
                    mean_p = b.fdiv(sum_p, count_safe)
                    m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))
                else:
                    # No-tail formula: byte-identical to the original
                    # (pre-tail-support) emission -- a single
                    # count_val = b.const_f32(count_p) call emitted right
                    # after m2_p and reused by both Welford branches below
                    # (unlike NG==1's own no-tail branch, which duplicates
                    # the call once per branch instead -- this shape
                    # predates this lever's tail support and is preserved
                    # exactly, not changed to match).
                    count_p = float(spec.elems_per_thread)
                    inv_count_p = b.const_f32(1.0 / count_p)
                    mean_p = b.fmul(sum_p, inv_count_p)
                    m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))
                    count_val = b.const_f32(count_p)

                if spec.fast_reduce:
                    mean, var = welford_block_reduce_stable_wave_prologue(
                        b,
                        mean_p,
                        m2_p,
                        count_val,
                        lds_mean,
                        lds_m2,
                        lds_count,
                        tid,
                        block_size=BS,
                        wave_size=spec.wave_size,
                    )
                else:
                    mean, var = welford_block_reduce_stable(
                        b,
                        mean_p,
                        m2_p,
                        count_val,
                        lds_mean,
                        lds_m2,
                        lds_count,
                        tid,
                        block_size=BS,
                    )

                with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
                    old_mean = b.global_load_f32(RunningMean, channel)
                    old_var = b.global_load_f32(RunningVar, channel)
                    one_minus_mom = b.fsub(b.const_f32(1.0), momentum)
                    new_running_mean = b.fadd(
                        b.fmul(one_minus_mom, old_mean), b.fmul(momentum, mean)
                    )
                    new_running_var = b.fadd(
                        b.fmul(one_minus_mom, old_var), b.fmul(momentum, var)
                    )
                    b.global_store(RunningMean, channel, new_running_mean)
                    b.global_store(RunningVar, channel, new_running_var)
                    if spec.save_mean_invstd:
                        inv_std_save = b.rsqrt(b.fadd(var, eps))
                        store_scalar_from_f32(b, Mean, channel, mean, dtype=spec.dtype)
                        store_scalar_from_f32(
                            b, InvStd, channel, inv_std_save, dtype=spec.dtype
                        )

                inv_std = b.rsqrt(b.fadd(var, eps))
                gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
                beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
                scale = b.fmul(inv_std, gamma_v)

                for row_n, x_tile_n in zip(og_rows, og_x_tiles):
                    y_tile_n = make_tile_window(
                        y_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                    )

                    def pass2_body(n_off, _k, x_scalars, _x_tile_n=x_tile_n):
                        x_scalars = _x_tile_n.load_vec_as_f32(
                            b, b.const_i32(0), n_off, n=VEC
                        )
                        return [
                            b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
                            for i in range(VEC)
                        ]

                    pass2_row_chunks(
                        b,
                        y_tile_n,
                        tid=tid_within_row,
                        block_size=hw_threads,
                        vec=VEC,
                        elems_per_thread=ept_row_ng,
                        body=pass2_body,
                        cached_f32=(),
                    )

                    # Masked scalar tail write-back, same idiom as NG==1's
                    # pass-2 tail: a real scf_if guard, not select -- a
                    # store cannot be branch-free (see the comment above
                    # NG==1's pass-2 tail for why).
                    for tc in range(tail_chunks_ng):
                        idx_col = b.add(
                            b.const_i32(main_elems_per_row_ng + tc * hw_threads),
                            tid_within_row,
                        )
                        valid = b.cmp_lt(idx_col, b.const_i32(HW))
                        with b.scf_if(valid):
                            flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_col)
                            x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                            y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                            store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)
            else:
                # Inference: is_valid_spec guarantees training=False here,
                # so there is no Welford/LDS state to thread across the
                # sub-group boundary -- just the scalar running-stat read
                # and the elementwise affine, same as the NG==1 inference
                # branch above, applied with a different thread
                # decomposition. Emission order preserved byte-for-byte
                # from before training support was added above.
                mean = b.global_load_f32(RunningMean, channel)
                var = b.global_load_f32(RunningVar, channel)
                inv_std = b.rsqrt(b.fadd(var, eps))
                gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
                beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
                scale = b.fmul(inv_std, gamma_v)

                c_hw_threads = b.const_i32(hw_threads)
                lidihw = b.div(tid, c_hw_threads)
                tid_within_row = b.mod(tid, c_hw_threads)

                # Python-unrolled over N // NG outer steps; each step, NG
                # sub-groups of hw_threads lanes each sweep NG different
                # samples' rows concurrently (sample index = og*NG + lidihw,
                # a runtime value since lidihw depends on tid).
                for og in range(N // NG):
                    n_runtime = b.add(b.const_i32(og * NG), lidihw)
                    row_n = b.add(b.mul(n_runtime, b.const_i32(C)), channel)
                    y_tile_n = make_tile_window(
                        y_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                    )
                    x_tile_n = make_tile_window(
                        x_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
                    )

                    def pass2_body(n_off, _k, x_scalars, _x_tile_n=x_tile_n):
                        x_scalars = _x_tile_n.load_vec_as_f32(
                            b, b.const_i32(0), n_off, n=VEC
                        )
                        return [
                            b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
                            for i in range(VEC)
                        ]

                    pass2_row_chunks(
                        b,
                        y_tile_n,
                        tid=tid_within_row,
                        block_size=hw_threads,
                        vec=VEC,
                        elems_per_thread=ept_row_ng,
                        body=pass2_body,
                        cached_f32=(),
                    )

                    # Masked scalar tail write-back -- same idiom as the
                    # training sub-branch's pass-2 tail above.
                    for tc in range(tail_chunks_ng):
                        idx_col = b.add(
                            b.const_i32(main_elems_per_row_ng + tc * hw_threads),
                            tid_within_row,
                        )
                        valid = b.cmp_lt(idx_col, b.const_i32(HW))
                        with b.scf_if(valid):
                            flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_col)
                            x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                            y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                            store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)

    return b.kernel


def _build_batchnorm2d_dynamic(spec: BatchNorm2DSpec) -> KernelDef:
    """AOT-generic BatchNorm2D: one compiled binary serves any runtime
    ``N/H/W`` (``C`` only ever mattered for row addressing and host-side
    grid sizing, both already runtime-safe -- see ``batchnorm2d_grid``).
    Requires ``channels_per_block == n_groups == 1`` (enforced by
    :func:`is_valid_spec`): one CTA per channel, no sub-group split.

    The N-loop (``build_batchnorm2d``'s Python-unrolled ``for n in
    range(N)``) becomes a genuine runtime ``scf.for``
    (:meth:`rocke.core.ir.IRBuilder.scf_for_iter`) carrying the running
    Welford sum/sumsq/tail-count accumulators across samples; nested
    inside each sample, the per-row ``H*W`` sweep is itself a runtime loop
    (:func:`rocke.helpers.sweep.sweep_row_chunks_dynamic`) plus a
    ``vec``-wide masked scalar tail -- the same main/tail split
    ``build_batchnorm2d``'s compile-time path uses, computed at runtime.

    Always streams X twice (no per-sample register cache: a runtime trip
    count can't back one sized at build time), and gamma/beta/running
    stats are loaded exactly once (they are per-channel scalars, same as
    the static path).
    """
    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    cap = spec.n * spec.c  # capacity placeholder; not used for addressing

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Beta = b.param(
        "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    RunningMean = b.param("RunningMean", PtrType(F32, "global"), noalias=True)
    RunningVar = b.param("RunningVar", PtrType(F32, "global"), noalias=True)
    if spec.save_mean_invstd:
        Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True)
        InvStd = b.param(
            "InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)
    eps = b.param("eps", F32)
    momentum = b.param("momentum", F32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    channel = b.block_id_x()  # channels_per_block == 1: no CPB loop

    # Row stride is the runtime HW (not a compile-time shape): the
    # TensorDescriptor already accepts an SSA Value stride (see
    # helpers/tensor_view.py), so a runtime row length just works.
    x_view = make_global_view(X, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    y_view = make_global_view(Y, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))

    lds_mean = lds_m2 = lds_count = None
    if spec.training:
        lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
        lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
        lds_count = make_lds_view(
            b, dtype=F32, shape=(BS,), name_hint="lds_count"
        ).base

    if spec.training:
        # --- Pass 1: runtime N-loop of runtime-trip-count row sweeps -----
        def pass1_body(_n_off, x_scalars, iter_vars):
            sum_p, sumsq_p = iter_vars
            sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
            sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
            sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))
            return [sum_p, sumsq_p]

        def outer_pass1_body(n, outer_iter_vars):
            sum_p, sumsq_p, tail_count_p = outer_iter_vars
            row_n = b.add(b.mul(n, p_c), channel)
            x_tile_n = make_tile_window(
                x_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
            )
            (sum_p, sumsq_p), main_elems, _tail = sweep_row_chunks_dynamic(
                b,
                x_tile_n,
                tid=tid,
                block_size=BS,
                vec=VEC,
                length=HW,
                iter_args=[("sum_p_hw", sum_p), ("sumsq_p_hw", sumsq_p)],
                body=pass1_body,
                iv_name="hw0",
            )
            for tc in range(VEC):
                idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
                valid = b.cmp_lt(idx_col, HW)
                idx_safe = b.smin(idx_col, b.sub(HW, b.const_i32(1)))
                flat = b.add(b.mul(row_n, HW), idx_safe)
                x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                x_masked = b.select(valid, x_raw, b.const_f32(0.0))
                valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                sum_p = b.fadd(sum_p, x_masked)
                sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
                tail_count_p = b.fadd(tail_count_p, valid_f)
            return [sum_p, sumsq_p, tail_count_p]

        for_op = b.scf_for_iter(
            b.const_i32(0),
            p_n,
            b.const_i32(1),
            [
                ("sum_p", b.const_f32(0.0)),
                ("sumsq_p", b.const_f32(0.0)),
                ("tail_count_p", b.const_f32(0.0)),
            ],
            iv_name="n0",
        )
        with for_op as (n_iv, outer_iter_vars):
            new_vals = outer_pass1_body(n_iv, outer_iter_vars)
            b.scf_yield(*new_vals)
        sum_p, sumsq_p, tail_count_p = for_op.results

        # Loop-invariant across samples: every n sweeps the same HW, so the
        # per-thread main-loop element count is the same each iteration --
        # compute the total once rather than carrying it through the loop.
        trip_count = b.div(HW, b.const_i32(BS * VEC))
        main_per_thread_total = b.mul(p_n, b.mul(trip_count, b.const_i32(VEC)))
        count_val = b.fadd(b.sitofp_f32(main_per_thread_total), tail_count_p)

        mean_p = b.fdiv(sum_p, count_val)
        m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

        if spec.fast_reduce:
            mean, var = welford_block_reduce_stable_wave_prologue(
                b,
                mean_p,
                m2_p,
                count_val,
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
                wave_size=spec.wave_size,
            )
        else:
            mean, var = welford_block_reduce_stable(
                b, mean_p, m2_p, count_val, lds_mean, lds_m2, lds_count, tid, block_size=BS
            )

        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            old_mean = b.global_load_f32(RunningMean, channel)
            old_var = b.global_load_f32(RunningVar, channel)
            one_minus_mom = b.fsub(b.const_f32(1.0), momentum)
            new_running_mean = b.fadd(
                b.fmul(one_minus_mom, old_mean), b.fmul(momentum, mean)
            )
            new_running_var = b.fadd(
                b.fmul(one_minus_mom, old_var), b.fmul(momentum, var)
            )
            b.global_store(RunningMean, channel, new_running_mean)
            b.global_store(RunningVar, channel, new_running_var)
            if spec.save_mean_invstd:
                inv_std_save = b.rsqrt(b.fadd(var, eps))
                store_scalar_from_f32(b, Mean, channel, mean, dtype=spec.dtype)
                store_scalar_from_f32(b, InvStd, channel, inv_std_save, dtype=spec.dtype)
    else:
        mean = b.global_load_f32(RunningMean, channel)
        var = b.global_load_f32(RunningVar, channel)

    inv_std = b.rsqrt(b.fadd(var, eps))
    gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
    beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
    scale = b.fmul(inv_std, gamma_v)

    # --- Pass 2: runtime N-loop (no carried state) of runtime write sweeps
    def pass2_body(n_off, x_tile_n):
        x_scalars = x_tile_n.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        return [
            b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
            for i in range(VEC)
        ]

    outer_for = b.scf_for(b.const_i32(0), p_n, b.const_i32(1), iv_name="n1")
    with outer_for as n_iv:
        row_n = b.add(b.mul(n_iv, p_c), channel)
        x_tile_n = make_tile_window(
            x_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
        )
        y_tile_n = make_tile_window(
            y_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
        )
        trip_count2 = b.div(HW, b.const_i32(BS * VEC))
        main_elems2 = b.mul(trip_count2, b.const_i32(BS * VEC))
        pass2_row_chunks_dynamic(
            b,
            y_tile_n,
            tid=tid,
            block_size=BS,
            vec=VEC,
            main_elems_per_row=main_elems2,
            trip_count=trip_count2,
            body=lambda n_off, _x_tile_n=x_tile_n: pass2_body(n_off, _x_tile_n),
            iv_name="hw1",
        )
        for tc in range(VEC):
            idx_col = b.add(b.add(main_elems2, b.const_i32(tc * BS)), tid)
            valid = b.cmp_lt(idx_col, HW)
            with b.scf_if(valid):
                flat = b.add(b.mul(row_n, HW), idx_col)
                x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)

    return b.kernel


def batchnorm2d_grid(c: int, spec: BatchNorm2DSpec) -> Tuple[int, int, int]:
    return (int(c) // spec.channels_per_block, 1, 1)


def batchnorm2d_signature(spec: BatchNorm2DSpec):
    from ...helpers.spec import SignatureBuilder

    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Gamma", spec.dtype)
        .ptr("Beta", spec.dtype)
        .ptr("Y", spec.dtype)
        .ptr("RunningMean", "f32")
        .ptr("RunningVar", "f32")
    )
    if spec.save_mean_invstd:
        sb.ptr("Mean", spec.dtype).ptr("InvStd", spec.dtype)
    return (
        sb.scalar("N", "i32")
        .scalar("C", "i32")
        .scalar("H", "i32")
        .scalar("W", "i32")
        .scalar("eps", "f32")
        .scalar("momentum", "f32")
        .build()
    )


# ===================================================================== #
# Cross-CTA spatial-split reduction (two-kernel, workspace-based)
# ===================================================================== #
#
# Motivating gap: the single-kernel path above grids one CTA per
# (channels_per_block) channels -- for a large-H*W/small-C shape (e.g.
# N=1,C=16,H=640,W=640 on gfx950's 256 CUs) that is as few as 16 CTAs, most
# of the device idle, and there is no axis to split one channel's spatial
# reduction across more CTAs without a cross-CTA combine step. Unlike
# GEMM's split-K (plain float accumulation, composes under a single atomic
# add), a Welford (mean, M2, count) triple does **not** compose under
# atomics, so this is a genuine two-kernel design, not an epilogue tweak:
#
#   1. :func:`build_batchnorm2d_reduce_partial` -- one CTA per
#      ``(channel-group, spatial split)``, writes a partial ``(mean, M2,
#      count)`` triple per split to a caller-owned workspace.
#   2. :func:`build_batchnorm2d_combine_apply` -- one CTA per channel-group
#      (same grid as the single-kernel path), merges that channel's
#      ``split_hw`` partials via the SAME :func:`welford_block_reduce_stable`
#      /``_wave_prologue`` primitive the single-kernel reduction already
#      uses (feeding each thread one workspace slot instead of a row-sweep
#      partial -- the merge math does not care where a thread's ``(mean,
#      M2, count)`` came from), updates the running stats, and performs the
#      normal pass-2 normalize+write.
#
# Deliberately NOT wired into ``rocke.dispatch.families.batchnorm``'s
# ``KernelCandidate``/``dispatch_batchnorm`` surface: that abstraction is
# built around one candidate = one kernel launch plus a static
# (grid, block) pair, with no notion of an inter-kernel workspace a caller
# must allocate/zero/thread through two launches. Extending that shared
# abstraction to a multi-kernel candidate is a bigger, cross-family change
# than this feature's scope -- callers that want this lever build+launch
# the two kernels directly, in order, sharing one ``Workspace`` buffer
# sized by :func:`batchnorm2d_split_workspace_floats`.
#
# Training-only: inference has no batch reduction to split.


@dataclass(frozen=True)
class BatchNorm2DSplitSpec:
    """One cross-CTA spatial-split batchnorm2d reduction instance.

    Shares ``n/c/h/w/block_size/vec/dtype/channels_per_block/fast_reduce/
    wave_size`` field names with :class:`BatchNorm2DSpec` by convention, not
    inheritance -- this is a standalone, training-only spec for the
    two-kernel path above, not a drop-in for the single-kernel builders.
    """

    n: int
    c: int
    h: int
    w: int
    split_hw: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    channels_per_block: int = 1
    fast_reduce: bool = False
    wave_size: int = 64
    name: str = "rocke_batchnorm2d_split"

    @property
    def hw(self) -> int:
        return self.h * self.w

    @property
    def hw_chunk(self) -> int:
        """Per-split spatial chunk size: ``ceil(H*W / split_hw)``.

        The last split's actual valid range can be shorter than this (when
        ``H*W`` is not a multiple of ``split_hw``) -- every read in
        :func:`build_batchnorm2d_reduce_partial` is masked against the true
        ``H*W`` bound, not just this chunk size, so a short last split only
        contributes fewer valid elements rather than reading out of range.
        """
        return -(-self.hw // self.split_hw)

    def kernel_name(self, suffix: str) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.n}C{self.c}H{self.h}W{self.w}",
            f"b{self.block_size}",
            f"v{self.vec}",
            f"split{self.split_hw}",
            f"cpb{self.channels_per_block}" if self.channels_per_block != 1 else "",
            suffix,
            flags={"fr": self.fast_reduce},
        )


def batchnorm2d_split_workspace_floats(spec: BatchNorm2DSplitSpec) -> int:
    """f32 element count the caller must allocate (and need not zero-init:
    every slot is written exactly once by :func:`build_batchnorm2d_reduce_partial`
    before :func:`build_batchnorm2d_combine_apply` reads it) for
    ``Workspace``: one ``(mean, M2, count)`` triple per ``(channel, split
    index)``."""
    return int(spec.c) * int(spec.split_hw) * 3


def is_valid_split_spec(
    spec: BatchNorm2DSplitSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one split-reduction config on ``arch``.

    ``split_hw <= block_size`` is required: :func:`build_batchnorm2d_combine_apply`
    gives each thread at most one workspace slot (``tid < split_hw``), not a
    per-thread loop over multiple slots -- ``split_hw`` is meant to stay
    small (amortizing one CTA's worth of spatial work across a handful of
    CTAs), so this is not a practical limitation.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.n <= 0 or spec.c <= 0 or spec.h <= 0 or spec.w <= 0:
        return False, "n, c, h, w must all be positive"
    if spec.split_hw <= 1:
        return False, "split_hw must be > 1 (use the single-kernel path otherwise)"
    if spec.channels_per_block <= 0:
        return False, "channels_per_block must be positive"
    if spec.c % spec.channels_per_block != 0:
        return False, (
            f"c {spec.c} not divisible by channels_per_block "
            f"{spec.channels_per_block}"
        )
    if spec.split_hw > spec.block_size:
        return False, (
            f"split_hw {spec.split_hw} > block_size {spec.block_size}: the "
            "combine kernel gives each thread at most one workspace slot"
        )
    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, (
            f"fast_reduce requires block_size {spec.block_size} divisible by "
            f"wave_size {spec.wave_size}"
        )

    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=None,  # every HW chunk is masked, not exact-fit
        )
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    bytes_lds = 3 * spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def batchnorm2d_reduce_partial_grid(
    c: int, spec: BatchNorm2DSplitSpec
) -> Tuple[int, int, int]:
    return (int(c) // spec.channels_per_block, int(spec.split_hw), 1)


def build_batchnorm2d_reduce_partial(spec: BatchNorm2DSplitSpec) -> KernelDef:
    """Kernel A of the split-reduction pair.

    Grid ``(C // channels_per_block, split_hw, 1)``: ``block_id_x`` selects
    the channel-group (same meaning as the single-kernel path's grid),
    ``block_id_y`` selects which of this channel's ``split_hw`` spatial
    chunks this CTA reduces. Writes that chunk's ``(mean, M2, count)`` triple
    to ``Workspace[(channel*split_hw + split_id)*3 + {0,1,2}]`` -- no
    running-stat update, no ``Y`` write; that is
    :func:`build_batchnorm2d_combine_apply`'s job once every split for a
    channel has landed in the workspace.

    Every element this kernel reads is masked via ``select`` (branch-free,
    the same idiom ``build_batchnorm2d``'s HW tail uses): ``hw_chunk`` is
    not generally divisible by ``block_size * vec``, and the last split's
    chunk can run past ``H*W``, so every thread's sweep is masked uniformly
    rather than split into a fast/tail pair -- simpler and correct; this is
    new infrastructure, not yet perf-tuned the way the single-kernel path's
    fast path is.
    """
    ok, why = is_valid_split_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d split spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    N, C, HW = spec.n, spec.c, spec.hw
    hw_chunk = spec.hw_chunk
    chunks_per_thread = -(-hw_chunk // (BS * VEC))
    CPB = spec.channels_per_block

    b = IRBuilder(spec.kernel_name("reduce"))
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Workspace = b.param("Workspace", PtrType(F32, "global"), noalias=True)
    _ = b.param("N", I32)  # noqa: F841 - ABI symmetry; baked as spec.n
    _ = b.param("C", I32)  # noqa: F841 - ABI symmetry; baked as spec.c
    _ = b.param("H", I32)  # noqa: F841 - ABI symmetry; baked as spec.h
    _ = b.param("W", I32)  # noqa: F841 - ABI symmetry; baked as spec.w

    tid = b.thread_id_x()
    base_channel = b.block_id_x()
    split_id = b.block_id_y()

    lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
    lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
    lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    hw_start = b.mul(split_id, b.const_i32(hw_chunk))
    c_hw = b.const_i32(HW)

    # Total valid-element count for this split, independent of ``cb``
    # (every channel in this CTA's channels_per_block group shares the same
    # spatial chunk): ``hw_start < HW`` always holds for a split_id the
    # launch grid actually produces (``hw_chunk = ceil(HW/split_hw)`` means
    # the last split's start is strictly < HW), so ``remaining`` is always
    # positive and this needs no additional clamp. Computed once here
    # (not re-derived from the per-thread masked ``count_p`` sums) because
    # ``welford_block_reduce_stable`` only returns ``(mean, var)``, not the
    # merged count -- ``M2 = var * count`` needs this exact total to invert.
    remaining = b.sub(c_hw, hw_start)
    valid_len = b.smin(remaining, b.const_i32(hw_chunk))
    total_count = b.sitofp_f32(b.mul(b.const_i32(N), valid_len))

    for cb in range(CPB):
        if CPB == 1:
            channel = base_channel
        else:
            channel = b.add(b.mul(base_channel, b.const_i32(CPB)), b.const_i32(cb))
            if cb > 0:
                b.sync()

        sum_p = b.const_f32(0.0)
        sumsq_p = b.const_f32(0.0)
        count_p = b.const_f32(0.0)

        for n in range(N):
            row_n = b.add(b.const_i32(n * C), channel)
            row_base = b.mul(row_n, c_hw)
            for k in range(chunks_per_thread):
                for i in range(VEC):
                    local = b.add(
                        b.const_i32(k * BS * VEC + i), b.mul(tid, b.const_i32(VEC))
                    )
                    idx_local = b.add(hw_start, local)
                    valid = b.land(
                        b.cmp_lt(local, b.const_i32(hw_chunk)),
                        b.cmp_lt(idx_local, c_hw),
                    )
                    idx_safe = b.smin(idx_local, b.const_i32(HW - 1))
                    flat = b.add(row_base, idx_safe)
                    x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                    x_masked = b.select(valid, x_raw, b.const_f32(0.0))
                    valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                    sum_p = b.fadd(sum_p, x_masked)
                    sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
                    count_p = b.fadd(count_p, valid_f)

        is_empty = b.fcmp("oeq", count_p, b.const_f32(0.0))
        safe_count = b.select(is_empty, b.const_f32(1.0), count_p)
        mean_p = b.fdiv(sum_p, safe_count)
        m2_p = b.select(
            is_empty,
            b.const_f32(0.0),
            b.fsub(sumsq_p, b.fmul(mean_p, sum_p)),
        )
        mean_p = b.select(is_empty, b.const_f32(0.0), mean_p)

        if spec.fast_reduce:
            mean, var = welford_block_reduce_stable_wave_prologue(
                b,
                mean_p,
                m2_p,
                count_p,
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
                wave_size=spec.wave_size,
            )
        else:
            mean, var = welford_block_reduce_stable(
                b,
                mean_p,
                m2_p,
                count_p,
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
            )

        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            slot = b.mul(
                b.add(b.mul(channel, b.const_i32(spec.split_hw)), split_id),
                b.const_i32(3),
            )
            # welford_block_reduce_stable returns (mean, variance =
            # M2_total/count_total), not M2 itself -- re-derive
            # M2_total = var * total_count so the combine kernel's merge
            # input matches build_batchnorm2d's pass-1 shape exactly,
            # (mean, M2, count), not (mean, variance, count).
            m2_out = b.fmul(var, total_count)
            b.global_store(Workspace, slot, mean)
            b.global_store(Workspace, b.add(slot, b.const_i32(1)), m2_out)
            b.global_store(Workspace, b.add(slot, b.const_i32(2)), total_count)

    return b.kernel


def batchnorm2d_combine_apply_grid(
    c: int, spec: BatchNorm2DSplitSpec
) -> Tuple[int, int, int]:
    return (int(c) // spec.channels_per_block, 1, 1)


def build_batchnorm2d_combine_apply(spec: BatchNorm2DSplitSpec) -> KernelDef:
    """Kernel B of the split-reduction pair.

    Grid ``(C // channels_per_block, 1, 1)`` -- same grid as the single-
    kernel path. Each thread ``tid < split_hw`` reads one workspace slot for
    its channel (the Welford identity -- ``mean=0, M2=0, count=0`` -- for
    ``tid >= split_hw``, via the same branch-free ``select`` masking
    ``build_batchnorm2d``'s HW tail uses), merges all ``split_hw`` partials
    with the SAME :func:`welford_block_reduce_stable`/``_wave_prologue``
    primitive the single-kernel path's pass 1 reduction uses -- that merge
    math does not care whether a thread's partial came from a row sweep or
    a workspace read. Updates the running stats, then re-reads ``X`` and
    writes the normalized ``Y`` over the full ``H*W``: structurally the same
    pass 2 as :func:`build_batchnorm2d`'s NG==1 training path (including its
    HW scalar tail), just always in "reload" mode since nothing is cached
    across the kernel-launch boundary.

    Does not support ``save_mean_invstd`` (out of scope for this first cut
    of the split-reduction lever).
    """
    ok, why = is_valid_split_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d split spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    N, C, HW = spec.n, spec.c, spec.hw
    CPB = spec.channels_per_block
    split_hw = spec.split_hw

    vec_bs = BS * VEC
    chunks_main = HW // vec_bs
    ept_row_main = chunks_main * VEC
    main_elems_per_row = ept_row_main * BS
    tail_per_row = HW - main_elems_per_row
    tail_chunks = (tail_per_row + BS - 1) // BS if tail_per_row else 0

    b = IRBuilder(spec.kernel_name("combine"))
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Workspace = b.param(
        "Workspace", PtrType(F32, "global"), noalias=True, readonly=True
    )
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Beta = b.param(
        "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    RunningMean = b.param("RunningMean", PtrType(F32, "global"), noalias=True)
    RunningVar = b.param("RunningVar", PtrType(F32, "global"), noalias=True)
    _ = b.param("N", I32)  # noqa: F841 - ABI symmetry; baked as spec.n
    _ = b.param("C", I32)  # noqa: F841 - ABI symmetry; baked as spec.c
    _ = b.param("H", I32)  # noqa: F841 - ABI symmetry; baked as spec.h
    _ = b.param("W", I32)  # noqa: F841 - ABI symmetry; baked as spec.w
    eps = b.param("eps", F32)
    momentum = b.param("momentum", F32)

    tid = b.thread_id_x()
    base_channel = b.block_id_x()

    x_view = make_naive_tensor_view_packed(X, shape=(N * C, HW), dtype=io_ty)
    y_view = make_naive_tensor_view_packed(Y, shape=(N * C, HW), dtype=io_ty)

    lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
    lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
    lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    for cb in range(CPB):
        if CPB == 1:
            channel = base_channel
        else:
            channel = b.add(b.mul(base_channel, b.const_i32(CPB)), b.const_i32(cb))
            if cb > 0:
                b.sync()

        has_slot = b.cmp_lt(tid, b.const_i32(split_hw))
        slot_tid = b.select(has_slot, tid, b.const_i32(0))
        slot = b.mul(
            b.add(b.mul(channel, b.const_i32(split_hw)), slot_tid), b.const_i32(3)
        )
        mean_raw = b.global_load_f32(Workspace, slot)
        m2_raw = b.global_load_f32(Workspace, b.add(slot, b.const_i32(1)))
        count_raw = b.global_load_f32(Workspace, b.add(slot, b.const_i32(2)))
        mean_p = b.select(has_slot, mean_raw, b.const_f32(0.0))
        m2_p = b.select(has_slot, m2_raw, b.const_f32(0.0))
        count_p = b.select(has_slot, count_raw, b.const_f32(0.0))

        if spec.fast_reduce:
            mean, var = welford_block_reduce_stable_wave_prologue(
                b,
                mean_p,
                m2_p,
                count_p,
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
                wave_size=spec.wave_size,
            )
        else:
            mean, var = welford_block_reduce_stable(
                b,
                mean_p,
                m2_p,
                count_p,
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
            )

        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            old_mean = b.global_load_f32(RunningMean, channel)
            old_var = b.global_load_f32(RunningVar, channel)
            one_minus_mom = b.fsub(b.const_f32(1.0), momentum)
            new_running_mean = b.fadd(
                b.fmul(one_minus_mom, old_mean), b.fmul(momentum, mean)
            )
            new_running_var = b.fadd(
                b.fmul(one_minus_mom, old_var), b.fmul(momentum, var)
            )
            b.global_store(RunningMean, channel, new_running_mean)
            b.global_store(RunningVar, channel, new_running_var)

        inv_std = b.rsqrt(b.fadd(var, eps))
        gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
        beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
        scale = b.fmul(inv_std, gamma_v)

        for n in range(N):
            row_n = b.add(b.const_i32(n * C), channel)
            y_tile_n = make_tile_window(
                y_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
            )
            x_tile_n = make_tile_window(
                x_view, lengths=(1, HW), origin=(row_n, b.const_i32(0))
            )

            def pass2_body(n_off, _k, _x_scalars, _x_tile_n=x_tile_n):
                x_scalars = _x_tile_n.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
                return [
                    b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
                    for i in range(VEC)
                ]

            pass2_row_chunks(
                b,
                y_tile_n,
                tid=tid,
                block_size=BS,
                vec=VEC,
                elems_per_thread=ept_row_main,
                body=pass2_body,
                cached_f32=(),
            )

            for tc in range(tail_chunks):
                idx_col = b.add(b.const_i32(main_elems_per_row + tc * BS), tid)
                valid_tail = b.cmp_lt(idx_col, b.const_i32(HW))
                with b.scf_if(valid_tail):
                    flat = b.add(b.mul(row_n, b.const_i32(HW)), idx_col)
                    x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                    y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                    store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)

    return b.kernel


def batchnorm2d_split_signature(spec: BatchNorm2DSplitSpec, *, kind: str):
    """``kind="reduce"`` for :func:`build_batchnorm2d_reduce_partial`'s ABI,
    ``kind="combine"`` for :func:`build_batchnorm2d_combine_apply`'s."""
    from ...helpers.spec import SignatureBuilder

    sb = SignatureBuilder()
    if kind == "reduce":
        sb.ptr("X", spec.dtype).ptr("Workspace", "f32")
    elif kind == "combine":
        (
            sb.ptr("X", spec.dtype)
            .ptr("Workspace", "f32")
            .ptr("Gamma", spec.dtype)
            .ptr("Beta", spec.dtype)
            .ptr("Y", spec.dtype)
            .ptr("RunningMean", "f32")
            .ptr("RunningVar", "f32")
        )
    else:
        raise ValueError(f"unknown batchnorm2d split kernel kind {kind!r}")
    return (
        sb.scalar("N", "i32")
        .scalar("C", "i32")
        .scalar("H", "i32")
        .scalar("W", "i32")
        .build()
        if kind == "reduce"
        else sb.scalar("N", "i32")
        .scalar("C", "i32")
        .scalar("H", "i32")
        .scalar("W", "i32")
        .scalar("eps", "f32")
        .scalar("momentum", "f32")
        .build()
    )
