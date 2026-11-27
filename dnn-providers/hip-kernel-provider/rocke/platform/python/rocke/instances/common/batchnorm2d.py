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
from ...helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    _warp_xor_reduce,
    block_lds_reduce_pair,
    block_lds_reduce_with_wave_prologue,
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


DType = Literal["f16", "bf16", "f32"]


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


def _wave_size_error(spec) -> str:
    """Refusal text for a ``wave_size`` that is not a positive power of two, else ``""``."""
    w = spec.wave_size
    if w <= 0 or w & (w - 1):
        return "wave_size must be a positive power of two"
    return ""


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
        return (
            False,
            "save_mean_invstd requires training=True (inference has no batch mean/var to save)",
        )

    if spec.channels_per_block <= 0:
        return False, "channels_per_block must be positive"
    if not spec.dynamic and spec.c % spec.channels_per_block != 0:
        return False, (
            f"c {spec.c} not divisible by channels_per_block "
            f"{spec.channels_per_block}"
        )
    # In dynamic mode spec.c is a capacity placeholder, not the real shape:
    # CPB needs no kernel-body change (C is never read in-kernel, only for
    # host-side grid sizing), so "CPB divides C" moves from a build-time
    # check here to a dispatch-time admission check against the REAL
    # runtime C (see dispatch/families/batchnorm.py) -- exactly how conv's
    # tile-shape capabilities are admitted against a runtime problem
    # instead of baked into the build.

    if spec.n_groups <= 0:
        return False, "n_groups must be positive"
    if spec.block_size % spec.n_groups != 0:
        return False, (
            f"block_size {spec.block_size} not divisible by n_groups "
            f"{spec.n_groups}"
        )
    if not spec.dynamic and spec.n % spec.n_groups != 0:
        return False, f"n {spec.n} not divisible by n_groups {spec.n_groups}"
    # Dynamic mode's "N divides n_groups evenly" requirement moves from
    # this build-time check to a dispatch-time admission check against
    # the REAL runtime N, same as channels_per_block above -- the kernel
    # body's og-loop (N // NG, computed at runtime) needs no masking for
    # it, it just must never be dispatched against an N it doesn't divide.

    if spec.fast_reduce and not spec.training:
        return (
            False,
            "fast_reduce requires training=True (inference has no reduction to speed up)",
        )
    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, (
            f"fast_reduce requires block_size {spec.block_size} divisible by "
            f"wave_size {spec.wave_size}"
        )

    # ``block_size`` is the real launch parameter (one CTA's thread count),
    # so it alone -- not ``hw_threads`` -- is what must come from the
    # hardware-meaningful ``allowed_block_sizes`` set; dtype/vec are
    # independent of ``n_groups``.
    # f32 data: 4 or 2 elements per thread (16 B / 8 B accesses); 8 would be
    # a 32 B access per thread and is not a candidate.
    _vecs = (2, 4) if spec.dtype == "f32" else (2, 4, 8)
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            allowed_dtypes=("f16", "fp16", "bf16", "f32"),
            allowed_vecs=_vecs,
        )
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
        lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

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
                        idx_col = b.add(b.const_i32(main_elems_per_row + tc * BS), tid)
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
    ``N/H/W``. ``C`` only ever mattered for row addressing and host-side
    grid sizing (``batchnorm2d_grid = C // channels_per_block``), both
    already runtime-safe -- so ``channels_per_block > 1`` needs no kernel
    change at all: the CPB loop below is identical to
    ``build_batchnorm2d``'s, Python-unrolled over the compile-time CPB,
    and dispatch (not this function) is what must only ever launch a
    dynamic CPB>1 candidate against a runtime ``C`` that divides it
    evenly (see :func:`is_valid_spec`'s dynamic branch and
    ``dispatch/families/batchnorm.py``).

    ``n_groups > 1`` DOES need kernel-body work (unlike CPB): the
    ``for og in range(N // NG):`` outer step in ``build_batchnorm2d``'s
    static ``n_groups>1`` branch has a Python-time trip count today, so it
    becomes a genuine runtime ``scf.for`` here too (``N // NG`` computed
    at runtime via ``b.div``), with each ``og`` step's
    ``hw_threads = BS // NG``-wide row sweep going through the same
    ``sweep_row_chunks_dynamic`` + masked-tail machinery as the
    ``n_groups == 1`` path, just scoped to ``hw_threads`` instead of
    ``BS`` and to ``tid_within_row`` instead of ``tid`` -- mirroring how
    ``build_batchnorm2d``'s two branches are already a full duplication of
    each other rather than one parameterized over ``hw_threads`` (per
    that function's own comment: keeping them separate is what lets the
    ``n_groups == 1`` emission stay untouched).

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
    stats are loaded exactly once per channel (they are per-channel
    scalars, same as the static path).
    """
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, CPB, NG = spec.block_size, spec.vec, spec.channels_per_block, spec.n_groups
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
    base_channel = b.block_id_x()

    # Row stride is the runtime HW (not a compile-time shape): the
    # TensorDescriptor already accepts an SSA Value stride (see
    # helpers/tensor_view.py), so a runtime row length just works.
    x_view = make_global_view(X, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    y_view = make_global_view(Y, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))

    lds_mean = lds_m2 = lds_count = None
    if spec.training:
        lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
        lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
        lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    # CPB channels, one fully resolved at a time, same shape as
    # build_batchnorm2d's own CPB loop (including the "every lane done
    # reading channel cb's LDS before any lane writes channel cb+1's LDS"
    # sync) -- CPB==1 collapses back to exactly the pre-CPB-support
    # emission, so this is byte-identical to before for the default case.
    for cb in range(CPB):
        if CPB == 1:
            channel = base_channel
        else:
            channel = b.add(b.mul(base_channel, b.const_i32(CPB)), b.const_i32(cb))
            if spec.training and cb > 0:
                b.sync()

        if NG == 1:
            if spec.training:
                # --- Pass 1: runtime N-loop of runtime-trip-count row sweeps -
                def pass1_body(_n_off, x_scalars, iter_vars):
                    sum_p, sumsq_p = iter_vars
                    sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
                    sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
                    sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))
                    return [sum_p, sumsq_p]

                def outer_pass1_body(n, outer_iter_vars, _channel=channel):
                    sum_p, sumsq_p, tail_count_p = outer_iter_vars
                    row_n = b.add(b.mul(n, p_c), _channel)
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

                # Loop-invariant across samples: every n sweeps the same HW,
                # so the per-thread main-loop element count is the same each
                # iteration -- compute the total once rather than carrying it
                # through the loop.
                trip_count = b.div(HW, b.const_i32(BS * VEC))
                main_per_thread_total = b.mul(p_n, b.mul(trip_count, b.const_i32(VEC)))
                count_val = b.fadd(b.sitofp_f32(main_per_thread_total), tail_count_p)
                # A thread can own zero elements (H*W < block_size*vec);
                # max(count,1) keeps its partial at (0, 0, 0) instead of NaN.
                count_safe = b.fmax(count_val, b.const_f32(1.0))

                mean_p = b.fdiv(sum_p, count_safe)
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
                    # Running variance is Bessel-corrected (M / (M - 1));
                    # Y and InvStd use the biased one.
                    m_total = b.sitofp_f32(b.mul(p_n, HW))
                    unbias = b.fdiv(
                        m_total,
                        b.fmax(b.fsub(m_total, b.const_f32(1.0)), b.const_f32(1.0)),
                    )
                    new_running_var = b.fadd(
                        b.fmul(one_minus_mom, old_var),
                        b.fmul(momentum, b.fmul(var, unbias)),
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
                mean = b.global_load_f32(RunningMean, channel)
                var = b.global_load_f32(RunningVar, channel)

            inv_std = b.rsqrt(b.fadd(var, eps))
            gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
            beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
            scale = b.fmul(inv_std, gamma_v)

            # --- Pass 2: runtime N-loop (no carried state) of runtime write
            # sweeps ----------------------------------------------------
            def pass2_body(n_off, x_tile_n, _mean=mean, _scale=scale, _beta_v=beta_v):
                x_scalars = x_tile_n.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
                return [
                    b.fadd(b.fmul(b.fsub(x_scalars[i], _mean), _scale), _beta_v)
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
        else:
            # --- n_groups > 1: N and H*W split across the CTA's threads
            # simultaneously, runtime-generic counterpart of
            # build_batchnorm2d's n_groups>1 branch. ``og`` ranges over a
            # RUNTIME ``N // NG`` (b.div(p_n, NG)) instead of a Python-time
            # one; each thread's sample index is still
            # ``n_runtime = og*NG + lidihw``, and the per-og row sweep is
            # the same sweep_row_chunks_dynamic + masked-tail idiom the
            # n_groups==1 branch uses, just scoped to
            # ``hw_threads = BS // NG`` threads and ``tid_within_row``
            # instead of the full block/``tid``.
            hw_threads = BS // NG
            c_hw_threads = b.const_i32(hw_threads)
            lidihw = b.div(tid, c_hw_threads)
            tid_within_row = b.mod(tid, c_hw_threads)
            n_outer_trip = b.div(p_n, b.const_i32(NG))

            if spec.training:

                def pass1_body_ng(_n_off, x_scalars, iter_vars):
                    sum_p, sumsq_p = iter_vars
                    sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
                    sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
                    sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))
                    return [sum_p, sumsq_p]

                def outer_pass1_body_ng(og, outer_iter_vars, _channel=channel):
                    sum_p, sumsq_p, tail_count_p = outer_iter_vars
                    n_runtime = b.add(b.mul(og, b.const_i32(NG)), lidihw)
                    row_n = b.add(b.mul(n_runtime, p_c), _channel)
                    x_tile_n = make_tile_window(
                        x_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
                    )
                    (sum_p, sumsq_p), main_elems, _tail = sweep_row_chunks_dynamic(
                        b,
                        x_tile_n,
                        tid=tid_within_row,
                        block_size=hw_threads,
                        vec=VEC,
                        length=HW,
                        iter_args=[("sum_p_hw_ng", sum_p), ("sumsq_p_hw_ng", sumsq_p)],
                        body=pass1_body_ng,
                        iv_name="hw0_ng",
                    )
                    for tc in range(VEC):
                        idx_col = b.add(
                            b.add(main_elems, b.const_i32(tc * hw_threads)),
                            tid_within_row,
                        )
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

                for_op_ng = b.scf_for_iter(
                    b.const_i32(0),
                    n_outer_trip,
                    b.const_i32(1),
                    [
                        ("sum_p", b.const_f32(0.0)),
                        ("sumsq_p", b.const_f32(0.0)),
                        ("tail_count_p", b.const_f32(0.0)),
                    ],
                    iv_name="og0",
                )
                with for_op_ng as (og_iv, outer_iter_vars_ng):
                    new_vals_ng = outer_pass1_body_ng(og_iv, outer_iter_vars_ng)
                    b.scf_yield(*new_vals_ng)
                sum_p, sumsq_p, tail_count_p = for_op_ng.results

                # Same loop-invariant-total shortcut as n_groups==1, scoped
                # to hw_threads. Always uses the NaN-safe max(count,1) form
                # (build_batchnorm2d's static NG>1 path only needs it when
                # Python build-time arithmetic proves a tail exists; here
                # HW is runtime, so whether a tail exists can't be known at
                # build time -- the safe form covers both cases uniformly).
                trip_count_ng = b.div(HW, b.const_i32(hw_threads * VEC))
                main_per_thread_total_ng = b.mul(
                    n_outer_trip, b.mul(trip_count_ng, b.const_i32(VEC))
                )
                count_val = b.fadd(b.sitofp_f32(main_per_thread_total_ng), tail_count_p)
                count_safe = b.fmax(count_val, b.const_f32(1.0))

                mean_p = b.fdiv(sum_p, count_safe)
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
                    # Running variance is Bessel-corrected (M / (M - 1));
                    # Y and InvStd use the biased one.
                    m_total = b.sitofp_f32(b.mul(p_n, HW))
                    unbias = b.fdiv(
                        m_total,
                        b.fmax(b.fsub(m_total, b.const_f32(1.0)), b.const_f32(1.0)),
                    )
                    new_running_var = b.fadd(
                        b.fmul(one_minus_mom, old_var),
                        b.fmul(momentum, b.fmul(var, unbias)),
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
                mean = b.global_load_f32(RunningMean, channel)
                var = b.global_load_f32(RunningVar, channel)

            inv_std = b.rsqrt(b.fadd(var, eps))
            gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
            beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
            scale = b.fmul(inv_std, gamma_v)

            def pass2_body_ng(
                n_off, x_tile_n, _mean=mean, _scale=scale, _beta_v=beta_v
            ):
                x_scalars = x_tile_n.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
                return [
                    b.fadd(b.fmul(b.fsub(x_scalars[i], _mean), _scale), _beta_v)
                    for i in range(VEC)
                ]

            outer_for_ng = b.scf_for(
                b.const_i32(0), n_outer_trip, b.const_i32(1), iv_name="og1"
            )
            with outer_for_ng as og_iv:
                n_runtime = b.add(b.mul(og_iv, b.const_i32(NG)), lidihw)
                row_n = b.add(b.mul(n_runtime, p_c), channel)
                x_tile_n = make_tile_window(
                    x_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
                )
                y_tile_n = make_tile_window(
                    y_view, lengths=(1, 1), origin=(row_n, b.const_i32(0))
                )
                trip_count2_ng = b.div(HW, b.const_i32(hw_threads * VEC))
                main_elems2_ng = b.mul(trip_count2_ng, b.const_i32(hw_threads * VEC))
                pass2_row_chunks_dynamic(
                    b,
                    y_tile_n,
                    tid=tid_within_row,
                    block_size=hw_threads,
                    vec=VEC,
                    main_elems_per_row=main_elems2_ng,
                    trip_count=trip_count2_ng,
                    body=lambda n_off, _x_tile_n=x_tile_n: pass2_body_ng(
                        n_off, _x_tile_n
                    ),
                    iv_name="hw1_ng",
                )
                for tc in range(VEC):
                    idx_col = b.add(
                        b.add(main_elems2_ng, b.const_i32(tc * hw_threads)),
                        tid_within_row,
                    )
                    valid = b.cmp_lt(idx_col, HW)
                    with b.scf_if(valid):
                        flat = b.add(b.mul(row_n, HW), idx_col)
                        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                        y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                        store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)

    return b.kernel


def batchnorm2d_grid(
    c: int, spec: BatchNorm2DSpec, *, n: int = 0, h: int = 0, w: int = 0
) -> Tuple[int, int, int]:
    if isinstance(spec, BatchNorm2DFlatInferSpec):
        return batchnorm2d_flat_infer_grid(n, c, h, w, spec)
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
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
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


# ===================================================================== #
# Flat elementwise inference (AOT-generic)
# ===================================================================== #
#
# Inference is a pure elementwise stream: ``y = x * scale[ch] + shift[ch]``
# with ``scale``/``shift`` derived from per-channel running stats. The
# one-CTA-per-channel kernels above tie parallelism to ``C``; this kernel
# decouples it by gridding over the flat ``N*C*H*W`` element space, each
# thread owning ``unroll`` groups of ``vec`` consecutive elements. The
# channel of a group is recovered from the flat index (``row = e / HW``,
# ``ch = row % C``), and ``scale``/``shift`` are computed once per group.
#
# ``aligned=True``: ``HW % vec == 0`` (checked at dispatch), so a group never
# straddles two rows and one channel serves the whole group.
# ``aligned=False``: a group may cross one row boundary (needs ``HW >= vec``),
# so the next row's channel parameters are computed as well and selected per
# lane. Requires ``(N*C*H*W) % vec == 0`` in both cases so no ragged tail
# exists, and ``N*C*H*W`` below 2**31 so flat i32 indices cannot wrap.


@dataclass(frozen=True)
class BatchNorm2DFlatInferSpec:
    """Flat elementwise BatchNorm2D inference instance (always AOT-generic)."""

    block_size: int = 256
    vec: int = 8
    unroll: int = 1
    aligned: bool = True
    dtype: DType = "f16"
    wave_size: int = 64
    name: str = "rocke_batchnorm2d_flat_infer"
    # Interface parity with BatchNorm2DSpec for the shared launch/cache code.
    training: bool = False
    save_mean_invstd: bool = False
    dynamic: bool = True
    channels_per_block: int = 1
    n_groups: int = 1
    fast_reduce: bool = False

    @property
    def tile_elems(self) -> int:
        return self.block_size * self.vec * self.unroll

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            f"u{self.unroll}",
            flags={"al": self.aligned, "dyn": True},
        )


def is_valid_flat_infer_spec(
    spec: BatchNorm2DFlatInferSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.vec not in (1, 2, 4, 8):
        return False, "vec must be 1/2/4/8"
    if spec.unroll not in (1, 2, 4, 8, 16):
        return False, "unroll must be 1/2/4/8/16"
    if spec.block_size % spec.wave_size != 0 or not 0 < spec.block_size <= 1024:
        return False, "block_size must be a wave multiple <= 1024"
    if spec.vec == 1 and not spec.aligned:
        return False, "vec=1 never straddles rows; use aligned"
    return True, ""


def flat_infer_admits(
    spec: BatchNorm2DFlatInferSpec, n: int, c: int, h: int, w: int
) -> Tuple[bool, str]:
    """Runtime-shape admission for one flat inference candidate."""
    hw = h * w
    total = n * c * hw
    if total <= 0 or total + spec.tile_elems >= 2**31:
        return False, "element count out of i32 range"
    if total % spec.vec != 0:
        return False, "N*C*H*W not a multiple of vec"
    if spec.aligned:
        if hw % spec.vec != 0:
            return False, "aligned variant needs H*W % vec == 0"
    else:
        if hw % spec.vec == 0:
            return False, "unaligned variant only for H*W % vec != 0"
        if hw < spec.vec:
            return False, "H*W smaller than vec"
    return True, ""


def batchnorm2d_flat_infer_grid(
    n: int, c: int, h: int, w: int, spec: BatchNorm2DFlatInferSpec
) -> Tuple[int, int, int]:
    total = int(n) * int(c) * int(h) * int(w)
    return ((total + spec.tile_elems - 1) // spec.tile_elems, 1, 1)


def build_batchnorm2d_flat_infer(spec: BatchNorm2DFlatInferSpec) -> KernelDef:
    ok, why = is_valid_flat_infer_spec(spec)
    if not ok:
        raise ValueError(why)
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, U = spec.block_size, spec.vec, spec.unroll

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
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)
    eps = b.param("eps", F32)
    b.param("momentum", F32)  # unused in inference; kept for the shared kernarg layout

    HW = b.mul(p_h, p_w)
    total = b.mul(b.mul(p_n, p_c), HW)
    tid = b.thread_id_x()
    tile_base = b.mul(b.block_id_x(), b.const_i32(BS * VEC * U))
    last_start = b.sub(
        total, b.const_i32(VEC)
    )  # >= 0: total is a positive multiple of VEC
    c_last = b.sub(p_c, b.const_i32(1))

    def load_x(e):
        if VEC == 1:
            return [load_scalar_as_f32(b, X, e, dtype=spec.dtype)]
        v = load_vec(b, X, e, dtype=spec.dtype, n=VEC)
        return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(VEC)]

    def scale_shift(ch):
        mean = b.global_load_f32(RunningMean, ch)
        var = b.global_load_f32(RunningVar, ch)
        g = load_scalar_as_f32(b, Gamma, ch, dtype=spec.dtype)
        bt = load_scalar_as_f32(b, Beta, ch, dtype=spec.dtype)
        sc = b.fmul(b.rsqrt(b.fadd(var, eps)), g)
        return sc, b.fsub(bt, b.fmul(mean, sc))

    # Phase 1: issue every group's index math, X load and parameter loads.
    groups = []
    for u in range(U):
        e0 = b.add(tile_base, b.mul(b.add(tid, b.const_i32(u * BS)), b.const_i32(VEC)))
        valid = b.cmp_lt(e0, total)
        es = b.smin(e0, last_start)  # clamped: loads always in-bounds
        row = b.div(es, HW)
        ch0 = b.sub(row, b.mul(b.div(row, p_c), p_c))
        xs = load_x(es)
        sc0, sh0 = scale_shift(ch0)
        if spec.aligned:
            lanes = None
        else:
            col0 = b.sub(es, b.mul(row, HW))
            ch1 = b.select(
                b.cmp_ge(ch0, c_last), b.const_i32(0), b.add(ch0, b.const_i32(1))
            )
            sc1, sh1 = scale_shift(ch1)
            lanes = (col0, sc1, sh1)
        groups.append((es, valid, xs, sc0, sh0, lanes))

    # Phase 2: compute and store.
    for es, valid, xs, sc0, sh0, lanes in groups:
        ys = []
        for i in range(VEC):
            if lanes is None:
                sc, sh = sc0, sh0
            else:
                col0, sc1, sh1 = lanes
                cross = b.cmp_ge(b.add(col0, b.const_i32(i)), HW)
                sc = b.select(cross, sc1, sc0)
                sh = b.select(cross, sh1, sh0)
            ys.append(b.fma(xs[i], sc, sh))
        with b.scf_if(valid):
            if VEC == 1:
                store_scalar_from_f32(b, Y, es, ys[0], dtype=spec.dtype)
            else:
                store_vec(b, Y, es, pack_f32_to(b, ys, dtype=spec.dtype), n=VEC)
    return b.kernel


# ===================================================================== #
# Dynamic (AOT-generic) cross-CTA split reduction, training forward
# ===================================================================== #
#
# Shape-generic counterpart of the static pair above. One compiled binary
# per ``(block_size, vec, fast_reduce)`` serves any runtime N/C/H/W *and*
# any split factors, because the split geometry is passed as kernargs:
#
#   grid = (C, SH, SN)      block_id_x = channel
#                           block_id_y = spatial split  (hw chunk ``hw_chunk``)
#                           block_id_z = batch split    (``n_chunk`` samples)
#
#   1. reduce kernel  -- one CTA per (channel, split): a partial
#      ``(mean, M2, count)`` triple for its (sample range x spatial chunk)
#      goes to ``Workspace[(channel*S + sn*SH + sh)*3 + {0,1,2}]``.
#   2. apply kernel   -- same grid: every CTA merges its channel's ``S``
#      partials (S <= block_size, one slot per thread), then normalizes its
#      own chunk. The CTA owning split (0, 0) also writes running stats and
#      the saved mean / inverse std. Merging redundantly per CTA keeps the
#      apply pass spread over C*S CTAs, instead of falling back to one CTA
#      per channel.
#
# The workspace needs no zero-init: every slot of every launched split is
# written exactly once by the reduce kernel. The pair is training-only; the
# reduce kernel is a reusable per-(channel, split) statistics primitive.


@dataclass(frozen=True)
class BatchNorm2DSplitDynSpec:
    """One dynamic split-reduction kernel pair (both kernels share the spec)."""

    block_size: int = 256
    vec: int = 8
    dtype: DType = "f16"
    fast_reduce: bool = False
    save_mean_invstd: bool = True
    wave_size: int = 64
    dynamic: bool = True
    name: str = "rocke_batchnorm2d_splitdyn"

    def kernel_name(self, kind: str = "reduce") -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            kind,
            "dyn",
            flags={"fr": self.fast_reduce},
        )


@dataclass(frozen=True)
class BatchNorm2DSplitPlan:
    """Runtime split geometry of one launch pair (kernel arguments + grid)."""

    n_chunk: int
    hw_chunk: int
    sn: int
    sh: int

    @property
    def splits(self) -> int:
        return self.sn * self.sh


# Per-CTA element counts ride in f32 partial counts: stay below 2**24.
_SPLIT_MAX_CTA_ELEMS = 1 << 24


def batchnorm2d_split_plan(
    spec: BatchNorm2DSplitDynSpec,
    n: int,
    h: int,
    w: int,
    *,
    sn_req: int,
    sh_req: int,
) -> "BatchNorm2DSplitPlan | None":
    """Split geometry for a request of ``sn_req`` batch splits and ``sh_req``
    spatial splits, or ``None`` when the result is not a valid launch.

    The spatial chunk is rounded up to a multiple of ``block_size * vec`` so
    every CTA's main sweep is fully vectorized; the effective split counts are
    recomputed from the rounded chunk sizes so no launched CTA is empty.
    """
    hw = int(h) * int(w)
    g = spec.block_size * spec.vec
    if sn_req < 1 or sh_req < 1 or n <= 0 or hw <= 0:
        return None
    hw_chunk = -(-(-(-hw // sh_req)) // g) * g
    sh = -(-hw // hw_chunk)
    n_chunk = -(-n // sn_req)
    sn = -(-n // n_chunk)
    plan = BatchNorm2DSplitPlan(n_chunk=n_chunk, hw_chunk=hw_chunk, sn=sn, sh=sh)
    if plan.splits < 2 or plan.splits > spec.block_size:
        return None
    if n_chunk * min(hw_chunk, hw) >= _SPLIT_MAX_CTA_ELEMS:
        return None
    return plan


def batchnorm2d_split_dyn_workspace_floats(c: int, plan: BatchNorm2DSplitPlan) -> int:
    return int(c) * plan.splits * 3


def batchnorm2d_split_dyn_grid(
    c: int, plan: BatchNorm2DSplitPlan
) -> Tuple[int, int, int]:
    return (int(c), plan.sh, plan.sn)


def is_valid_split_dyn_spec(
    spec: BatchNorm2DSplitDynSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, "fast_reduce requires block_size divisible by wave_size"
    # 16-bit data: 4 or 8 elements per thread; f32 data: 2 or 4 (8 / 16 B accesses).
    _vecs = (2, 4) if spec.dtype == "f32" else (4, 8)
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=None,
            allowed_dtypes=("f16", "bf16", "f32"),
            allowed_vecs=_vecs,
        )
    )
    if not ok:
        return False, why
    if spec.block_size > target.max_threads_per_block:
        return False, "block_size exceeds max_threads_per_block"
    if not target.fits_lds(3 * spec.block_size * 4):
        return False, "LDS budget exceeded"
    return True, ""


def _split_dyn_block_reduce(b, spec, mean_p, m2_p, count_p, lds, tid):
    lds_mean, lds_m2, lds_count = lds
    if spec.fast_reduce:
        return welford_block_reduce_stable_wave_prologue(
            b,
            mean_p,
            m2_p,
            count_p,
            lds_mean,
            lds_m2,
            lds_count,
            tid,
            block_size=spec.block_size,
            wave_size=spec.wave_size,
        )
    return welford_block_reduce_stable(
        b,
        mean_p,
        m2_p,
        count_p,
        lds_mean,
        lds_m2,
        lds_count,
        tid,
        block_size=spec.block_size,
    )


def _split_dyn_geometry(b, p_n, p_nch, p_hwch, HW):
    """Per-CTA ranges from the grid ids:
    ``(sh_id, sn_id, hw_lo_safe, seg_len, n_lo, n_hi, n_cnt)``.

    ``hw_lo_safe`` is clamped into ``[0, HW)`` so every address derived from it
    stays inside the tensor even for an (unlaunched) empty split; ``seg_len``
    and ``n_cnt`` are clamped at zero.
    """
    sh_id = b.block_id_y()
    sn_id = b.block_id_z()
    hw_lo = b.mul(sh_id, p_hwch)
    hw_lo_safe = b.smin(hw_lo, b.sub(HW, b.const_i32(1)))
    seg_len = b.smax(b.smin(b.sub(HW, hw_lo), p_hwch), b.const_i32(0))
    n_lo = b.mul(sn_id, p_nch)
    n_hi = b.smin(p_n, b.add(n_lo, p_nch))
    n_cnt = b.smax(b.sub(n_hi, n_lo), b.const_i32(0))
    return sh_id, sn_id, hw_lo_safe, seg_len, n_lo, n_hi, n_cnt


def build_batchnorm2d_split_dyn_reduce(spec: BatchNorm2DSplitDynSpec) -> KernelDef:
    """Reduce kernel of the dynamic split pair (see the section comment)."""
    ok, why = is_valid_split_dyn_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d dynamic split spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    cap = 1

    b = IRBuilder(spec.kernel_name("reduce"))
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Workspace = b.param("Workspace", PtrType(F32, "global"), noalias=True)
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)
    p_nch = b.param("NChunk", I32)
    p_hwch = b.param("HwChunk", I32)
    p_sh = b.param("SplitHw", I32)
    p_sn = b.param("SplitN", I32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    channel = b.block_id_x()
    sh_id, sn_id, hw_lo, seg_len, n_lo, n_hi, n_cnt = _split_dyn_geometry(
        b, p_n, p_nch, p_hwch, HW
    )
    x_view = make_global_view(X, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_mean", "lds_m2", "lds_count")
    )
    seg_last = b.smax(b.sub(seg_len, b.const_i32(1)), b.const_i32(0))

    def pass1_body(_n_off, x_scalars, iter_vars):
        sum_p, sumsq_p = iter_vars
        sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
        sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
        sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))
        return [sum_p, sumsq_p]

    def outer_body(n, outer_iter_vars):
        sum_p, sumsq_p, tail_count_p = outer_iter_vars
        row_n = b.add(b.mul(n, p_c), channel)
        x_tile_n = make_tile_window(x_view, lengths=(1, 1), origin=(row_n, hw_lo))
        (sum_p, sumsq_p), main_elems, _tail = sweep_row_chunks_dynamic(
            b,
            x_tile_n,
            tid=tid,
            block_size=BS,
            vec=VEC,
            length=seg_len,
            iter_args=[("sum_p_hw", sum_p), ("sumsq_p_hw", sumsq_p)],
            body=pass1_body,
            iv_name="hw0",
        )
        for tc in range(VEC):
            idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
            valid = b.cmp_lt(idx_col, seg_len)
            idx_safe = b.smin(idx_col, seg_last)
            flat = b.add(b.add(b.mul(row_n, HW), hw_lo), idx_safe)
            x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
            x_masked = b.select(valid, x_raw, b.const_f32(0.0))
            valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
            sum_p = b.fadd(sum_p, x_masked)
            sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
            tail_count_p = b.fadd(tail_count_p, valid_f)
        return [sum_p, sumsq_p, tail_count_p]

    for_op = b.scf_for_iter(
        n_lo,
        n_hi,
        b.const_i32(1),
        [
            ("sum_p", b.const_f32(0.0)),
            ("sumsq_p", b.const_f32(0.0)),
            ("tail_count_p", b.const_f32(0.0)),
        ],
        iv_name="n0",
    )
    with for_op as (n_iv, outer_iter_vars):
        b.scf_yield(*outer_body(n_iv, outer_iter_vars))
    sum_p, sumsq_p, tail_count_p = for_op.results

    trip_count = b.div(seg_len, b.const_i32(BS * VEC))
    main_per_thread_total = b.mul(n_cnt, b.mul(trip_count, b.const_i32(VEC)))
    count_val = b.fadd(b.sitofp_f32(main_per_thread_total), tail_count_p)
    count_safe = b.fmax(count_val, b.const_f32(1.0))
    mean_p = b.fdiv(sum_p, count_safe)
    m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

    mean, var = _split_dyn_block_reduce(b, spec, mean_p, m2_p, count_val, lds, tid)

    total_count = b.sitofp_f32(b.mul(n_cnt, seg_len))
    is_empty = b.fcmp("oeq", total_count, b.const_f32(0.0))
    mean_out = b.select(is_empty, b.const_f32(0.0), mean)
    m2_out = b.select(is_empty, b.const_f32(0.0), b.fmul(var, total_count))
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        split_idx = b.add(b.mul(sn_id, p_sh), sh_id)
        splits = b.mul(p_sh, p_sn)
        slot = b.mul(b.add(b.mul(channel, splits), split_idx), b.const_i32(3))
        b.global_store(Workspace, slot, mean_out)
        b.global_store(Workspace, b.add(slot, b.const_i32(1)), m2_out)
        b.global_store(Workspace, b.add(slot, b.const_i32(2)), total_count)
    return b.kernel


def build_batchnorm2d_split_dyn_apply(spec: BatchNorm2DSplitDynSpec) -> KernelDef:
    """Apply kernel of the dynamic split pair (see the section comment)."""
    ok, why = is_valid_split_dyn_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d dynamic split spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    cap = 1

    b = IRBuilder(spec.kernel_name("apply"))
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
    p_nch = b.param("NChunk", I32)
    p_hwch = b.param("HwChunk", I32)
    p_sh = b.param("SplitHw", I32)
    p_sn = b.param("SplitN", I32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    channel = b.block_id_x()
    sh_id, sn_id, hw_lo, seg_len, n_lo, n_hi, n_cnt = _split_dyn_geometry(
        b, p_n, p_nch, p_hwch, HW
    )
    x_view = make_global_view(X, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    y_view = make_global_view(Y, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_mean", "lds_m2", "lds_count")
    )

    # Merge this channel's partials: thread t owns split slot t.
    splits = b.mul(p_sh, p_sn)
    has_slot = b.cmp_lt(tid, splits)
    slot_tid = b.select(has_slot, tid, b.const_i32(0))
    slot = b.mul(b.add(b.mul(channel, splits), slot_tid), b.const_i32(3))
    mean_raw = b.global_load_f32(Workspace, slot)
    m2_raw = b.global_load_f32(Workspace, b.add(slot, b.const_i32(1)))
    count_raw = b.global_load_f32(Workspace, b.add(slot, b.const_i32(2)))
    mean_p = b.select(has_slot, mean_raw, b.const_f32(0.0))
    m2_p = b.select(has_slot, m2_raw, b.const_f32(0.0))
    count_p = b.select(has_slot, count_raw, b.const_f32(0.0))
    mean, var = _split_dyn_block_reduce(b, spec, mean_p, m2_p, count_p, lds, tid)

    owner = b.land(b.cmp_eq(sh_id, b.const_i32(0)), b.cmp_eq(sn_id, b.const_i32(0)))
    owner = b.land(owner, b.cmp_eq(tid, b.const_i32(0)))
    with b.scf_if(owner):
        old_mean = b.global_load_f32(RunningMean, channel)
        old_var = b.global_load_f32(RunningVar, channel)
        one_minus_mom = b.fsub(b.const_f32(1.0), momentum)
        new_running_mean = b.fadd(
            b.fmul(one_minus_mom, old_mean), b.fmul(momentum, mean)
        )
        m_total = b.sitofp_f32(b.mul(p_n, HW))
        unbias = b.fdiv(
            m_total, b.fmax(b.fsub(m_total, b.const_f32(1.0)), b.const_f32(1.0))
        )
        new_running_var = b.fadd(
            b.fmul(one_minus_mom, old_var), b.fmul(momentum, b.fmul(var, unbias))
        )
        b.global_store(RunningMean, channel, new_running_mean)
        b.global_store(RunningVar, channel, new_running_var)
        if spec.save_mean_invstd:
            inv_std_save = b.rsqrt(b.fadd(var, eps))
            store_scalar_from_f32(b, Mean, channel, mean, dtype=spec.dtype)
            store_scalar_from_f32(b, InvStd, channel, inv_std_save, dtype=spec.dtype)

    inv_std = b.rsqrt(b.fadd(var, eps))
    gamma_v = load_scalar_as_f32(b, Gamma, channel, dtype=spec.dtype)
    beta_v = load_scalar_as_f32(b, Beta, channel, dtype=spec.dtype)
    scale = b.fmul(inv_std, gamma_v)

    trip_count = b.div(seg_len, b.const_i32(BS * VEC))
    main_elems = b.mul(trip_count, b.const_i32(BS * VEC))

    outer_for = b.scf_for(n_lo, n_hi, b.const_i32(1), iv_name="n1")
    with outer_for as n_iv:
        row_n = b.add(b.mul(n_iv, p_c), channel)
        x_tile_n = make_tile_window(x_view, lengths=(1, 1), origin=(row_n, hw_lo))
        y_tile_n = make_tile_window(y_view, lengths=(1, 1), origin=(row_n, hw_lo))

        def body(n_off, _x_tile_n=x_tile_n):
            x_scalars = _x_tile_n.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
            return [
                b.fadd(b.fmul(b.fsub(x_scalars[i], mean), scale), beta_v)
                for i in range(VEC)
            ]

        pass2_row_chunks_dynamic(
            b,
            y_tile_n,
            tid=tid,
            block_size=BS,
            vec=VEC,
            main_elems_per_row=main_elems,
            trip_count=trip_count,
            body=body,
            iv_name="hw1",
        )
        for tc in range(VEC):
            idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
            with b.scf_if(b.cmp_lt(idx_col, seg_len)):
                flat = b.add(b.add(b.mul(row_n, HW), hw_lo), idx_col)
                x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), scale), beta_v)
                store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)
    return b.kernel


# ===================================================================== #
# Backward (spatial mode, saved mean / inverse std)
# ===================================================================== #
#
# Per channel, with ``xhat = (x - mean) * invstd`` and ``M = N*H*W``:
#
#     dbias[c]  = sum(dy)
#     dscale[c] = sum(dy * xhat)
#     dx        = scale * invstd / M * (M * dy - dbias - xhat * dscale)
#               = a * dy - k2 * (x - mean) - k3
#       with a = scale * invstd, k2 = a * invstd * dscale / M, k3 = a * dbias / M
#
# Inputs ``X``/``dY``/``dX`` use the data dtype; ``Scale``, the saved
# ``Mean``/``InvStd`` and the outputs ``dScale``/``dBias`` are always f32 (the
# MIOpen driver's ``bnormfp16``/``bnormbfp16`` accumulation type).  Both
# reductions are plain sums, so partial results of any decomposition merge by
# addition (no Welford state, no counts).
#
#   * single launch (:func:`build_batchnorm2d_bwd_dynamic`): one CTA per
#     ``channels_per_block`` channels, same thread mapping as the forward
#     dynamic path (``n_groups`` splits the CTA over samples);
#   * two launches (:func:`build_batchnorm2d_bwd_split_reduce` /
#     ``_apply``): the forward split geometry; the reduce kernel stores
#     per-(channel, split) partial sums, the apply kernel merges them.
#
# Kernel names carry ``bwd``, so cache keys can never collide with a forward
# kernel.


@dataclass(frozen=True)
class BatchNorm2DBwdSpec:
    """One dynamic single-launch backward instance (n/c/h/w are runtime)."""

    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    channels_per_block: int = 1
    n_groups: int = 1
    fast_reduce: bool = False
    wave_size: int = 64
    dynamic: bool = True
    name: str = "rocke_batchnorm2d_bwd"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            f"cpb{self.channels_per_block}" if self.channels_per_block != 1 else "",
            f"ng{self.n_groups}" if self.n_groups != 1 else "",
            flags={"fr": self.fast_reduce, "dyn": self.dynamic},
        )


def is_valid_bwd_spec(
    spec: BatchNorm2DBwdSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.channels_per_block <= 0 or spec.n_groups <= 0:
        return False, "channels_per_block / n_groups must be positive"
    if spec.block_size % spec.n_groups != 0:
        return False, "block_size not divisible by n_groups"
    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, "fast_reduce requires block_size divisible by wave_size"
    _vecs = (2, 4) if spec.dtype == "f32" else (2, 4, 8)
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            allowed_dtypes=("f16", "bf16", "f32"),
            allowed_vecs=_vecs,
        )
    )
    if not ok:
        return False, why
    if spec.block_size > target.max_threads_per_block:
        return False, "block_size exceeds max_threads_per_block"
    if not target.fits_lds(2 * spec.block_size * 4):
        return False, "LDS budget exceeded"
    return True, ""


def batchnorm2d_bwd_grid(c: int, spec: BatchNorm2DBwdSpec) -> Tuple[int, int, int]:
    return (int(c) // spec.channels_per_block, 1, 1)


def _bwd_block_sums(b, spec, s1, s2, lds, tid):
    """Block-wide sums of two per-thread f32 partials (result on every lane)."""
    lds_a, lds_c = lds
    if spec.fast_reduce:
        t1 = block_lds_reduce_with_wave_prologue(
            b, s1, lds_a, tid, block_size=spec.block_size, wave_size=spec.wave_size
        )
        t2 = block_lds_reduce_with_wave_prologue(
            b, s2, lds_c, tid, block_size=spec.block_size, wave_size=spec.wave_size
        )
        return t1, t2
    return block_lds_reduce_pair(
        b, s1, s2, lds_a, lds_c, tid, block_size=spec.block_size
    )


def _bwd_params(b, spec):
    """Shared leading kernel params (X, dY, Scale, Mean, InvStd) and dims."""
    io_ty = io_ir_type(spec.dtype)
    P = {}
    P["X"] = b.param(
        "X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    P["dY"] = b.param(
        "dY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    return P


def _bwd_accumulate(
    b,
    spec,
    *,
    X,
    dY,
    x_view,
    dy_view,
    HW,
    mean_c,
    invstd_c,
    tid_row,
    threads,
    n_begin,
    n_end,
    row_of,
    hw_lo,
    length,
    seg_last,
    iv_prefix,
):
    """Per-thread partial sums ``(sum(dy), sum(dy * xhat))`` over the rows
    ``n_begin .. n_end`` (runtime loop) of one channel, each swept over
    ``[hw_lo, hw_lo + length)`` with ``threads`` threads per row."""
    VEC = spec.vec
    dtype = spec.dtype

    def outer_body(og, iv):
        s1, s2 = iv
        row_n = row_of(og)
        x_tile = make_tile_window(x_view, lengths=(1, 1), origin=(row_n, hw_lo))
        dy_tile = make_tile_window(dy_view, lengths=(1, 1), origin=(row_n, hw_lo))

        def body(n_off, x_scalars, iter_vars, _dy_tile=dy_tile):
            a1, a2 = iter_vars
            dys = _dy_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
            prods = [
                b.fmul(dys[i], b.fmul(b.fsub(x_scalars[i], mean_c), invstd_c))
                for i in range(VEC)
            ]
            a1 = b.fadd(a1, tree_reduce(b, b.fadd, list(dys)))
            a2 = b.fadd(a2, tree_reduce(b, b.fadd, prods))
            return [a1, a2]

        (s1, s2), main_elems, _tail = sweep_row_chunks_dynamic(
            b,
            x_tile,
            tid=tid_row,
            block_size=threads,
            vec=VEC,
            length=length,
            iter_args=[("s_dy_in", s1), ("s_dyx_in", s2)],
            body=body,
            iv_name=f"{iv_prefix}_hw",
        )
        for tc in range(VEC):
            idx_col = b.add(b.add(main_elems, b.const_i32(tc * threads)), tid_row)
            valid = b.cmp_lt(idx_col, length)
            idx_safe = b.smin(idx_col, seg_last)
            flat = b.add(b.add(b.mul(row_n, HW), hw_lo), idx_safe)
            x_raw = load_scalar_as_f32(b, X, flat, dtype=dtype)
            dy_raw = load_scalar_as_f32(b, dY, flat, dtype=dtype)
            dy_m = b.select(valid, dy_raw, b.const_f32(0.0))
            s1 = b.fadd(s1, dy_m)
            s2 = b.fadd(s2, b.fmul(dy_m, b.fmul(b.fsub(x_raw, mean_c), invstd_c)))
        return [s1, s2]

    for_op = b.scf_for_iter(
        n_begin,
        n_end,
        b.const_i32(1),
        [("s_dy", b.const_f32(0.0)), ("s_dyx", b.const_f32(0.0))],
        iv_name=f"{iv_prefix}_n",
    )
    with for_op as (n_iv, iv):
        b.scf_yield(*outer_body(n_iv, iv))
    return for_op.results


def _bwd_coefs(b, s1, s2, scale, mean, invstd, m_f):
    """``dx = a * dy - k2 * (x - mean) - k3``; returns ``(a, k2, k3)``."""
    inv_m = b.fdiv(b.const_f32(1.0), m_f)
    a = b.fmul(scale, invstd)
    k2 = b.fmul(b.fmul(a, invstd), b.fmul(s2, inv_m))
    k3 = b.fmul(a, b.fmul(s1, inv_m))
    return a, k2, k3


def _bwd_apply_rows(
    b,
    spec,
    *,
    X,
    dY,
    dX,
    x_view,
    dy_view,
    dx_view,
    HW,
    mean_c,
    coefs,
    tid_row,
    threads,
    n_begin,
    n_end,
    row_of,
    hw_lo,
    length,
    iv_prefix,
):
    """Write ``dx`` for the rows ``n_begin .. n_end`` of one channel."""
    VEC = spec.vec
    a, k2, k3 = coefs
    trip_count = b.div(length, b.const_i32(threads * VEC))
    main_elems = b.mul(trip_count, b.const_i32(threads * VEC))

    def dx_of(x, dy):
        return b.fsub(b.fsub(b.fmul(a, dy), b.fmul(k2, b.fsub(x, mean_c))), k3)

    outer = b.scf_for(n_begin, n_end, b.const_i32(1), iv_name=f"{iv_prefix}_n2")
    with outer as og:
        row_n = row_of(og)
        x_tile = make_tile_window(x_view, lengths=(1, 1), origin=(row_n, hw_lo))
        dy_tile = make_tile_window(dy_view, lengths=(1, 1), origin=(row_n, hw_lo))
        dx_tile = make_tile_window(dx_view, lengths=(1, 1), origin=(row_n, hw_lo))

        def body(n_off, _x=x_tile, _dy=dy_tile):
            xs = _x.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
            dys = _dy.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
            return [dx_of(xs[i], dys[i]) for i in range(VEC)]

        pass2_row_chunks_dynamic(
            b,
            dx_tile,
            tid=tid_row,
            block_size=threads,
            vec=VEC,
            main_elems_per_row=main_elems,
            trip_count=trip_count,
            body=body,
            iv_name=f"{iv_prefix}_hw2",
        )
        for tc in range(VEC):
            idx_col = b.add(b.add(main_elems, b.const_i32(tc * threads)), tid_row)
            with b.scf_if(b.cmp_lt(idx_col, length)):
                flat = b.add(b.add(b.mul(row_n, HW), hw_lo), idx_col)
                x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
                dy_raw = load_scalar_as_f32(b, dY, flat, dtype=spec.dtype)
                store_scalar_from_f32(
                    b, dX, flat, dx_of(x_raw, dy_raw), dtype=spec.dtype
                )


def build_batchnorm2d_bwd_dynamic(spec: BatchNorm2DBwdSpec) -> KernelDef:
    """Single-launch dynamic backward: one CTA per ``channels_per_block``
    channels; ``n_groups`` splits the CTA's threads over samples."""
    ok, why = is_valid_bwd_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d backward spec: {why}")
    io_ty = io_ir_type(spec.dtype)
    BS, CPB, NG = spec.block_size, spec.channels_per_block, spec.n_groups

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS
    P = _bwd_params(b, spec)
    X, dY = P["X"], P["dY"]
    Scale = b.param("Scale", PtrType(F32, "global"), noalias=True, readonly=True)
    Mean = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    InvStd = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    dX = b.param("dX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    dScale = b.param("dScale", PtrType(F32, "global"), noalias=True, writeonly=True)
    dBias = b.param("dBias", PtrType(F32, "global"), noalias=True, writeonly=True)
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    base_channel = b.block_id_x()
    cap = 1
    x_view = make_global_view(X, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    dy_view = make_global_view(dY, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    dx_view = make_global_view(dX, shape=(cap, cap), dtype=io_ty, strides=(HW, 1))
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_s1", "lds_s2")
    )
    m_f = b.sitofp_f32(b.mul(p_n, HW))
    hw_lo = b.const_i32(0)
    seg_last = b.sub(HW, b.const_i32(1))

    hw_threads = BS // NG
    if NG == 1:
        tid_row, lidihw, n_trip = tid, b.const_i32(0), p_n
    else:
        tid_row = b.mod(tid, b.const_i32(hw_threads))
        lidihw = b.div(tid, b.const_i32(hw_threads))
        n_trip = b.div(p_n, b.const_i32(NG))

    for cb in range(CPB):
        if CPB == 1:
            channel = base_channel
        else:
            channel = b.add(b.mul(base_channel, b.const_i32(CPB)), b.const_i32(cb))
            if cb > 0:
                b.sync()
        mean_c = b.global_load_f32(Mean, channel)
        invstd_c = b.global_load_f32(InvStd, channel)
        scale_c = b.global_load_f32(Scale, channel)

        def row_of(og, _ch=channel):
            n_rt = og if NG == 1 else b.add(b.mul(og, b.const_i32(NG)), lidihw)
            return b.add(b.mul(n_rt, p_c), _ch)

        s1, s2 = _bwd_accumulate(
            b,
            spec,
            X=X,
            dY=dY,
            x_view=x_view,
            dy_view=dy_view,
            HW=HW,
            mean_c=mean_c,
            invstd_c=invstd_c,
            tid_row=tid_row,
            threads=hw_threads,
            n_begin=b.const_i32(0),
            n_end=n_trip,
            row_of=row_of,
            hw_lo=hw_lo,
            length=HW,
            seg_last=seg_last,
            iv_prefix="b1",
        )
        t1, t2 = _bwd_block_sums(b, spec, s1, s2, lds, tid)
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(dScale, channel, t2)
            b.global_store(dBias, channel, t1)
        coefs = _bwd_coefs(b, t1, t2, scale_c, mean_c, invstd_c, m_f)
        _bwd_apply_rows(
            b,
            spec,
            X=X,
            dY=dY,
            dX=dX,
            x_view=x_view,
            dy_view=dy_view,
            dx_view=dx_view,
            HW=HW,
            mean_c=mean_c,
            coefs=coefs,
            tid_row=tid_row,
            threads=hw_threads,
            n_begin=b.const_i32(0),
            n_end=n_trip,
            row_of=row_of,
            hw_lo=hw_lo,
            length=HW,
            iv_prefix="b1",
        )
    return b.kernel


@dataclass(frozen=True)
class BatchNorm2DBwdSplitSpec(BatchNorm2DSplitDynSpec):
    """Dynamic split backward pair (same geometry/plan machinery as the
    forward split; only the kernel name differs)."""

    name: str = "rocke_batchnorm2d_bwd_splitdyn"


def batchnorm2d_bwd_split_workspace_floats(c: int, plan: BatchNorm2DSplitPlan) -> int:
    return int(c) * plan.splits * 2


class _SumSpec:
    """Adapter giving the split spec the fields ``_bwd_block_sums`` reads."""

    def __init__(self, s):
        self.block_size, self.fast_reduce, self.wave_size = (
            s.block_size,
            s.fast_reduce,
            s.wave_size,
        )


def build_batchnorm2d_bwd_split_reduce(spec: BatchNorm2DBwdSplitSpec) -> KernelDef:
    """Reduce kernel: per-(channel, split) partial ``(sum(dy), sum(dy * xhat))``."""
    ok, why = is_valid_split_dyn_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d backward split spec: {why}")
    io_ty = io_ir_type(spec.dtype)
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name("reduce"))
    b.kernel.attrs["max_workgroup_size"] = BS
    P = _bwd_params(b, spec)
    X, dY = P["X"], P["dY"]
    Mean = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    InvStd = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    Workspace = b.param("Workspace", PtrType(F32, "global"), noalias=True)
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)
    p_nch = b.param("NChunk", I32)
    p_hwch = b.param("HwChunk", I32)
    p_sh = b.param("SplitHw", I32)
    p_sn = b.param("SplitN", I32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    channel = b.block_id_x()
    sh_id, sn_id, hw_lo, seg_len, n_lo, n_hi, n_cnt = _split_dyn_geometry(
        b, p_n, p_nch, p_hwch, HW
    )
    x_view = make_global_view(X, shape=(1, 1), dtype=io_ty, strides=(HW, 1))
    dy_view = make_global_view(dY, shape=(1, 1), dtype=io_ty, strides=(HW, 1))
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_s1", "lds_s2")
    )
    seg_last = b.smax(b.sub(seg_len, b.const_i32(1)), b.const_i32(0))
    mean_c = b.global_load_f32(Mean, channel)
    invstd_c = b.global_load_f32(InvStd, channel)

    s1, s2 = _bwd_accumulate(
        b,
        _VecShim(spec),
        X=X,
        dY=dY,
        x_view=x_view,
        dy_view=dy_view,
        HW=HW,
        mean_c=mean_c,
        invstd_c=invstd_c,
        tid_row=tid,
        threads=BS,
        n_begin=n_lo,
        n_end=n_hi,
        row_of=lambda n: b.add(b.mul(n, p_c), channel),
        hw_lo=hw_lo,
        length=seg_len,
        seg_last=seg_last,
        iv_prefix="r",
    )
    t1, t2 = _bwd_block_sums(b, _SumSpec(spec), s1, s2, lds, tid)
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        split_idx = b.add(b.mul(sn_id, p_sh), sh_id)
        splits = b.mul(p_sh, p_sn)
        slot = b.mul(b.add(b.mul(channel, splits), split_idx), b.const_i32(2))
        b.global_store(Workspace, slot, t1)
        b.global_store(Workspace, b.add(slot, b.const_i32(1)), t2)
    return b.kernel


class _VecShim:
    """Spec view exposing ``vec`` / ``dtype`` to the shared sweep helpers."""

    def __init__(self, s):
        self.vec, self.dtype = s.vec, s.dtype


def build_batchnorm2d_bwd_split_apply(spec: BatchNorm2DBwdSplitSpec) -> KernelDef:
    """Apply kernel: merge a channel's partial sums, write ``dx`` for this CTA's
    chunk; split (0, 0) also writes ``dScale`` / ``dBias``."""
    ok, why = is_valid_split_dyn_spec(spec)
    if not ok:
        raise ValueError(f"invalid batchnorm2d backward split spec: {why}")
    io_ty = io_ir_type(spec.dtype)
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name("apply"))
    b.kernel.attrs["max_workgroup_size"] = BS
    P = _bwd_params(b, spec)
    X, dY = P["X"], P["dY"]
    Scale = b.param("Scale", PtrType(F32, "global"), noalias=True, readonly=True)
    Mean = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    InvStd = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    Workspace = b.param(
        "Workspace", PtrType(F32, "global"), noalias=True, readonly=True
    )
    dX = b.param("dX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    dScale = b.param("dScale", PtrType(F32, "global"), noalias=True, writeonly=True)
    dBias = b.param("dBias", PtrType(F32, "global"), noalias=True, writeonly=True)
    p_n = b.param("N", I32)
    p_c = b.param("C", I32)
    p_h = b.param("H", I32)
    p_w = b.param("W", I32)
    p_nch = b.param("NChunk", I32)
    p_hwch = b.param("HwChunk", I32)
    p_sh = b.param("SplitHw", I32)
    p_sn = b.param("SplitN", I32)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    channel = b.block_id_x()
    sh_id, sn_id, hw_lo, seg_len, n_lo, n_hi, n_cnt = _split_dyn_geometry(
        b, p_n, p_nch, p_hwch, HW
    )
    x_view = make_global_view(X, shape=(1, 1), dtype=io_ty, strides=(HW, 1))
    dy_view = make_global_view(dY, shape=(1, 1), dtype=io_ty, strides=(HW, 1))
    dx_view = make_global_view(dX, shape=(1, 1), dtype=io_ty, strides=(HW, 1))
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_s1", "lds_s2")
    )

    splits = b.mul(p_sh, p_sn)
    has_slot = b.cmp_lt(tid, splits)
    slot_tid = b.select(has_slot, tid, b.const_i32(0))
    slot = b.mul(b.add(b.mul(channel, splits), slot_tid), b.const_i32(2))
    p1 = b.select(has_slot, b.global_load_f32(Workspace, slot), b.const_f32(0.0))
    p2 = b.select(
        has_slot,
        b.global_load_f32(Workspace, b.add(slot, b.const_i32(1))),
        b.const_f32(0.0),
    )
    t1, t2 = _bwd_block_sums(b, _SumSpec(spec), p1, p2, lds, tid)

    mean_c = b.global_load_f32(Mean, channel)
    invstd_c = b.global_load_f32(InvStd, channel)
    scale_c = b.global_load_f32(Scale, channel)
    owner = b.land(b.cmp_eq(sh_id, b.const_i32(0)), b.cmp_eq(sn_id, b.const_i32(0)))
    owner = b.land(owner, b.cmp_eq(tid, b.const_i32(0)))
    with b.scf_if(owner):
        b.global_store(dScale, channel, t2)
        b.global_store(dBias, channel, t1)
    m_f = b.sitofp_f32(b.mul(p_n, HW))
    coefs = _bwd_coefs(b, t1, t2, scale_c, mean_c, invstd_c, m_f)
    _bwd_apply_rows(
        b,
        _VecShim(spec),
        X=X,
        dY=dY,
        dX=dX,
        x_view=x_view,
        dy_view=dy_view,
        dx_view=dx_view,
        HW=HW,
        mean_c=mean_c,
        coefs=coefs,
        tid_row=tid,
        threads=BS,
        n_begin=n_lo,
        n_end=n_hi,
        row_of=lambda n: b.add(b.mul(n, p_c), channel),
        hw_lo=hw_lo,
        length=seg_len,
        iv_prefix="a",
    )
    return b.kernel


# ===================================================================== #
# NHWC (channels-last) kernels: infer, training forward, backward
# ===================================================================== #
#
# Memory is ``R = N*H*W`` rows of ``C`` contiguous channels. A CTA owns a
# channel tile of ``CT = tx * vec`` channels and a chunk of rows; thread
# ``(tx, ty)`` keeps ``vec`` contiguous channels (one 16 B access) and walks
# the rows ``ty, ty + TY, ...`` of the chunk (``TY = block_size / tx``),
# ``unroll`` rows per loop trip.  Grid = ``(row splits, channel tiles, 1)``.
#
#   * infer: per-thread scale / shift folded once from the four per-channel
#     vectors, then a streaming ``y = x * scale + shift`` over the chunk;
#   * training / backward, single launch ("fused"): one CTA per channel tile
#     owns every row: statistics pass, tile reduction over ``ty`` in LDS,
#     then the apply pass;
#   * training / backward, two launches ("reduce" + "apply"): the reduce
#     kernel stores one partial pair per (row split, channel) in a planar
#     workspace ``[split][which][channel]``, the apply kernel merges the
#     partials of its channel tile (redundantly per CTA) and writes the output.
#
# Partials are plain sums, so a merge is an addition and the workspace needs
# no initialisation. Training sums are shifted by the channel's first sample
# to keep ``E[x^2] - E[x]^2`` well conditioned. Kernel names carry ``nhwc``.

_NHWC_KINDS = {
    "infer": ("infer",),
    "train": ("fused", "reduce", "apply"),
    "bwd": ("fused", "reduce", "apply"),
}


@dataclass(frozen=True)
class BatchNorm2DNhwcSpec:
    """One dynamic NHWC instance (all of N / C / H / W are runtime)."""

    mode: str = "train"  # "infer" | "train" | "bwd"
    block_size: int = 256
    tx: int = 8
    vec: int = 8
    unroll: int = 4
    dtype: DType = "f16"
    wave_size: int = 64
    dynamic: bool = True
    name: str = "rocke_batchnorm2d_nhwc"

    @property
    def ty(self) -> int:
        return self.block_size // self.tx

    @property
    def tile_channels(self) -> int:
        return self.tx * self.vec

    def kernel_name(self, kind: str = "fused") -> str:
        return kernel_name_join(
            self.name,
            self.mode,
            kind,
            self.dtype,
            f"b{self.block_size}",
            f"tx{self.tx}",
            f"v{self.vec}",
            f"u{self.unroll}",
        )


def is_valid_nhwc_spec(
    spec: BatchNorm2DNhwcSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.mode not in _NHWC_KINDS:
        return False, f"unknown mode {spec.mode!r}"
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.vec not in (2, 4, 8):
        return False, "vec must be 2, 4 or 8"
    if spec.unroll not in (1, 2, 4, 8, 16):
        return False, "unroll must be 1/2/4/8/16"
    if spec.tx <= 0 or spec.tx & (spec.tx - 1) or spec.block_size % spec.tx != 0:
        return False, "tx must be a power of two dividing block_size"
    ty = spec.block_size // spec.tx
    if ty & (ty - 1):
        return False, "block_size / tx must be a power of two"
    if (
        not 0 < spec.block_size <= target.max_threads_per_block
        or spec.block_size % spec.wave_size
    ):
        return False, "block_size must be a wave multiple within the device limit"
    if not target.fits_lds(2 * spec.block_size * spec.vec * 4):
        return False, "LDS budget exceeded"
    return True, ""


def nhwc_admits(
    spec: BatchNorm2DNhwcSpec, n: int, c: int, h: int, w: int
) -> Tuple[bool, str]:
    """Runtime-shape admission of one NHWC instance."""
    rows = int(n) * int(h) * int(w)
    if rows <= 0 or c <= 0:
        return False, "empty tensor"
    if c % spec.vec != 0:
        return False, "C not a multiple of vec"
    if spec.tile_channels // 2 >= c:
        return False, "channel tile more than twice C"
    if rows * c + spec.block_size * spec.vec * spec.unroll >= 2**31:
        return False, "element count out of i32 range"
    return True, ""


@dataclass(frozen=True)
class BatchNorm2DNhwcPlan:
    """Runtime geometry of one NHWC launch (kernel arguments + grid)."""

    rows_per_cta: int
    splits: int
    tiles: int

    @property
    def ctas(self) -> int:
        return self.splits * self.tiles


NHWC_MAX_SPLITS = 1024
NHWC_MIN_SPLIT_CTAS = 128
NHWC_MIN_FUSED_TILES = 4
NHWC_FINE_RPC = (
    True  # also try rows-per-CTA at 1.25x, 1.5x and 1.75x of each power-of-two step
)


def _nhwc_rpc_steps(rpc: int, base: int) -> Tuple[int, ...]:
    """Rows-per-CTA values between ``rpc`` and ``2 * rpc``: the power of two
    itself plus finer steps that stay multiples of ``base``."""
    if not NHWC_FINE_RPC:
        return (rpc,)
    steps = [rpc]
    for num in (5, 6, 7):
        r = rpc * num // 4
        if r % base == 0 and r != rpc:
            steps.append(r)
    return tuple(steps)


def batchnorm2d_nhwc_plans(
    spec: BatchNorm2DNhwcSpec, variant: str, n: int, c: int, h: int, w: int
) -> Tuple[BatchNorm2DNhwcPlan, ...]:
    """Valid launch geometries of ``variant`` (``"infer"``, ``"fused"`` or
    ``"split"``) for one shape: rows per CTA are a power-of-two multiple of
    ``ty * unroll``, so no launched CTA is empty."""
    ok, _ = nhwc_admits(spec, n, c, h, w)
    if not ok:
        return ()
    rows = int(n) * int(h) * int(w)
    tiles = -(-int(c) // spec.tile_channels)
    if variant == "fused":
        if spec.mode == "infer" or tiles < NHWC_MIN_FUSED_TILES:
            return ()
        return (BatchNorm2DNhwcPlan(rows_per_cta=rows, splits=1, tiles=tiles),)
    if variant == "split" and spec.mode == "infer":
        return ()
    base = spec.ty * spec.unroll
    out = []
    rpc = base
    while True:
        splits = -(-rows // rpc)
        if variant == "split":
            if splits < 2:
                break
            for r in _nhwc_rpc_steps(rpc, base):
                s_r = -(-rows // r)
                if 2 <= s_r <= NHWC_MAX_SPLITS and s_r * tiles >= NHWC_MIN_SPLIT_CTAS:
                    out.append(BatchNorm2DNhwcPlan(r, s_r, tiles))
        else:
            if splits * tiles >= NHWC_MIN_SPLIT_CTAS // 2 or rpc == base:
                out.append(BatchNorm2DNhwcPlan(rpc, splits, tiles))
            if splits == 1:
                break
        rpc *= 2
    return tuple(out)


def batchnorm2d_nhwc_grid(c: int, plan: BatchNorm2DNhwcPlan) -> Tuple[int, int, int]:
    return (plan.splits, plan.tiles, 1)


def batchnorm2d_nhwc_workspace_floats(c: int, plan: BatchNorm2DNhwcPlan) -> int:
    return 2 * int(c) * plan.splits


class _NhwcCtx:
    """Per-thread tile coordinates shared by every NHWC kernel."""

    def __init__(self, b, spec, p_n, p_c, p_h, p_w, split):
        VEC, TX = spec.vec, spec.tx
        self.b, self.spec, self.p_c = b, spec, p_c
        self.tid = b.thread_id_x()
        self.tx = b.mod(self.tid, b.const_i32(TX))
        self.ty = b.div(self.tid, b.const_i32(TX))
        self.split = split
        tile = b.block_id_y()
        self.ch0 = b.add(
            b.mul(tile, b.const_i32(spec.tile_channels)),
            b.mul(self.tx, b.const_i32(VEC)),
        )
        self.ch_ok = b.cmp_lt(self.ch0, p_c)
        self.ch0s = b.select(self.ch_ok, self.ch0, b.const_i32(0))
        self.col = b.mul(self.tx, b.const_i32(VEC))
        self.rows = b.mul(b.mul(p_n, p_h), p_w)

    def rows_of_split(self, rows_per_cta):
        b = self.b
        r_lo = b.mul(self.split, rows_per_cta)
        r_hi = b.smin(self.rows, b.add(r_lo, rows_per_cta))
        return r_lo, r_hi

    def owner(self):
        b = self.b
        o = b.land(
            b.cmp_eq(self.split, b.const_i32(0)), b.cmp_eq(self.ty, b.const_i32(0))
        )
        return b.land(o, self.ch_ok)


def _nhwc_row_index(b, ctx, r, r_last):
    rs = b.smin(r, r_last)
    return b.add(b.mul(rs, ctx.p_c), ctx.ch0s)


def _nhwc_accumulate(b, spec, ctx, r_lo, r_hi, load_rows, accumulate, n_acc):
    """Per-thread accumulation over ``r_lo + ty, + TY, ...`` (below ``r_hi``).

    ``load_rows(idx)`` issues the loads of one row (``idx`` = flat element
    index, always in range); ``accumulate(acc, loaded, valid)`` folds it in."""
    TY, U = spec.ty, spec.unroll
    r_last = b.smax(b.sub(r_hi, b.const_i32(1)), b.const_i32(0))
    inits = [(f"nacc{i}", b.const_f32(0.0)) for i in range(n_acc)]
    for_op = b.scf_for_iter(
        b.add(r_lo, ctx.ty), r_hi, b.const_i32(TY * U), inits, iv_name="nr"
    )
    with for_op as (r_iv, acc):
        loaded, valids = [], []
        for u in range(U):
            ru = b.add(r_iv, b.const_i32(u * TY))
            valids.append(b.cmp_lt(ru, r_hi))
            loaded.append(load_rows(_nhwc_row_index(b, ctx, ru, r_last)))
        acc = list(acc)
        for u in range(U):
            acc = accumulate(acc, loaded[u], valids[u])
        b.scf_yield(*acc)
    return list(for_op.results)


def _nhwc_tile_totals(b, spec, ctx, acc, lds_a, lds_b):
    """Sum the two per-thread ``vec``-vectors over ``ty`` through LDS; every
    thread gets the totals of its own channels."""
    VEC, CT = spec.vec, spec.tile_channels
    mine = b.add(b.mul(ctx.ty, b.const_i32(CT)), ctx.col)
    for v in range(VEC):
        slot = b.add(mine, b.const_i32(v))
        b.smem_store_vN_f32(lds_a, [slot], acc[v], 1)
        b.smem_store_vN_f32(lds_b, [slot], acc[VEC + v], 1)
    b.sync()
    s = spec.ty // 2
    while s >= 1:
        with b.scf_if(b.cmp_lt(ctx.ty, b.const_i32(s))):
            for v in range(VEC):
                slot = b.add(mine, b.const_i32(v))
                other = b.add(slot, b.const_i32(s * CT))
                for lds in (lds_a, lds_b):
                    x0 = b.vec_extract(b.smem_load_vN_f32(lds, slot, n=1), 0)
                    x1 = b.vec_extract(b.smem_load_vN_f32(lds, other, n=1), 0)
                    b.smem_store_vN_f32(lds, [slot], b.fadd(x0, x1), 1)
        b.sync()
        s //= 2
    ta, tb = [], []
    for v in range(VEC):
        slot = b.add(ctx.col, b.const_i32(v))
        ta.append(b.vec_extract(b.smem_load_vN_f32(lds_a, slot, n=1), 0))
        tb.append(b.vec_extract(b.smem_load_vN_f32(lds_b, slot, n=1), 0))
    return ta, tb


def _nhwc_lds(b, spec):
    return tuple(
        make_lds_view(
            b, dtype=F32, shape=(spec.block_size * spec.vec,), name_hint=nm
        ).base
        for nm in ("lds_a", "lds_b")
    )


def _nhwc_store_partials(b, spec, ctx, Workspace, acc, p_c):
    """Thread row 0 of every channel group stores its two ``vec``-vectors."""
    VEC = spec.vec
    with b.scf_if(b.land(b.cmp_eq(ctx.ty, b.const_i32(0)), ctx.ch_ok)):
        for k in range(2):
            idx = b.add(
                b.mul(b.add(b.mul(ctx.split, b.const_i32(2)), b.const_i32(k)), p_c),
                ctx.ch0,
            )
            b.global_store_vN(
                Workspace, idx, b.vec_pack(acc[k * VEC : (k + 1) * VEC], F32), VEC
            )


def _nhwc_merge_partials(b, spec, ctx, Workspace, p_c, p_splits):
    """Per-thread sum of the planar partials ``s = ty, ty + TY, ...``."""
    VEC, TY = spec.vec, spec.ty
    inits = [(f"mrg{i}", b.const_f32(0.0)) for i in range(2 * VEC)]
    for_op = b.scf_for_iter(ctx.ty, p_splits, b.const_i32(TY), inits, iv_name="ms")
    with for_op as (s_iv, acc):
        out = []
        for k in range(2):
            idx = b.add(
                b.mul(b.add(b.mul(s_iv, b.const_i32(2)), b.const_i32(k)), p_c), ctx.ch0s
            )
            vals = load_vec(b, Workspace, idx, dtype="f32", n=VEC)
            for v in range(VEC):
                out.append(b.fadd(acc[k * VEC + v], b.vec_extract(vals, v)))
        b.scf_yield(*out)
    return list(for_op.results)


def _nhwc_apply_rows(b, spec, ctx, r_lo, r_hi, load_rows, compute, store):
    """Streaming pass over this thread's rows: ``compute(loaded)`` returns the
    ``vec`` f32 results, ``store(idx, ys)`` writes them."""
    TY, U = spec.ty, spec.unroll
    r_last = b.smax(b.sub(r_hi, b.const_i32(1)), b.const_i32(0))
    for_op = b.scf_for(b.add(r_lo, ctx.ty), r_hi, b.const_i32(TY * U), iv_name="ar")
    with for_op as r_iv:
        loaded, valids, idxs = [], [], []
        for u in range(U):
            ru = b.add(r_iv, b.const_i32(u * TY))
            valids.append(b.land(b.cmp_lt(ru, r_hi), ctx.ch_ok))
            idxs.append(_nhwc_row_index(b, ctx, ru, r_last))
            loaded.append(load_rows(idxs[-1]))
        for u in range(U):
            ys = compute(loaded[u])
            with b.scf_if(valids[u]):
                store(idxs[u], ys)


def _nhwc_f32_scalars(b, ptr, ctx, VEC, dtype):
    return [
        load_scalar_as_f32(b, ptr, b.add(ctx.ch0s, b.const_i32(v)), dtype=dtype)
        for v in range(VEC)
    ]


def _nhwc_x_loader(b, spec, X):
    return lambda idx: load_vec_as_f32(b, X, idx, dtype=spec.dtype, n=spec.vec)


def _nhwc_scale_shift(b, spec, ctx, Gamma, Beta, mean, var, eps):
    """Per-channel ``(scale, shift)`` with ``y = x * scale + shift``."""
    VEC = spec.vec
    g = _nhwc_f32_scalars(b, Gamma, ctx, VEC, spec.dtype)
    bt = _nhwc_f32_scalars(b, Beta, ctx, VEC, spec.dtype)
    sc = [b.fmul(b.rsqrt(b.fadd(var[v], eps)), g[v]) for v in range(VEC)]
    sh = [b.fsub(bt[v], b.fmul(mean[v], sc[v])) for v in range(VEC)]
    return sc, sh


def _nhwc_stream_y(b, spec, Y, sc, sh):
    VEC = spec.vec
    compute = lambda xs: [b.fma(xs[v], sc[v], sh[v]) for v in range(VEC)]  # noqa: E731
    store = lambda idx, ys: store_vec(
        b, Y, idx, pack_f32_to(b, ys, dtype=spec.dtype), n=VEC
    )  # noqa: E731
    return compute, store


def build_batchnorm2d_nhwc_infer(spec: BatchNorm2DNhwcSpec) -> KernelDef:
    ok, why = is_valid_nhwc_spec(spec)
    if not ok or spec.mode != "infer":
        raise ValueError(f"invalid NHWC infer spec: {why or spec.mode}")
    io_ty, VEC = io_ir_type(spec.dtype), spec.vec
    b = IRBuilder(spec.kernel_name("infer"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
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
    p_n, p_c = b.param("N", I32), b.param("C", I32)
    p_h, p_w = b.param("H", I32), b.param("W", I32)
    eps = b.param("eps", F32)
    b.param("momentum", F32)
    p_rpc = b.param("RowsPerCta", I32)
    ctx = _NhwcCtx(b, spec, p_n, p_c, p_h, p_w, b.block_id_x())
    mean = [
        b.global_load_f32(RunningMean, b.add(ctx.ch0s, b.const_i32(v)))
        for v in range(VEC)
    ]
    var = [
        b.global_load_f32(RunningVar, b.add(ctx.ch0s, b.const_i32(v)))
        for v in range(VEC)
    ]
    sc, sh = _nhwc_scale_shift(b, spec, ctx, Gamma, Beta, mean, var, eps)
    r_lo, r_hi = ctx.rows_of_split(p_rpc)
    compute, store = _nhwc_stream_y(b, spec, Y, sc, sh)
    _nhwc_apply_rows(
        b, spec, ctx, r_lo, r_hi, _nhwc_x_loader(b, spec, X), compute, store
    )
    return b.kernel


# ---- training ---------------------------------------------------------


def _nhwc_train_shift(b, spec, ctx, X):
    """Channel shift = first sample of the channel (row 0)."""
    return load_vec_as_f32(b, X, ctx.ch0s, dtype=spec.dtype, n=spec.vec)


def _nhwc_train_acc(b, spec, X, shift):
    VEC = spec.vec

    def accumulate(acc, xs, valid):
        out = list(acc)
        for v in range(VEC):
            d = b.select(valid, b.fsub(xs[v], shift[v]), b.const_f32(0.0))
            out[v] = b.fadd(out[v], d)
            out[VEC + v] = b.fma(d, d, out[VEC + v])
        return out

    return accumulate


def _nhwc_train_stats(b, spec, ctx, shift, t1, t2, p_n, p_h, p_w, eps):
    """``(mean, var, invstd)`` per channel from the shifted sums."""
    VEC = spec.vec
    m_f = b.sitofp_f32(ctx.rows)
    inv_m = b.fdiv(b.const_f32(1.0), m_f)
    mean, var, inv = [], [], []
    for v in range(VEC):
        ms = b.fmul(t1[v], inv_m)
        vr = b.fmax(b.fsub(b.fmul(t2[v], inv_m), b.fmul(ms, ms)), b.const_f32(0.0))
        mean.append(b.fadd(shift[v], ms))
        var.append(vr)
        inv.append(b.rsqrt(b.fadd(vr, eps)))
    return mean, var, inv, m_f


def _nhwc_train_epilogue(
    b, spec, ctx, mean, var, inv, m_f, RunningMean, RunningVar, Mean, InvStd, momentum
):
    VEC = spec.vec
    with b.scf_if(ctx.owner()):
        one_minus = b.fsub(b.const_f32(1.0), momentum)
        unbias = b.fdiv(m_f, b.fmax(b.fsub(m_f, b.const_f32(1.0)), b.const_f32(1.0)))
        for v in range(VEC):
            ch = b.add(ctx.ch0, b.const_i32(v))
            old_m = b.global_load_f32(RunningMean, ch)
            old_v = b.global_load_f32(RunningVar, ch)
            b.global_store(
                RunningMean,
                ch,
                b.fadd(b.fmul(one_minus, old_m), b.fmul(momentum, mean[v])),
            )
            b.global_store(
                RunningVar,
                ch,
                b.fadd(
                    b.fmul(one_minus, old_v), b.fmul(momentum, b.fmul(var[v], unbias))
                ),
            )
            store_scalar_from_f32(b, Mean, ch, mean[v], dtype=spec.dtype)
            store_scalar_from_f32(b, InvStd, ch, inv[v], dtype=spec.dtype)


def _nhwc_train_params(b, spec, kind):
    io_ty = io_ir_type(spec.dtype)
    P = {}
    P["X"] = b.param(
        "X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    if kind == "reduce":
        P["Workspace"] = b.param("Workspace", PtrType(F32, "global"), noalias=True)
    else:
        if kind == "apply":
            P["Workspace"] = b.param(
                "Workspace", PtrType(F32, "global"), noalias=True, readonly=True
            )
        P["Gamma"] = b.param(
            "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
        P["Beta"] = b.param(
            "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
        P["Y"] = b.param(
            "Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
        )
        P["RunningMean"] = b.param("RunningMean", PtrType(F32, "global"), noalias=True)
        P["RunningVar"] = b.param("RunningVar", PtrType(F32, "global"), noalias=True)
        P["Mean"] = b.param(
            "Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
        P["InvStd"] = b.param(
            "InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    for nm in ("N", "C", "H", "W"):
        P[nm] = b.param(nm, I32)
    if kind != "reduce":
        P["eps"] = b.param("eps", F32)
        P["momentum"] = b.param("momentum", F32)
    if kind != "fused":
        P["RowsPerCta"] = b.param("RowsPerCta", I32)
    if kind == "apply":
        P["Splits"] = b.param("Splits", I32)
    return P


def build_batchnorm2d_nhwc_train(spec: BatchNorm2DNhwcSpec, kind: str) -> KernelDef:
    ok, why = is_valid_nhwc_spec(spec)
    if not ok or spec.mode != "train" or kind not in ("fused", "reduce", "apply"):
        raise ValueError(f"invalid NHWC training spec: {why or (spec.mode, kind)}")
    b = IRBuilder(spec.kernel_name(kind))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    P = _nhwc_train_params(b, spec, kind)
    X = P["X"]
    ctx = _NhwcCtx(
        b,
        spec,
        P["N"],
        P["C"],
        P["H"],
        P["W"],
        b.block_id_x() if kind != "fused" else b.const_i32(0),
    )
    lds_a, lds_b = _nhwc_lds(b, spec)
    shift = _nhwc_train_shift(b, spec, ctx, X)
    loader = _nhwc_x_loader(b, spec, X)
    if kind == "apply":
        acc = _nhwc_merge_partials(b, spec, ctx, P["Workspace"], P["C"], P["Splits"])
    else:
        if kind == "fused":
            r_lo, r_hi = b.const_i32(0), ctx.rows
        else:
            r_lo, r_hi = ctx.rows_of_split(P["RowsPerCta"])
        acc = _nhwc_accumulate(
            b,
            spec,
            ctx,
            r_lo,
            r_hi,
            loader,
            _nhwc_train_acc(b, spec, X, shift),
            2 * spec.vec,
        )
    t1, t2 = _nhwc_tile_totals(b, spec, ctx, acc, lds_a, lds_b)
    if kind == "reduce":
        _nhwc_store_partials(b, spec, ctx, P["Workspace"], t1 + t2, P["C"])
        return b.kernel
    mean, var, inv, m_f = _nhwc_train_stats(
        b, spec, ctx, shift, t1, t2, P["N"], P["H"], P["W"], P["eps"]
    )
    _nhwc_train_epilogue(
        b,
        spec,
        ctx,
        mean,
        var,
        inv,
        m_f,
        P["RunningMean"],
        P["RunningVar"],
        P["Mean"],
        P["InvStd"],
        P["momentum"],
    )
    sc, sh = _nhwc_scale_shift(b, spec, ctx, P["Gamma"], P["Beta"], mean, var, P["eps"])
    compute, store = _nhwc_stream_y(b, spec, P["Y"], sc, sh)
    if kind == "fused":
        r_lo, r_hi = b.const_i32(0), ctx.rows
    else:
        r_lo, r_hi = ctx.rows_of_split(P["RowsPerCta"])
    _nhwc_apply_rows(b, spec, ctx, r_lo, r_hi, loader, compute, store)
    return b.kernel


# ---- backward ---------------------------------------------------------


def _nhwc_bwd_params(b, spec, kind):
    io_ty = io_ir_type(spec.dtype)
    P = {}
    P["X"] = b.param(
        "X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    P["dY"] = b.param(
        "dY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    if kind != "reduce":
        P["Scale"] = b.param(
            "Scale", PtrType(F32, "global"), noalias=True, readonly=True
        )
    P["Mean"] = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    P["InvStd"] = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    if kind != "fused":
        P["Workspace"] = b.param(
            "Workspace",
            PtrType(F32, "global"),
            noalias=True,
            readonly=(kind == "apply"),
        )
    if kind != "reduce":
        P["dX"] = b.param(
            "dX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
        )
        P["dScale"] = b.param(
            "dScale", PtrType(F32, "global"), noalias=True, writeonly=True
        )
        P["dBias"] = b.param(
            "dBias", PtrType(F32, "global"), noalias=True, writeonly=True
        )
    for nm in ("N", "C", "H", "W"):
        P[nm] = b.param(nm, I32)
    if kind != "fused":
        P["RowsPerCta"] = b.param("RowsPerCta", I32)
    if kind == "apply":
        P["Splits"] = b.param("Splits", I32)
    return P


def build_batchnorm2d_nhwc_bwd(spec: BatchNorm2DNhwcSpec, kind: str) -> KernelDef:
    ok, why = is_valid_nhwc_spec(spec)
    if not ok or spec.mode != "bwd" or kind not in ("fused", "reduce", "apply"):
        raise ValueError(f"invalid NHWC backward spec: {why or (spec.mode, kind)}")
    VEC = spec.vec
    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name(kind))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    P = _nhwc_bwd_params(b, spec, kind)
    X, dY = P["X"], P["dY"]
    ctx = _NhwcCtx(
        b,
        spec,
        P["N"],
        P["C"],
        P["H"],
        P["W"],
        b.block_id_x() if kind != "fused" else b.const_i32(0),
    )
    lds_a, lds_b = _nhwc_lds(b, spec)
    mean = [
        b.global_load_f32(P["Mean"], b.add(ctx.ch0s, b.const_i32(v)))
        for v in range(VEC)
    ]
    invstd = [
        b.global_load_f32(P["InvStd"], b.add(ctx.ch0s, b.const_i32(v)))
        for v in range(VEC)
    ]
    if kind == "fused":
        r_lo, r_hi = b.const_i32(0), ctx.rows
    else:
        r_lo, r_hi = ctx.rows_of_split(P["RowsPerCta"])

    if kind == "apply":
        acc = _nhwc_merge_partials(b, spec, ctx, P["Workspace"], P["C"], P["Splits"])
    else:

        def load_rows(idx):
            return (
                load_vec_as_f32(b, X, idx, dtype=spec.dtype, n=VEC),
                load_vec_as_f32(b, dY, idx, dtype=spec.dtype, n=VEC),
            )

        def accumulate(acc, loaded, valid):
            xs, dys = loaded
            out = list(acc)
            for v in range(VEC):
                dym = b.select(valid, dys[v], b.const_f32(0.0))
                xh = b.fmul(b.fsub(xs[v], mean[v]), invstd[v])
                out[v] = b.fadd(out[v], dym)
                out[VEC + v] = b.fma(dym, xh, out[VEC + v])
            return out

        acc = _nhwc_accumulate(b, spec, ctx, r_lo, r_hi, load_rows, accumulate, 2 * VEC)
    t1, t2 = _nhwc_tile_totals(
        b, spec, ctx, acc, lds_a, lds_b
    )  # sum(dy), sum(dy * xhat)
    if kind == "reduce":
        _nhwc_store_partials(b, spec, ctx, P["Workspace"], t1 + t2, P["C"])
        return b.kernel

    with b.scf_if(ctx.owner()):
        for v in range(VEC):
            ch = b.add(ctx.ch0, b.const_i32(v))
            b.global_store(P["dScale"], ch, t2[v])
            b.global_store(P["dBias"], ch, t1[v])
    scale = [
        b.global_load_f32(P["Scale"], b.add(ctx.ch0s, b.const_i32(v)))
        for v in range(VEC)
    ]
    m_f = b.sitofp_f32(ctx.rows)
    coefs = [
        _bwd_coefs(b, t1[v], t2[v], scale[v], mean[v], invstd[v], m_f)
        for v in range(VEC)
    ]

    def load_rows2(idx):
        return (
            load_vec_as_f32(b, X, idx, dtype=spec.dtype, n=VEC),
            load_vec_as_f32(b, dY, idx, dtype=spec.dtype, n=VEC),
        )

    def compute(loaded):
        xs, dys = loaded
        out = []
        for v in range(VEC):
            a, k2, k3 = coefs[v]
            out.append(
                b.fsub(
                    b.fsub(b.fmul(a, dys[v]), b.fmul(k2, b.fsub(xs[v], mean[v]))), k3
                )
            )
        return out

    store = lambda idx, ys: store_vec(
        b, P["dX"], idx, pack_f32_to(b, ys, dtype=spec.dtype), n=VEC
    )  # noqa: E731
    _nhwc_apply_rows(b, spec, ctx, r_lo, r_hi, load_rows2, compute, store)
    return b.kernel


def build_batchnorm2d_nhwc(spec: BatchNorm2DNhwcSpec, kind: str) -> KernelDef:
    """Dispatch on ``spec.mode`` to the NHWC builder of ``kind``."""
    if spec.mode == "infer":
        return build_batchnorm2d_nhwc_infer(spec)
    if spec.mode == "train":
        return build_batchnorm2d_nhwc_train(spec, kind)
    return build_batchnorm2d_nhwc_bwd(spec, kind)


# =====================================================================
# Multi-channel-per-wave small-tensor kernels (NCHW spatial BatchNorm)
# =====================================================================
#
# One CTA owns ``cpw`` channels (a group of ``64 // cpw`` lanes per channel)
# and every sample of them.  The ``waves`` waves of the CTA split the samples
# (sample ``k * waves + wave``), each lane keeps its ``npw * slots`` elements
# in registers, so the tensor is read once: the sums come from a butterfly
# inside the lane group plus one LDS exchange across waves, and the output is
# written from the registers.  Training sums are taken around a per-channel
# shift (the first element of the channel).

_MCW_MODES = ("train", "bwd")
MCW_MAX_REGS = 192


@dataclass(frozen=True)
class BatchNorm2DMcwSpec:
    """One dynamic multi-channel-per-wave instance (n / c / h / w runtime)."""

    mode: str = "train"  # "train" | "bwd"
    waves: int = 4
    cpw: int = 1
    npw: int = 32
    slots: int = 1
    vec: int = 1
    dtype: DType = "f16"
    wave_size: int = 64
    dynamic: bool = True
    name: str = "rocke_batchnorm2d_mcw"

    @property
    def block_size(self) -> int:
        return self.waves * self.wave_size

    @property
    def lpc(self) -> int:
        return self.wave_size // self.cpw

    @property
    def regs(self) -> int:
        return self.npw * self.slots * self.vec * (1 if self.mode == "train" else 2)

    def kernel_name(self, kind: str = "fused") -> str:
        return kernel_name_join(
            self.name,
            self.mode,
            self.dtype,
            f"w{self.waves}",
            f"cp{self.cpw}",
            f"n{self.npw}",
            f"s{self.slots}",
            f"v{self.vec}" if self.vec != 1 else "",
        )


def is_valid_mcw_spec(
    spec: BatchNorm2DMcwSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if spec.mode not in _MCW_MODES:
        return False, f"unknown mode {spec.mode!r}"
    if spec.dtype not in ("f16", "bf16", "f32"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.cpw not in (1, 2, 4, 8, 16, 32, 64):
        return False, "cpw must be a power of two dividing the wave"
    if spec.waves <= 0 or spec.waves & (spec.waves - 1):
        return False, "waves must be a power of two"
    if spec.vec not in ((1, 2, 4) if spec.dtype == "f32" else (1, 2, 4, 8)):
        return False, "vec out of range"
    if spec.npw <= 0 or spec.slots <= 0:
        return False, "npw / slots must be positive"
    if spec.regs > MCW_MAX_REGS:
        return False, "register-cached elements per lane out of budget"
    if spec.block_size > target.max_threads_per_block:
        return False, "block_size exceeds max_threads_per_block"
    if not target.fits_lds(2 * spec.waves * spec.cpw * 4):
        return False, "LDS budget exceeded"
    return True, ""


def _ceil_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def mcw_admits(
    spec: BatchNorm2DMcwSpec, n: int, c: int, h: int, w: int
) -> Tuple[bool, str]:
    """Runtime-shape admission of one multi-channel-per-wave instance.

    The sample count per wave and the slot count are the smallest power of
    two covering the shape, so no instance is more than twice too large."""
    n, c, hw = int(n), int(c), int(h) * int(w)
    if n <= 0 or c <= 0 or hw <= 0:
        return False, "empty tensor"
    if c % spec.cpw != 0:
        return False, "C not a multiple of cpw"
    if spec.waves > n:
        return False, "more waves than samples"
    if spec.npw != _ceil_pow2(-(-n // spec.waves)):
        return False, "npw does not match the sample count"
    if hw % spec.vec != 0:
        return False, "spatial size not a multiple of vec"
    if spec.slots != _ceil_pow2(-(-(hw // spec.vec) // spec.lpc)):
        return False, "slots do not match the spatial size"
    if n * c * hw + spec.block_size >= 2**31:
        return False, "element count out of i32 range"
    return True, ""


def batchnorm2d_mcw_grid(c: int, spec: BatchNorm2DMcwSpec) -> Tuple[int, int, int]:
    return (int(c) // spec.cpw, 1, 1)


class _McwCtx:
    """Lane coordinates and cached-element addressing shared by both kernels."""

    def __init__(self, b, spec, p_n, p_c, p_h, p_w):
        W, CPW, LPC = spec.waves, spec.cpw, spec.lpc
        self.b, self.spec, self.p_n, self.p_c = b, spec, p_n, p_c
        self.hw = b.mul(p_h, p_w)
        tid = b.thread_id_x()
        self.tid = tid
        wsz = b.const_i32(spec.wave_size)
        self.wave = b.div(tid, wsz)
        lane = b.mod(tid, wsz)
        self.lane = lane
        self.cj = b.div(lane, b.const_i32(LPC))
        self.p = b.mod(lane, b.const_i32(LPC))
        self.channel = b.add(b.mul(b.block_id_x(), b.const_i32(CPW)), self.cj)
        hw_last = b.sub(b.div(self.hw, b.const_i32(spec.vec)), b.const_i32(1))
        self.hw_idx, self.hw_ok = [], []
        for s in range(spec.slots):
            hs = b.add(self.p, b.const_i32(LPC * s))
            self.hw_ok.append(b.cmp_le(hs, hw_last))
            self.hw_idx.append(b.smin(hs, hw_last))
        n_last = b.sub(p_n, b.const_i32(1))
        self.n_idx, self.n_ok = [], []
        for k in range(spec.npw):
            nk = b.add(b.const_i32(k * W), self.wave)
            self.n_ok.append(b.cmp_lt(nk, p_n))
            self.n_idx.append(b.smin(nk, n_last))
        self.owner = b.land(
            b.cmp_eq(self.wave, b.const_i32(0)), b.cmp_eq(self.p, b.const_i32(0))
        )

    def elems(self):
        """``(flat index, valid)`` of every cached element, sample-major."""
        b, spec = self.b, self.spec
        out = []
        for k in range(spec.npw):
            base = b.mul(b.add(b.mul(self.n_idx[k], self.p_c), self.channel), self.hw)
            for s in range(spec.slots):
                col = (
                    self.hw_idx[s]
                    if spec.vec == 1
                    else b.mul(self.hw_idx[s], b.const_i32(spec.vec))
                )
                out.append((b.add(base, col), b.land(self.n_ok[k], self.hw_ok[s])))
        return out


def _mcw_load(b, spec, ptr, idx):
    """The ``vec`` consecutive elements at ``idx`` as f32 scalars."""
    if spec.vec == 1:
        return [load_scalar_as_f32(b, ptr, idx, dtype=spec.dtype)]
    return list(load_vec_as_f32(b, ptr, idx, dtype=spec.dtype, n=spec.vec))


def _mcw_store(b, spec, ptr, idx, vals):
    if spec.vec == 1:
        store_scalar_from_f32(b, ptr, idx, vals[0], dtype=spec.dtype)
    else:
        store_vec(b, ptr, idx, pack_f32_to(b, vals, dtype=spec.dtype), n=spec.vec)


def _mcw_group_sums(b, spec, ctx, vals, lds):
    """Sum each of ``vals`` (per-lane f32 partials) over its channel's lanes
    and waves; every lane gets the totals of its own channel."""
    W, CPW, LPC = spec.waves, spec.cpw, spec.lpc
    tot = [_warp_xor_reduce(b, v, combine="sum", wave_size=LPC) for v in vals]
    if W == 1:
        return tot
    nv = len(vals)
    with b.scf_if(b.cmp_eq(ctx.p, b.const_i32(0))):
        for q in range(nv):
            slot = b.add(b.mul(ctx.wave, b.const_i32(CPW)), ctx.cj)
            b.smem_store_vN_f32(lds, [b.add(slot, b.const_i32(q * W * CPW))], tot[q], 1)
    b.sync()
    out = []
    for q in range(nv):
        parts = []
        for w in range(W):
            slot = b.add(ctx.cj, b.const_i32(q * W * CPW + w * CPW))
            parts.append(b.vec_extract(b.smem_load_vN_f32(lds, slot, n=1), 0))
        out.append(tree_reduce(b, b.fadd, parts))
    return out


def _mcw_lds(b, spec, n_vals):
    return make_lds_view(
        b, dtype=F32, shape=(n_vals * spec.waves * spec.cpw,), name_hint="lds_mcw"
    ).base


def build_batchnorm2d_mcw_train(spec: BatchNorm2DMcwSpec) -> KernelDef:
    ok, why = is_valid_mcw_spec(spec)
    if not ok or spec.mode != "train":
        raise ValueError(f"invalid mcw training spec: {why or spec.mode}")
    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
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
    Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True)
    InvStd = b.param("InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True)
    p_n, p_c = b.param("N", I32), b.param("C", I32)
    p_h, p_w = b.param("H", I32), b.param("W", I32)
    eps = b.param("eps", F32)
    momentum = b.param("momentum", F32)

    ctx = _McwCtx(b, spec, p_n, p_c, p_h, p_w)
    lds = _mcw_lds(b, spec, 2)
    shift = load_scalar_as_f32(b, X, b.mul(ctx.channel, ctx.hw), dtype=spec.dtype)
    gamma = load_scalar_as_f32(b, Gamma, ctx.channel, dtype=spec.dtype)
    beta = load_scalar_as_f32(b, Beta, ctx.channel, dtype=spec.dtype)
    elems = ctx.elems()
    xs = [_mcw_load(b, spec, X, idx) for idx, _ in elems]
    zero = b.const_f32(0.0)
    ds = [
        b.select(ok_, b.fsub(x, shift), zero)
        for xv, (_, ok_) in zip(xs, elems)
        for x in xv
    ]
    s1 = tree_reduce(b, b.fadd, list(ds))
    s2 = tree_reduce(b, b.fadd, [b.fmul(d, d) for d in ds])
    t1, t2 = _mcw_group_sums(b, spec, ctx, [s1, s2], lds)

    m_f = b.sitofp_f32(b.mul(p_n, ctx.hw))
    inv_m = b.fdiv(b.const_f32(1.0), m_f)
    ms = b.fmul(t1, inv_m)
    var = b.fmax(b.fsub(b.fmul(t2, inv_m), b.fmul(ms, ms)), zero)
    mean = b.fadd(shift, ms)
    inv = b.rsqrt(b.fadd(var, eps))
    with b.scf_if(ctx.owner):
        one_minus = b.fsub(b.const_f32(1.0), momentum)
        unbias = b.fdiv(m_f, b.fmax(b.fsub(m_f, b.const_f32(1.0)), b.const_f32(1.0)))
        old_m = b.global_load_f32(RunningMean, ctx.channel)
        old_v = b.global_load_f32(RunningVar, ctx.channel)
        b.global_store(
            RunningMean,
            ctx.channel,
            b.fadd(b.fmul(one_minus, old_m), b.fmul(momentum, mean)),
        )
        b.global_store(
            RunningVar,
            ctx.channel,
            b.fadd(b.fmul(one_minus, old_v), b.fmul(momentum, b.fmul(var, unbias))),
        )
        store_scalar_from_f32(b, Mean, ctx.channel, mean, dtype=spec.dtype)
        store_scalar_from_f32(b, InvStd, ctx.channel, inv, dtype=spec.dtype)
    sc = b.fmul(inv, gamma)
    sh = b.fsub(beta, b.fmul(mean, sc))
    for xv, (idx, ok_) in zip(xs, elems):
        with b.scf_if(ok_):
            _mcw_store(b, spec, Y, idx, [b.fma(x, sc, sh) for x in xv])
    return b.kernel


def build_batchnorm2d_mcw_bwd(spec: BatchNorm2DMcwSpec) -> KernelDef:
    ok, why = is_valid_mcw_spec(spec)
    if not ok or spec.mode != "bwd":
        raise ValueError(f"invalid mcw backward spec: {why or spec.mode}")
    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    dY = b.param("dY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Scale = b.param("Scale", PtrType(F32, "global"), noalias=True, readonly=True)
    Mean = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    InvStd = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    dX = b.param("dX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    dScale = b.param("dScale", PtrType(F32, "global"), noalias=True, writeonly=True)
    dBias = b.param("dBias", PtrType(F32, "global"), noalias=True, writeonly=True)
    p_n, p_c = b.param("N", I32), b.param("C", I32)
    p_h, p_w = b.param("H", I32), b.param("W", I32)

    ctx = _McwCtx(b, spec, p_n, p_c, p_h, p_w)
    lds = _mcw_lds(b, spec, 2)
    mean = b.global_load_f32(Mean, ctx.channel)
    invstd = b.global_load_f32(InvStd, ctx.channel)
    scale = b.global_load_f32(Scale, ctx.channel)
    elems = ctx.elems()
    xs = [_mcw_load(b, spec, X, idx) for idx, _ in elems]
    dys = [_mcw_load(b, spec, dY, idx) for idx, _ in elems]
    zero = b.const_f32(0.0)
    dym = [b.select(ok_, dy, zero) for dyv, (_, ok_) in zip(dys, elems) for dy in dyv]
    prods = [
        b.fmul(d, b.fmul(b.fsub(x, mean), invstd))
        for d, x in zip(dym, [x for xv in xs for x in xv])
    ]
    s1 = tree_reduce(b, b.fadd, list(dym))
    s2 = tree_reduce(b, b.fadd, prods)
    t1, t2 = _mcw_group_sums(b, spec, ctx, [s1, s2], lds)
    with b.scf_if(ctx.owner):
        b.global_store(dScale, ctx.channel, t2)
        b.global_store(dBias, ctx.channel, t1)
    m_f = b.sitofp_f32(b.mul(p_n, ctx.hw))
    a, k2, k3 = _bwd_coefs(b, t1, t2, scale, mean, invstd, m_f)
    for xv, dyv, (idx, ok_) in zip(xs, dys, elems):
        with b.scf_if(ok_):
            _mcw_store(
                b,
                spec,
                dX,
                idx,
                [
                    b.fsub(b.fsub(b.fmul(a, dy), b.fmul(k2, b.fsub(x, mean))), k3)
                    for x, dy in zip(xv, dyv)
                ],
            )
    return b.kernel


def build_batchnorm2d_mcw(spec: BatchNorm2DMcwSpec) -> KernelDef:
    return (
        build_batchnorm2d_mcw_train(spec)
        if spec.mode == "train"
        else build_batchnorm2d_mcw_bwd(spec)
    )
