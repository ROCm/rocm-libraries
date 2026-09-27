# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end fused-MoE forward orchestrator.

Production runtime that drives the full fused-MoE forward pipeline
(routing logits in -> output activations out) as a sequence of
:func:`~rocke.runtime.launcher.launch_kernel` chained launches plus
a per-expert grouped-GEMM dispatch loop. Composes the existing single-
purpose instance launchers:

* :class:`~rocke.instances.common.topk_softmax.TopkSoftmaxSpec` (single
  launch) for the router.
* :class:`~rocke.instances.common.moe_sorting.MoeSortingLauncher` for the
  3-phase sort (histogram -> scan -> scatter).
* :class:`~rocke.instances.common.fused_moe.FusedMoeLauncher` for the
  3-phase MoE-specific chain (gather -> silu_mul -> topk_reduce).
* :class:`~rocke.instances.common.grouped_gemm.GroupedGemmLauncher` for
  per-expert gate / up / down GEMMs.

Pipeline (call graph executed in declaration order on one HIP stream)::

    +-----------+    +---------------+    +-----------+    +-----------+
    |  router   | -> | sort (3 ker.) | -> |  gather   | -> |  per-exp  |
    | topk-soft |    | hist+scan+    |    | (1 ker.)  |    |  gate GEMM|
    | (1 ker.)  |    | scatter       |    +-----------+    | (E launch)|
    +-----------+    +---------------+                     +-----------+
                                                                 |
                                                                 v
    +-----------+    +-----------+    +-----------+    +-----------+
    | topk_red  | <- |  per-exp  | <- | silu_mul  | <- |  per-exp  |
    | (1 ker.)  |    |  down GEMM|    | (1 ker.)  |    |  up   GEMM|
    +-----------+    | (E launch)|    +-----------+    | (E launch)|
                     +-----------+                     +-----------+

For ``E`` experts the total launch count is ``5 + 3*E`` (5 fixed
streaming kernels + per-expert gate/up/down GEMMs). The 5 fixed
kernels are submitted via a single
:func:`~rocke.runtime.launcher.launch_kernel` call -- one
:func:`~rocke.runtime.launcher.make_kernel` closure per kernel,
plus two :class:`~rocke.runtime.launcher.make_kernel`-shape lambdas
that ``hipMemset`` the histogram + counter buffers between sort
phases. The per-expert GEMMs go through
:class:`~rocke.instances.GroupedGemmLauncher`, which loops in Python
and issues one HIP launch per expert; the launcher caches its
HSACO + module + function so the per-call cost is just args packing
+ ``hipModuleLaunchKernel``.

Per-expert dispatch detail
--------------------------
Per-expert GEMMs need ``(M=count[e], N=I_or_H, K=H_or_I, A_off,
B_off, C_off)`` per expert. ``count[e]`` and ``offsets[e]`` are
device-side i32 buffers populated by the sort scan. To dispatch
GEMMs from the host, we copy these two ``(experts,)`` i32 arrays
back to CPU once (via ``.cpu()``) immediately after the sort.
This is a small D->H copy (typically <= 128 i32s) but it does add a
host-side stall to the timing path. The single-launch persistent
grouped-GEMM (CK's ``persistent=True`` design point) lifts the
host roundtrip; it is listed as a v2 follow-on in the grouped-GEMM
docstring.

Numerical contract
------------------
Final accumulation happens in f32 (the topk-weighted reduce kernel
atomic-adds into an f32 ``Y`` accumulator). The orchestrator owns a
``Y_f32`` workspace tensor; the user-supplied ``Y`` output buffer is
populated via a torch dtype-aware ``copy_`` after the chain
completes (so the user sees ``f16`` / ``bf16`` output even though the
kernel-level reduce is f32). The cast is included inside the timed
region of :meth:`forward` (when ``time_kernel=True``) so the
benchmark number matches what an end-to-end caller would observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from ...helpers.compile import compile_kernel
from ...runtime.launcher import (
    KernelLauncher,
    StreamConfig,
    WorkspacePool,
    WorkspaceSpec,
    launch_kernel,
    make_kernel,
)
from .fused_moe import (
    FusedMoeLauncher,
    FusedMoeSpec,
    build_moe_silu_mul_packed,
    build_moe_static_scatter_gather,
    moe_gather_grid,
    moe_silu_mul_packed_signature,
    moe_static_scatter_gather_grid,
    moe_static_scatter_gather_signature,
)
from .batched_gemm import (
    BatchedGemmSpec,
    batched_gemm_signature,
    build_batched_gemm,
)
from .gemm_universal import (
    TileSpec,
    TraitSpec,
)
from .grouped_gemm import (
    GroupedGemmLauncher,
    GroupedGemmProblem,
    GroupedGemmSpec,
    build_grouped_gemm,
)
from .moe_sorting import MoeSortingLauncher, MoeSortingSpec
from .moe_gemm_fused import (
    FusedDownReduceGemmSpec,
    FusedGateUpSiluGemmSpec,
    FusedInterleavedGateUpSiluGemmSpec,
    build_moe_down_reduce_gemm,
    build_moe_gate_up_silu_gemm,
    build_moe_interleaved_gate_up_silu_gemm,
    moe_down_reduce_gemm_grid,
    moe_down_reduce_gemm_grouped_grid,
    moe_down_reduce_gemm_signature,
    moe_gate_up_silu_gemm_grid,
    moe_gate_up_silu_gemm_grouped_grid,
    moe_gate_up_silu_gemm_signature,
    moe_interleaved_gate_up_silu_gemm_grid,
    moe_interleaved_gate_up_silu_gemm_signature,
)
from .topk_softmax import (
    TopkSoftmaxSpec,
    build_topk_softmax,
    topk_softmax_grid,
    topk_softmax_signature,
)


__all__ = [
    "FusedMoeForward",
    "FusedMoeForwardSpec",
]


def _default_gemm_tile() -> TileSpec:
    """Tuned MFMA tile for the fused-MoE batched GEMMs.

    Sweep result on MI355X (``examples/gfx950/moe/tune_gate_up_silu.py``):
    ``tile_m=32, tile_n=128, tile_k=64, warp_m=1, warp_n=2,
    atom=32x32x16`` is the fastest correctness-clean tile for the
    graph-captured packed path on both small and decode shapes:

    * small_T32_E4_K2_H128_I256: 0.039 ms (packed), 0.0394 ms
      (experimental fused)
    * decode_T1_E8_K2_H4096_I7168: 0.421 ms (packed), 0.442 ms
      (experimental fused)

    This keeps the production default on the faster packed path while
    leaving the fused gate+up+silu kernel available via
    ``use_experimental_fused_gate_up_silu=True`` for future tuning.
    """
    return TileSpec(
        tile_m=32,
        tile_n=128,
        tile_k=64,
        warp_m=1,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def _default_bf16_gemm_tile() -> TileSpec:
    """BF16-compatible default tile for fused-MoE batched GEMMs.

    BF16 on gfx950 only supports 16x16 MFMA atoms (no 32x32 atom).
    Using warp_tile=(16,16,32) with warp_m=2, warp_n=2 gives:
    tile_m=32, tile_n=32, tile_k=32, block_size=256.
    This satisfies load_vec >= 2: a_vecs=1024/2=512>=256 and b_vecs=2.

    The ``16x16x32`` bf16 atom is a gfx950-only WIDE atom; for gfx942
    use :func:`_default_f16_gemm_tile_gfx942` / the bf16 narrow tile
    below.
    """
    return TileSpec(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )


def _default_gemm_tile_gfx942() -> TileSpec:
    """gfx942 (CDNA3) f16 default tile.

    gfx942 lacks the wide f16 ``32x32x16`` atom that
    :func:`_default_gemm_tile` selects; the narrowest-K 32x32 f16 atom
    on gfx942 is ``32x32x8``. Same outer tile geometry
    (tile_m=32, tile_n=128, tile_k=64, warp_m=1, warp_n=2,
    block_size=128) so the orchestrator's grid / stride arithmetic is
    unchanged; only ``warp_tile_k`` drops 16 -> 8.
    """
    return TileSpec(
        tile_m=32,
        tile_n=128,
        tile_k=64,
        warp_m=1,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=8,
    )


def _default_bf16_gemm_tile_gfx942() -> TileSpec:
    """gfx942 (CDNA3) bf16 default tile.

    gfx942 lacks the wide bf16 ``16x16x32`` atom; the only bf16 atom is
    ``16x16x16``. Same outer geometry as :func:`_default_bf16_gemm_tile`
    (tile_m=32, tile_n=32, tile_k=32, warp_m=2, warp_n=2,
    block_size=256); only ``warp_tile_k`` drops 32 -> 16.
    """
    return TileSpec(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    )


def _default_gfx1250_wmma_gemm_tile() -> TileSpec:
    """Minimal gfx1250 WMMA tile for day-0 BF16/FP16 MoE expert GEMMs."""
    return TileSpec(
        tile_m=16,
        tile_n=16,
        tile_k=32,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )


def _large_batch_gemm_tile() -> TileSpec:
    """Measured winner for large hidden + T>=32 shapes.

    Sweep result (``/tmp/sweep_moe_tiles.py``):
    * batch32_E8_K2_H4096_I7168: t64n128k64_w2x2_atom32 + static = 0.5356 ms
      vs t32n128 static = 0.5360 ms (tie, slightly better).
    * prefill_T128_E8_K2_H4096_I7168: t64n128 dynamic = 0.9074 ms
      vs t32n128 dynamic = 1.0587 ms.

    Keep the default t32 tile for decode/small, but use this tile for
    larger T on production hidden sizes.
    """
    return TileSpec(
        tile_m=64,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def _sparse_batch_gemm_tile() -> TileSpec:
    """Measured winner for sparse-routing large-hidden shapes (E >> T*K/E).

    For sparse MoE (datacenter T128/E32/K5/H8192/I8192: ~20 routed tokens
    per expert) the per-expert padded GEMM rounds each expert's token
    count up to ``tile_m``. With the dense ``_large_batch_gemm_tile``
    ``tile_m=64`` that means ~20 real rows in a 64-row tile -> 3.2x of the
    MFMA work is on padded zero rows. Halving ``tile_m`` to 32 (and
    widening ``tile_n`` to 256 / ``warp_n`` to 4 to keep the warp tiling
    full) cuts the per-expert row rounding 64 -> 32 and roughly halves the
    padded GEMM work.

    Sweep result (perf_moe, best-of-5, datacenter T128/E32/K5/H8192/I8192,
    dynamic path on gfx950 / MI355X):

    * ``_large_batch_gemm_tile`` (t64n128k64 w2x2): 3.91 ms
    * t32n256k64 w1x4:                              3.48 ms
    * t32n128k128 w1x2 (this tile):                ~3.24 ms (deeper
      ``tile_k=128`` lifts the dominant gate+up GEMM's arithmetic
      intensity over the t32n256k64 variant)

    The wide ``32x32x16`` f16 atom is gfx950-only; gfx942 callers fall
    back to the narrow-K atom in :class:`FusedMoeForward.__init__`.
    """
    return TileSpec(
        tile_m=32,
        tile_n=128,
        tile_k=128,
        warp_m=1,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def _sparse_batch_gemm_tile_gfx942() -> TileSpec:
    """gfx942 (CDNA3) variant of :func:`_sparse_batch_gemm_tile`.

    gfx942 lacks the wide f16 ``32x32x16`` atom; use the narrow
    ``32x32x8`` atom (same outer geometry).
    """
    return TileSpec(
        tile_m=32,
        tile_n=128,
        tile_k=128,
        warp_m=1,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=8,
    )


def _resolve_launch_arch(arch: "str | None") -> str:
    """Resolve a launch-path target arch.

    Explicit ``arch`` wins; otherwise probe the running device via
    :func:`rocke.runtime.hip_module.get_device_arch` and fall back to
    ``"gfx950"`` when no HIP device is visible (cross-compile / static
    test environments). Tolerant of probe failure so the orchestrator
    stays import-time-safe without a HIP runtime.
    """
    if arch is not None:
        return arch
    try:
        from ...runtime.hip_module import get_device_arch

        dev = get_device_arch()
        if dev:
            return dev
    except Exception:
        pass
    return "gfx950"


def _wave_size_for_arch(arch: "str | None") -> int:
    from ...core.arch import ArchTarget

    return ArchTarget.from_gfx(_resolve_launch_arch(arch)).wave_size


def _gemm_dtype_to_universal(dtype: str) -> str:
    if dtype in ("f16", "fp16"):
        return "fp16"
    if dtype == "bf16":
        return "bf16"
    raise ValueError(f"unsupported gemm dtype {dtype!r}; expected f16 / bf16")


def _torch_dtype_for(dtype: str) -> Any:
    """Return the torch dtype for our internal dtype label.

    Imported lazily so the module does not pull torch at parse time.
    """
    import torch

    if dtype in ("f16", "fp16"):
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported activation dtype {dtype!r}")


def _ensure_2byte_dtype(dtype: str) -> int:
    """Element size in bytes for the activation dtype (fp16 / bf16)."""
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"only 2-byte activation dtypes supported (got {dtype!r})")
    return 2


@dataclass
class FusedMoeForwardSpec:
    """One concrete fused-MoE forward configuration.

    Captures the shapes + dtypes shared across every kernel in the
    pipeline, plus tunable per-component knobs (block sizes, GEMM
    tile, etc.). The component specs (router / sort / streaming /
    gemm) are derived from these top-level fields by helper methods
    so callers configure one spec per problem shape.
    """

    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    dtype: str = "f16"

    streaming_block_size: int = 256
    streaming_vec: int = 8  # widest fp16 vector (16 bytes); halves the
    # global-load instruction count vs the v1 vec=4 default. Bandwidth-
    # bound streaming kernels (gather / silu_mul / topk_reduce) benefit
    # measurably; the constraint is hidden % vec == 0 and intermediate %
    # vec == 0, which holds for every typical MoE shape (multiples of
    # 64 or 128).
    sort_block_size: int = 64
    router_block_size: int = 64
    use_static_offsets: Optional[bool] = None
    """Override static-offset routing selection.

    ``None`` keeps the measured auto heuristic. ``True`` forces the
    decode-oriented static route that skips histogram / scan / dynamic
    scatter, while ``False`` forces the dynamic sorting path.
    """
    static_slot_size: Optional[int] = None
    """Override rows reserved per expert in static-offset mode.

    ``None`` uses the conservative tile-aligned worst-case slot. Callers
    may set this to a smaller value only when that value still bounds the
    maximum routed pairs any one expert can receive.
    """

    gemm_tile: TileSpec = field(default_factory=_default_gemm_tile)
    arch: "str | None" = None
    """Target GPU architecture for kernel compilation.

    ``None`` (default) resolves to the running device via
    :func:`rocke.runtime.hip_module.get_device_arch`, falling back to
    ``"gfx950"`` when no HIP device is visible. Set explicitly
    (``"gfx942"`` / ``"gfx950"``) to cross-compile the whole forward for
    a specific target. When the caller leaves ``gemm_tile`` at its
    default, :class:`FusedMoeForward` swaps in an arch-legal MFMA tile
    (the default f16/bf16 tiles use gfx950-only WIDE atoms, which would
    crash comgr on gfx942).
    """
    name: str = "rocke_fused_moe_forward"
    use_experimental_fused_gate_up_silu: bool = False
    """Use the experimental dual-B MFMA gate+up+silu kernel.

    Default False because measurements on MI355X show the graph-
    captured packed path (one N=2I batched GEMM + packed silu_mul
    streaming kernel) is faster today. The experimental kernel is
    correctness-clean and kept as a research/tuning target.
    """
    use_experimental_interleaved_gate_up_silu: bool = True
    """Use interleaved gate/up GEMM + SiLU epilogue.

    Packs gate/up weights as adjacent N columns (gate_i, up_i), runs a
    single-B GEMM, and performs SiLU in the cshuffle epilogue from LDS
    pairs. This crosses the activation barrier inside the GEMM kernel
    without the GateUpPacked HBM round-trip. Promoted as the default
    after tuning on MI355X:

    * small: 0.0388 ms vs packed 0.0391 ms
    * decode1: 0.4401 ms vs packed 0.4495 ms
    * decode8: 0.4345 ms vs packed 0.4781 ms

    The older packed path remains available by setting this flag to
    False. The dual-B experimental path (``use_experimental_fused_*``)
    takes precedence if both flags are set.
    """
    use_experimental_fused_down_reduce: bool = True
    """Use the down-GEMM + topk-reduce fused kernel (default on).

    Computes down GEMM and performs the weighted f32 atomic-add into
    ``Y`` directly from the MFMA accumulator, removing ``DownOut`` and
    the separate ``topk_reduce`` launch.

    Now default-on after tuning: the atomic epilogue's f32 ``atomicrmw
    fadd`` is lowered with the AMDGPU native-FP-atomic memory-model
    metadata (``core/lower_llvm.py``), so each weighted reduce is a
    single ``global_atomic_add_f32`` instruction rather than a
    ``global_atomic_cmpswap`` retry loop. Measured on gfx950 (perf_moe,
    best-of-N, graph-replay):

    * decode  T8 E8 K2 H4096 I7168 : 0.369 ms -> 0.329 ms  (~10.5% faster)
    * decode  T1/T16 (same H/I)    : same ~10-12% win
    * datacenter T128 E32 K5 H8192 I8192 : neutral (3.93 -> 3.92 ms,
      compute-bound down GEMM; fusion savings amortized)

    Parity is byte-identical to the two-kernel path (max_abs unchanged).
    Set False to restore the legacy DownOut + separate topk_reduce path.
    """
    use_experimental_static_scatter_gather: bool = False
    """Fuse static scatter and gather into one streaming kernel.

    Correctness-clean but slower on MI355X in measurements:
    the one-CTA-per-routed-pair kernel copies the full hidden row and
    loses to the older two-kernel ``scatter -> gather`` sequence's
    memory pattern. Kept opt-in for further tuning.
    """
    preshuffle_w_down: bool = False
    """Pre-shuffle ``W_down`` and use a ``preshuffle_b=True`` BatchedGemm
    for the down stage.

    The host re-arranges ``W_down`` from ``(E, hidden, intermediate)``
    row-major to ``(E, k_tiles, n_tiles, block_n, block_k)`` contiguous
    so each per-K-tile B-load in the down GEMM collapses to one wide
    ``buffer_load_dwordx<N>`` per warp. Standalone batched-GEMM
    measurements show 1.5-2.1x speedup; integrating into the
    fused-MoE pipeline benefits the down stage proportionally.

    This is the optimization-runbook §12.1.H lever, finally
    implemented (was a documented but silently-ignored knob through
    rounds 1-9 of the case study).

    Constraints:
    * ``hidden % tile_n == 0`` and ``intermediate % tile_k == 0`` (MoE
      models with 64- or 128-aligned hidden / intermediate satisfy
      this trivially).
    * Adds a one-time host-side preshuffle on first ``forward``; the
      result is cached against ``W_down.data_ptr()`` so steady-state
      cost is zero.
    """
    preshuffle_w_gate_up_packed: bool = False
    """Pre-shuffle the packed gate+up weights and use a
    ``preshuffle_b=True`` BatchedGemm for the gate-up stage when the
    packed (non-interleaved) path is selected.

    Only takes effect when ``use_experimental_interleaved_gate_up_silu``
    is False (i.e. the ``gu_concat`` packed path is active).
    """
    preshuffle_w_gate_up_interleaved: bool = False
    """Pre-shuffle the interleaved gate+up weights and use a
    ``preshuffle_b=True`` interleaved gate-up GEMM kernel.

    Takes effect when ``use_experimental_interleaved_gate_up_silu``
    is True (the production-default path). Same layout transform as
    :attr:`preshuffle_w_down`, applied to the interleaved gate-up
    weight tensor ``WGateUp[e, 2*i, :] = W_gate[e, i, :];
    WGateUp[e, 2*i+1, :] = W_up[e, i, :]``.
    """
    use_grouped_gemm: bool = True
    """Use the grouped sorted-token GEMM dispatch for the dynamic path.

    The default (batched) dynamic path pads every expert's GEMM slot to a
    uniform ``MAX_PADDED_M`` and dispatches one launch over ``E`` batches
    (``block_id_z = expert``). For sparse routing (E >> routed tokens per
    expert -- e.g. the datacenter shape T128/E32/K5: ~20 tokens/expert)
    that runs full ``MAX_PADDED_M``-row MFMA tiles per expert, of which
    most rows are padded zeros.

    With this flag on, the dynamic path instead packs the routed tokens
    into a dense tile-aligned layout (each expert's ``count[e]`` rows
    rounded up to ``tile_m``, concatenated) and dispatches the
    ``grouped=True`` interleaved-gate-up-silu + down-reduce kernels over a
    flat M-block grid where each block looks up its expert from a
    host-built ``BlockExpertIds`` array. Total GEMM rows collapse from
    ``E * MAX_PADDED_M`` to ``sum_e ceil(count[e]/tile_m) * tile_m``
    (~total routed tokens rounded to ``tile_m``).

    Measured on gfx950 / MI355X (perf_moe, best-of-5, datacenter
    T128/E32/K5/H8192/I8192): the GEMM row count drops ~2x and the
    forward speeds up correspondingly. Numerically parity-clean
    (max_abs unchanged). Strict win for sparse routing; for dense
    routing (every expert fills its tile) it degenerates to the same
    row count as the batched path.

    Only affects the dynamic (host-roundtrip) path; the static / decode
    path is unchanged.
    """
    active_tile_skip_gemms: bool = True
    """Use ``trait.active_tile_skip=True`` MoE GEMM kernels.

    Default-on after tuning on MI355X (gfx950): inactive expert slots
    (``-1`` sentinel rows in the over-padded static-offset layout) skip
    all MFMAs / LDS reads / HBM stores via one ``scf.if`` at CTA entry.
    Measured (perf_moe, best-of-5, graph-replay):

    * decode  T8 E8 K2 H4096 I7168  : 0.369 -> 0.325 ms  (1.14x)
    * sparse  T8 E32 K2 H4096 I7168 : 1.070 -> 0.473 ms  (2.26x)
    * datacenter T128 E32 K5 H8192 I8192 : 3.938 -> 3.929 ms (neutral;
      dense routing, no static offsets, compute-bound)

    Parity byte-identical (max_abs unchanged). Strict no-op for
    fully-active calls. Set False to restore the dense GEMM path.

    The kernel takes the existing ``sorted_token_ids_padded`` buffer
    plus the ``slot_size`` constant, and at CTA entry checks whether
    the first row of its (expert, m-tile) slot has a valid token id
    (``>= 0``). Inactive tiles (``-1`` sentinel) skip all MFMAs,
    LDS reads, and HBM stores via a single ``scf.if`` predicate. This
    is the runbook §17.6 active-tile dispatch lever; it is the
    largest measured win for sparse-routing decode shapes (E >>
    topk*tokens), where the canonical kernel runs full GEMM work
    for inactive expert slots that contribute nothing.

    Applied to the interleaved gate-up GEMM and the down BatchedGemm
    when those launchers are active. Composes with the preshuffle
    knobs (each preshuffle_b launcher gets its own active-tile
    variant). The flag is a strict no-op for fully-active calls
    (every expert receives at least one token); standalone tests
    show 0 % overhead in that case.
    """

    @property
    def total_pairs(self) -> int:
        return self.tokens * self.topk

    def to_topk_softmax_spec(self) -> TopkSoftmaxSpec:
        return TopkSoftmaxSpec(
            n_per_row=self.experts,
            k=self.topk,
            dtype="f32",
            out_dtype="f32",
            block_size=self.router_block_size,
        )

    def to_sort_spec(self) -> MoeSortingSpec:
        return MoeSortingSpec(
            tokens=self.tokens,
            topk=self.topk,
            experts=self.experts,
            block_size=self.sort_block_size,
        )

    def to_fused_moe_spec(self) -> FusedMoeSpec:
        return FusedMoeSpec(
            tokens=self.tokens,
            experts=self.experts,
            topk=self.topk,
            hidden=self.hidden,
            intermediate=self.intermediate,
            dtype=self.dtype,
            block_size=self.streaming_block_size,
            vec=self.streaming_vec,
        )

    def to_gemm_spec(self, *, name_suffix: str) -> GroupedGemmSpec:
        # Per-group spec used by the (deprecated, multi-launch)
        # :class:`GroupedGemmLauncher` path. Retained for migration /
        # comparison; the production path uses
        # :meth:`to_batched_gemm_spec` for a single-launch dispatch
        # over E batches.
        wave_size = _wave_size_for_arch(self.arch)
        trait = (
            TraitSpec(pipeline="mem", epilogue="default", pad_m=True)
            if wave_size == 32
            else TraitSpec(pad_m=True)
        )
        return GroupedGemmSpec(
            name=f"{self.name}_{name_suffix}",
            tile=self.gemm_tile,
            trait=trait,
            wave_size=wave_size,
            dtype=_gemm_dtype_to_universal(self.dtype),
        )

    def to_batched_gemm_spec_preshuffle_b(self) -> BatchedGemmSpec:
        """Variant of :meth:`to_batched_gemm_spec` with
        ``trait.preshuffle_b=True``.

        The orchestrator builds a parallel launcher for the down (and
        optionally gate-up packed) stage when ``preshuffle_w_down``
        / ``preshuffle_w_gate_up_packed`` are enabled, so the
        non-preshuffle launcher remains intact for any stage that
        does not have host-shuffled weights.
        """
        wave_size = _wave_size_for_arch(self.arch)
        if wave_size == 32:
            trait = TraitSpec(
                pipeline="mem", epilogue="default", pad_m=True, pad_n=True
            )
        else:
            trait = TraitSpec(pad_m=True, pad_n=True, preshuffle_b=True)
        return BatchedGemmSpec(
            name=f"{self.name}_batched_gemm",
            tile=self.gemm_tile,
            trait=trait,
            wave_size=wave_size,
            dtype=_gemm_dtype_to_universal(self.dtype),
        )

    def to_batched_gemm_spec(self) -> BatchedGemmSpec:
        """Single-launch batched GEMM spec for per-expert dispatch.

        With ``batched=True`` the kernel reads ``block_id_z`` as the
        batch (= expert) index and adds ``z * stride_X`` to A / B / C
        base ptrs. The orchestrator pads each expert's slot to a
        uniform ``MAX_PADDED_M`` and dispatches one launch over
        ``E`` batches in one go -- replacing the per-expert
        ``E``-launch loop in :class:`GroupedGemmLauncher`.

        ``pad_m=True`` is left on for safety (a no-op when
        ``MAX_PADDED_M`` is already a multiple of ``tile_m``, which
        the orchestrator guarantees). ``pad_n=True`` is required
        because the same kernel is reused for the down GEMM, where
        ``N = hidden`` may be smaller than ``tile_n`` (small / decode
        scenarios with wide-N gate-up tiles). Without the mask, the
        cshuffle epilogue's wide global stores would write columns
        ``[hidden, tile_n)`` past each row boundary and corrupt
        adjacent rows of the per-expert C buffer, producing NaN
        downstream in the topk-weighted reduce. The mask is
        vec-aligned and a no-op when ``hidden % store_vec == 0`` and
        ``tile_n`` divides the logical N (true for the gate-up GEMM
        with ``N = 2*I``). Tile dims and dtypes match
        :meth:`to_gemm_spec` so the same MFMA kernel body is reused.

        Note: tried ``chiplet_swizzle=True`` + ``waves_per_eu=2`` on
        MI355X (matching CK Tile's ``fmoe_<traits>_uk`` template
        default), measured no improvement over the simpler
        ``pad_m=True`` configuration on the benchmark scenarios; the
        swizzle is left off to keep the kernel name surface minimal
        and the spec readable. Future tuning can re-enable per
        scenario if measurements support it.
        """
        wave_size = _wave_size_for_arch(self.arch)
        trait = (
            TraitSpec(pipeline="mem", epilogue="default", pad_m=True, pad_n=True)
            if wave_size == 32
            else TraitSpec(pad_m=True, pad_n=True)
        )
        return BatchedGemmSpec(
            name=f"{self.name}_batched_gemm",
            tile=self.gemm_tile,
            trait=trait,
            wave_size=wave_size,
            dtype=_gemm_dtype_to_universal(self.dtype),
        )


class FusedMoeForward:
    """End-to-end fused-MoE forward driver.

    Composes the four production launchers (topk_softmax,
    moe_sorting, fused_moe streaming kernels, grouped-gemm) into one
    pipeline driven by chained
    :func:`~rocke.runtime.launcher.launch_kernel` calls plus a
    Python-side per-expert dispatch loop. Construct once per
    :class:`FusedMoeForwardSpec`; call :meth:`forward` repeatedly with
    fresh inputs / outputs (the launcher caches HSACOs and module
    handles for the lifetime of the instance).

    See module docstring for the call graph and per-expert dispatch
    contract.
    """

    def __init__(self, spec: FusedMoeForwardSpec) -> None:
        # Resolve the compilation target up front so the tile policy can
        # pick arch-legal MFMA atoms (gfx942 lacks the wide f16/bf16
        # atoms the gfx950 default tiles use).
        self.arch = _resolve_launch_arch(spec.arch)
        is_gfx942 = self.arch == "gfx942"
        is_gfx1250 = self.arch == "gfx1250"
        if is_gfx1250:
            if spec.gemm_tile == _default_gemm_tile():
                spec.gemm_tile = _default_gfx1250_wmma_gemm_tile()
            # Day-0 gfx1250 MoE uses the universal-GEMM WMMA-safe path.
            # The fused/interleaved/down-reduce kernels below are still
            # MFMA-specific.
            spec.use_experimental_fused_gate_up_silu = False
            spec.use_experimental_interleaved_gate_up_silu = False
            spec.use_experimental_fused_down_reduce = False
            spec.preshuffle_w_down = False
            spec.preshuffle_w_gate_up_packed = False
            spec.preshuffle_w_gate_up_interleaved = False
            spec.active_tile_skip_gemms = False
            spec.use_grouped_gemm = False
        # Shape-aware tile policy from the tuning sweep. Only override
        # the auto/default tile; caller-supplied tiles are respected.
        is_bf16 = spec.dtype in ("bf16",)
        if is_bf16 and spec.gemm_tile == _default_gemm_tile():
            # 32x32 atom is F16-only; switch to a BF16-compatible tile.
            # gfx942 has no wide 16x16x32 bf16 atom -> use the narrow one.
            spec.gemm_tile = (
                _default_bf16_gemm_tile_gfx942()
                if is_gfx942
                else _default_bf16_gemm_tile()
            )
        elif (
            spec.gemm_tile == _default_gemm_tile()
            and spec.hidden >= 1024
            and spec.tokens >= 32
        ):
            # Large-hidden, multi-token regime. Pick the tile by routing
            # DENSITY (average routed tokens per expert = T*K/E):
            #
            # * dense (avg_per_expert > 24, e.g. E=8 T128 K2 -> 32): the
            #   per-expert GEMM slot fills a 64-row tile, so the wide
            #   ``_large_batch_gemm_tile`` (tile_m=64) wins (measured
            #   0.82 vs 0.92 ms on T128/E8/K2/H4096/I7168).
            # * sparse (avg_per_expert <= 24, e.g. datacenter E=32 T128
            #   K5 -> 20): each expert's ~20 real rows sit in an
            #   otherwise-padded tile. ``tile_m=64`` does 3.2x of its
            #   MFMA work on padded zero rows; the
            #   ``_sparse_batch_gemm_tile`` (tile_m=32) halves the
            #   per-expert row rounding and is ~11% faster on the
            #   datacenter shape (3.91 -> 3.48 ms).
            #
            # Both tiles use the wide gfx950-only ``32x32x16`` f16 atom;
            # gfx942 falls back to the narrow-K atom variants.
            avg_per_expert = (spec.tokens * spec.topk) / max(1, spec.experts)
            sparse = avg_per_expert <= 24
            if sparse:
                spec.gemm_tile = (
                    _sparse_batch_gemm_tile_gfx942()
                    if is_gfx942
                    else _sparse_batch_gemm_tile()
                )
            else:
                spec.gemm_tile = (
                    _default_gemm_tile_gfx942()
                    if is_gfx942
                    else _large_batch_gemm_tile()
                )
        elif is_gfx942 and spec.gemm_tile == _default_gemm_tile():
            # f16 decode/small default uses the wide 32x32x16 atom;
            # gfx942 needs the narrow 32x32x8 atom.
            spec.gemm_tile = _default_gemm_tile_gfx942()
        self.spec = spec
        self._sort_launcher = MoeSortingLauncher(spec.to_sort_spec(), arch=self.arch)
        self._fused_moe_launcher = FusedMoeLauncher(
            spec.to_fused_moe_spec(), arch=self.arch
        )
        self._pool = WorkspacePool()
        self._topk_launcher: Optional[KernelLauncher] = None
        self._gemm_launcher: Optional[GroupedGemmLauncher] = None
        self._batched_gemm_launcher: Optional[KernelLauncher] = None
        self._batched_gemm_preshuffle_b_launcher: Optional[KernelLauncher] = None
        self._w_down_preshuffled: Optional[Any] = None
        self._w_down_preshuffled_key: Optional[Tuple[int, int, int]] = None
        self._gu_concat_preshuffled: Optional[Any] = None
        self._gu_concat_preshuffled_key: Optional[Tuple[int, int, int]] = None
        self._gu_interleaved_preshuffled: Optional[Any] = None
        self._gu_interleaved_preshuffled_key: Optional[Tuple[int, int, int]] = None
        self._interleaved_gate_up_silu_preshuffle_launcher: Optional[KernelLauncher] = (
            None
        )
        # Parameterized launcher cache for the active-tile-skip variants
        # (and combinations with preshuffle_b). Keyed on
        # ``(kind, preshuffle_b, active_tile_skip)`` so each unique
        # MoE-GEMM-trait combo lives in its own HSACO slot. ``kind``
        # is one of ``"batched"`` or ``"interleaved_gate_up_silu"``.
        self._moe_gemm_launcher_cache: dict = {}
        # Optional HIP graph capture for the static-mode forward.
        # When enabled, the first forward call warms up + captures a
        # graph; subsequent calls replay it, eliminating per-launch
        # HIP driver dispatch overhead. Only valid for benchmark
        # loops where the input tensor pointers are stable across
        # calls; production callers should leave this disabled.
        self._captured_graph: Optional[Any] = None
        self._captured_graph_stream: Optional[Any] = None
        # Fused gate+up+silu MFMA kernel: computes gate and up
        # accumulators in one kernel and writes Hidden directly,
        # eliminating the GateUpPacked HBM roundtrip and the separate
        # silu_mul_packed launch in static-offset mode.
        self._gate_up_silu_launcher: Optional[KernelLauncher] = None
        self._interleaved_gate_up_silu_launcher: Optional[KernelLauncher] = None
        self._down_reduce_launcher: Optional[KernelLauncher] = None
        self._static_scatter_gather_launcher: Optional[KernelLauncher] = None
        # Cached "G1U1" packed weights: torch.cat([W_gate, W_up], dim=1)
        # of shape (E, 2*I, H). Built lazily on first forward call,
        # cached against (W_gate.data_ptr(), W_up.data_ptr()) so the
        # concat doesn't happen again unless weights change. Mirrors
        # AITER's gate-up packing: one fused batched GEMM with
        # ``N = 2*I`` halves the A-bandwidth + cuts launch count by
        # one vs running gate and up as separate batched GEMMs.
        self._gu_concat: Optional[Any] = None
        self._gu_concat_key: Optional[Tuple[int, int]] = None
        self._gu_interleaved: Optional[Any] = None
        self._gu_interleaved_key: Optional[Tuple[int, int]] = None
        self._silu_mul_packed_launcher: Optional[KernelLauncher] = None
        # Cached static-mode resources (lazily allocated on first
        # static-mode forward call). The "static-offset mode" path
        # replaces the dynamic histogram + scan + host-roundtrip with
        # fixed per-expert offsets ``[0, slot_size, 2*slot_size, ...]``,
        # at the cost of allocating per-expert slots large enough to
        # fit the worst-case ``count[e]``. Eliminates the host
        # roundtrip + 2 sort kernels and lets the entire forward run
        # as a single ``launch_kernel`` chain.
        self._static_offsets: Optional[Any] = None
        # Heuristic: enable static-offset mode when ``T*K*E`` is small
        # enough that the "fixed slot_size = T*K" over-padding doesn't
        # cost much. Tuning sweep result:
        # * batch32 (T*K*E=512): static wins (0.5356 ms vs 0.8268 ms)
        # * prefill128 (T*K*E=2048): dynamic wins (0.9074 ms vs 1.3656 ms)
        # So 512 is the measured cutoff on MI355X for the current tile
        # family. ``spec.use_static_offsets`` lets decode shapes with a
        # tighter fixed-slot contract opt in explicitly.
        if spec.use_static_offsets is None:
            self._use_static_offsets = spec.tokens * spec.topk * spec.experts <= 512
        else:
            self._use_static_offsets = bool(spec.use_static_offsets)
        # When in static-offset mode, the slot_size is the smallest
        # ``tile_m``-aligned size that fits the worst-case
        # ``count[e] = T*K`` (all tokens routed to one expert). For
        # balanced routing the actual ``count[e]`` is much smaller,
        # but the GEMM only does meaningful work for the rows
        # corresponding to actual tokens (the rest are zero-init).
        tile_m = spec.gemm_tile.tile_m
        default_static_slot_size = (
            (spec.tokens * spec.topk + tile_m - 1) // tile_m
        ) * tile_m
        if default_static_slot_size < tile_m:
            default_static_slot_size = tile_m
        if spec.static_slot_size is None:
            self._static_slot_size = default_static_slot_size
        else:
            if spec.static_slot_size <= 0:
                raise ValueError(
                    f"static_slot_size must be > 0 (got {spec.static_slot_size})"
                )
            self._static_slot_size = int(spec.static_slot_size)

    # ------------------------------------------------------------------
    # Lazy compile
    # ------------------------------------------------------------------

    def _ensure_topk_launcher(self) -> KernelLauncher:
        if self._topk_launcher is None:
            spec = self.spec.to_topk_softmax_spec()
            artifact = compile_kernel(
                build_topk_softmax(spec), arch=self.arch, capture_ir_text=False
            )
            self._topk_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=topk_softmax_signature(spec),
                cache_key=("topk_softmax", spec.kernel_name()),
            )
        return self._topk_launcher

    def _ensure_gemm_launcher(self) -> GroupedGemmLauncher:
        if self._gemm_launcher is None:
            gemm_spec = self.spec.to_gemm_spec(name_suffix="gemm")
            artifact = compile_kernel(
                build_grouped_gemm(gemm_spec), arch=self.arch, capture_ir_text=False
            )
            self._gemm_launcher = GroupedGemmLauncher(
                hsaco=artifact.hsaco, spec=gemm_spec
            )
        return self._gemm_launcher

    def _ensure_batched_gemm_launcher(self) -> KernelLauncher:
        """Compile + cache the single batched GEMM kernel.

        The same kernel is reused for all 3 stages (gate / up /
        down). Each call dispatches one launch over ``E`` batches
        (= experts), with per-expert offsets supplied via
        ``stride_a / stride_b / stride_c`` and ``block_id_z`` mapping
        to the expert index. This collapses the per-expert
        ``E``-launch loop down to ``3`` total GEMM launches.
        """
        if self._batched_gemm_launcher is None:
            spec = self.spec.to_batched_gemm_spec()
            artifact = compile_kernel(
                build_batched_gemm(spec), arch=self.arch, capture_ir_text=False
            )
            self._batched_gemm_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=batched_gemm_signature(spec),
                cache_key=("batched_gemm", spec.kernel_name()),
            )
        return self._batched_gemm_launcher

    def _grouped_gate_up_silu_launcher(self) -> KernelLauncher:
        """Compile + cache the grouped dual-B gate+up+silu kernel.

        Uses the dual-B kernel (separate ``W_gate`` / ``W_up`` pointers)
        rather than the interleaved single-B kernel so the grouped path
        needs no host-built ``(E, 2I, H)`` interleaved weight tensor (an
        extra ~8 GiB at the datacenter shape); it reads the caller's
        ``W_gate`` / ``W_up`` slabs directly.
        """
        key = ("grouped_gate_up_silu",)
        cached = self._moe_gemm_launcher_cache.get(key)
        if cached is not None:
            return cached
        spec = self._grouped_gate_up_spec()
        artifact = compile_kernel(
            build_moe_gate_up_silu_gemm(spec, arch=self.arch),
            arch=self.arch,
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=moe_gate_up_silu_gemm_signature(spec),
            cache_key=("moe_grouped_gate_up_silu", spec.kernel_name()),
        )
        self._moe_gemm_launcher_cache[key] = launcher
        return launcher

    def _grouped_gate_up_spec(self) -> FusedGateUpSiluGemmSpec:
        return FusedGateUpSiluGemmSpec(
            name=f"{self.spec.name}_gate_up_silu",
            tile=self.spec.gemm_tile,
            trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
            dtype=_gemm_dtype_to_universal(self.spec.dtype),
            grouped=True,
        )

    def _grouped_down_reduce_launcher(self) -> KernelLauncher:
        """Compile + cache the grouped down-reduce kernel."""
        key = ("grouped_down_reduce",)
        cached = self._moe_gemm_launcher_cache.get(key)
        if cached is not None:
            return cached
        spec = FusedDownReduceGemmSpec(
            name=f"{self.spec.name}_down_reduce",
            tile=self.spec.gemm_tile,
            trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
            dtype=_gemm_dtype_to_universal(self.spec.dtype),
            grouped=True,
        )
        artifact = compile_kernel(
            build_moe_down_reduce_gemm(spec, arch=self.arch),
            arch=self.arch,
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=moe_down_reduce_gemm_signature(spec),
            cache_key=("moe_grouped_down_reduce", spec.kernel_name()),
        )
        self._moe_gemm_launcher_cache[key] = launcher
        return launcher

    def _moe_batched_gemm_launcher(
        self, *, preshuffle_b: bool, active_tile_skip: bool
    ) -> KernelLauncher:
        """Return a cached BatchedGemm launcher for any combination of
        ``preshuffle_b`` and ``active_tile_skip`` traits.

        Builds the spec on first use, compiles to HSACO, and caches the
        :class:`KernelLauncher` keyed on the trait combination so each
        unique kernel binary lives in exactly one slot for the lifetime
        of this :class:`FusedMoeForward`.
        """
        key = ("batched", bool(preshuffle_b), bool(active_tile_skip))
        cached = self._moe_gemm_launcher_cache.get(key)
        if cached is not None:
            return cached
        wave_size = _wave_size_for_arch(self.spec.arch)
        trait = TraitSpec(
            pipeline="mem" if wave_size == 32 else "compv3",
            epilogue="default" if wave_size == 32 else "cshuffle",
            pad_m=True,
            pad_n=True,
            preshuffle_b=False if wave_size == 32 else preshuffle_b,
            active_tile_skip=False if wave_size == 32 else active_tile_skip,
        )
        spec = BatchedGemmSpec(
            name=f"{self.spec.name}_batched_gemm",
            tile=self.spec.gemm_tile,
            trait=trait,
            wave_size=wave_size,
            dtype=_gemm_dtype_to_universal(self.spec.dtype),
        )
        artifact = compile_kernel(
            build_batched_gemm(spec), arch=self.arch, capture_ir_text=False
        )
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=batched_gemm_signature(spec),
            cache_key=("batched_gemm_moe", spec.kernel_name()),
        )
        self._moe_gemm_launcher_cache[key] = launcher
        return launcher

    def _moe_interleaved_gate_up_silu_launcher(
        self, *, preshuffle_b: bool, active_tile_skip: bool
    ) -> KernelLauncher:
        """Return a cached interleaved-gate+up+silu launcher for any
        combination of ``preshuffle_b`` and ``active_tile_skip`` traits.

        Same caching contract as
        :meth:`_moe_batched_gemm_launcher` but for the
        :func:`build_moe_interleaved_gate_up_silu_gemm` body.
        """
        key = (
            "interleaved_gate_up_silu",
            bool(preshuffle_b),
            bool(active_tile_skip),
        )
        cached = self._moe_gemm_launcher_cache.get(key)
        if cached is not None:
            return cached
        trait = TraitSpec(
            pad_m=True,
            pad_n=True,
            epilogue="default",
            preshuffle_b=preshuffle_b,
            active_tile_skip=active_tile_skip,
        )
        spec = FusedInterleavedGateUpSiluGemmSpec(
            name=f"{self.spec.name}_interleaved_gate_up_silu",
            tile=self.spec.gemm_tile,
            trait=trait,
            dtype=_gemm_dtype_to_universal(self.spec.dtype),
        )
        artifact = compile_kernel(
            build_moe_interleaved_gate_up_silu_gemm(spec, arch=self.arch),
            arch=self.arch,
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=moe_interleaved_gate_up_silu_gemm_signature(spec),
            cache_key=("moe_interleaved_gate_up_silu_moe", spec.kernel_name()),
        )
        self._moe_gemm_launcher_cache[key] = launcher
        return launcher

    def _ensure_batched_gemm_preshuffle_b_launcher(self) -> KernelLauncher:
        """Compile + cache the ``preshuffle_b=True`` batched GEMM kernel.

        Used by stages whose B operand has been pre-shuffled on the
        host (``preshuffle_w_down`` and ``preshuffle_w_gate_up_packed``).
        Kernel name carries the ``preb`` flag so this binary lives in
        a separate cache slot from the canonical kernel.
        """
        if self._batched_gemm_preshuffle_b_launcher is None:
            spec = self.spec.to_batched_gemm_spec_preshuffle_b()
            artifact = compile_kernel(
                build_batched_gemm(spec), arch=self.arch, capture_ir_text=False
            )
            self._batched_gemm_preshuffle_b_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=batched_gemm_signature(spec),
                cache_key=("batched_gemm_preshuffle_b", spec.kernel_name()),
            )
        return self._batched_gemm_preshuffle_b_launcher

    @staticmethod
    def _host_preshuffle_b(W: Any, block_n: int, block_k: int) -> Any:
        """Re-arrange ``(E, N, K)`` row-major B  ->  ``(E, k_tiles, n_tiles,
        block_n, block_k)`` contiguous, the layout the
        ``preshuffle_b=True`` BatchedGemm expects.

        The returned tensor has the same total element count and
        per-batch byte size as the input -- only the in-tile element
        order changes.
        """
        E, N, K = W.shape
        if N % block_n or K % block_k:
            raise ValueError(
                f"preshuffle_b requires N({N}) % block_n({block_n}) == 0 and "
                f"K({K}) % block_k({block_k}) == 0"
            )
        n_tiles = N // block_n
        k_tiles = K // block_k
        return (
            W.view(E, n_tiles, block_n, k_tiles, block_k)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )

    def _ensure_w_down_preshuffled(self, W_down: Any) -> Any:
        """Cache + return the host-preshuffled W_down for the down stage.

        Keyed on ``(data_ptr, block_n, block_k)`` so reusing the same
        weights across calls (the production-inference pattern) reuses
        the cached preshuffle; passing different weights or changing
        the GEMM tile triggers a re-shuffle.
        """
        block_n = self.spec.gemm_tile.tile_n
        block_k = self.spec.gemm_tile.tile_k
        key = (int(W_down.data_ptr()), block_n, block_k)
        if self._w_down_preshuffled is not None and self._w_down_preshuffled_key == key:
            return self._w_down_preshuffled
        self._w_down_preshuffled = self._host_preshuffle_b(W_down, block_n, block_k)
        self._w_down_preshuffled_key = key
        return self._w_down_preshuffled

    def _ensure_gu_concat_preshuffled(self, W_gate: Any, W_up: Any) -> Any:
        """Cache + return host-preshuffled gate-up packed weights.

        Builds ``torch.cat([W_gate, W_up], dim=1)`` and preshuffles in
        one step (no intermediate canonical-layout cache).
        """
        import torch

        block_n = self.spec.gemm_tile.tile_n
        block_k = self.spec.gemm_tile.tile_k
        key = (int(W_gate.data_ptr()), int(W_up.data_ptr()), block_n)
        if (
            self._gu_concat_preshuffled is not None
            and self._gu_concat_preshuffled_key == key
        ):
            return self._gu_concat_preshuffled
        gu = torch.cat([W_gate, W_up], dim=1).contiguous()
        self._gu_concat_preshuffled = self._host_preshuffle_b(gu, block_n, block_k)
        self._gu_concat_preshuffled_key = key
        return self._gu_concat_preshuffled

    def _ensure_silu_mul_packed_launcher(self) -> KernelLauncher:
        """Compile + cache the packed silu_mul kernel.

        Variant that consumes a single ``(M, 2*I)`` GateUp buffer
        (the "G1U1" gate-up packing) instead of two separate
        ``(M, I)`` GateOut / UpOut buffers. Pairs with the fused
        gate+up batched GEMM whose ``N = 2*I`` and
        ``W = torch.cat([W_gate, W_up], dim=1)``.
        """
        if self._silu_mul_packed_launcher is None:
            fmoe_spec = self.spec.to_fused_moe_spec()
            artifact = compile_kernel(
                build_moe_silu_mul_packed(fmoe_spec),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._silu_mul_packed_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_silu_mul_packed_signature(fmoe_spec),
                cache_key=(
                    "moe_silu_mul_packed",
                    fmoe_spec.kernel_name("silu_mul_packed"),
                ),
            )
        return self._silu_mul_packed_launcher

    def _ensure_gate_up_silu_launcher(self) -> KernelLauncher:
        """Compile + cache fused gate+up GEMM with SiLU epilogue."""
        if self._gate_up_silu_launcher is None:
            spec = FusedGateUpSiluGemmSpec(
                name=f"{self.spec.name}_gate_up_silu",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
                dtype=_gemm_dtype_to_universal(self.spec.dtype),
            )
            artifact = compile_kernel(
                build_moe_gate_up_silu_gemm(spec, arch=self.arch),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._gate_up_silu_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_gate_up_silu_gemm_signature(spec),
                cache_key=("moe_gate_up_silu", spec.kernel_name()),
            )
        return self._gate_up_silu_launcher

    def _ensure_interleaved_gate_up_silu_launcher(self) -> KernelLauncher:
        """Compile + cache interleaved gate/up GEMM with SiLU epilogue."""
        if self._interleaved_gate_up_silu_launcher is None:
            spec = FusedInterleavedGateUpSiluGemmSpec(
                name=f"{self.spec.name}_interleaved_gate_up_silu",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
                dtype=_gemm_dtype_to_universal(self.spec.dtype),
            )
            artifact = compile_kernel(
                build_moe_interleaved_gate_up_silu_gemm(spec, arch=self.arch),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._interleaved_gate_up_silu_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_interleaved_gate_up_silu_gemm_signature(spec),
                cache_key=("moe_interleaved_gate_up_silu", spec.kernel_name()),
            )
        return self._interleaved_gate_up_silu_launcher

    def _ensure_interleaved_gate_up_silu_preshuffle_launcher(
        self,
    ) -> KernelLauncher:
        """Compile + cache the ``preshuffle_b=True`` variant of the
        interleaved gate/up GEMM kernel.

        Same kernel body as :meth:`_ensure_interleaved_gate_up_silu_launcher`
        but emits the wide contiguous B-load path (``b_global_load_vN`` on
        a host-preshuffled tile buffer) instead of the strided per-row
        path.
        """
        if self._interleaved_gate_up_silu_preshuffle_launcher is None:
            spec = FusedInterleavedGateUpSiluGemmSpec(
                name=f"{self.spec.name}_interleaved_gate_up_silu",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(
                    pad_m=True,
                    pad_n=True,
                    epilogue="default",
                    preshuffle_b=True,
                ),
                dtype=_gemm_dtype_to_universal(self.spec.dtype),
            )
            artifact = compile_kernel(
                build_moe_interleaved_gate_up_silu_gemm(spec, arch=self.arch),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._interleaved_gate_up_silu_preshuffle_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_interleaved_gate_up_silu_gemm_signature(spec),
                cache_key=(
                    "moe_interleaved_gate_up_silu_preshuffle_b",
                    spec.kernel_name(),
                ),
            )
        return self._interleaved_gate_up_silu_preshuffle_launcher

    def _ensure_gu_interleaved_preshuffled(self, W_gate: Any, W_up: Any) -> Any:
        """Cache + return the host-preshuffled interleaved gate-up
        weights for the ``preshuffle_b=True`` interleaved kernel.

        Builds the canonical interleaved layout
        (``out[:, 2*i, :] = W_gate[:, i, :], out[:, 2*i+1, :] = W_up[:, i, :]``)
        and applies the
        ``(E, k_tiles, n_tiles, block_n, block_k)`` shuffle.
        """
        import torch

        block_n = self.spec.gemm_tile.tile_n
        block_k = self.spec.gemm_tile.tile_k
        key = (int(W_gate.data_ptr()), int(W_up.data_ptr()), block_n)
        if (
            self._gu_interleaved_preshuffled is not None
            and self._gu_interleaved_preshuffled_key == key
        ):
            return self._gu_interleaved_preshuffled
        gu = (
            torch.stack((W_gate, W_up), dim=2)
            .reshape(W_gate.shape[0], 2 * W_gate.shape[1], W_gate.shape[2])
            .contiguous()
        )
        self._gu_interleaved_preshuffled = self._host_preshuffle_b(gu, block_n, block_k)
        self._gu_interleaved_preshuffled_key = key
        return self._gu_interleaved_preshuffled

    def _ensure_down_reduce_launcher(self) -> KernelLauncher:
        """Compile + cache fused down GEMM with weighted atomic epilogue."""
        if self._down_reduce_launcher is None:
            spec = FusedDownReduceGemmSpec(
                name=f"{self.spec.name}_down_reduce",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
                dtype=_gemm_dtype_to_universal(self.spec.dtype),
            )
            artifact = compile_kernel(
                build_moe_down_reduce_gemm(spec, arch=self.arch),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._down_reduce_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_down_reduce_gemm_signature(spec),
                cache_key=("moe_down_reduce", spec.kernel_name()),
            )
        return self._down_reduce_launcher

    def _ensure_static_scatter_gather_launcher(self) -> KernelLauncher:
        """Compile + cache static scatter+gather fused streaming kernel."""
        if self._static_scatter_gather_launcher is None:
            spec = self.spec.to_fused_moe_spec()
            artifact = compile_kernel(
                build_moe_static_scatter_gather(spec),
                arch=self.arch,
                capture_ir_text=False,
            )
            self._static_scatter_gather_launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=moe_static_scatter_gather_signature(spec),
                cache_key=(
                    "moe_static_scatter_gather",
                    spec.kernel_name("static_scatter_gather"),
                ),
            )
        return self._static_scatter_gather_launcher

    def _ensure_gu_concat(self, W_gate: Any, W_up: Any) -> Any:
        """Build / return the cached gate-up packed weight tensor.

        Concatenates ``W_gate`` and ``W_up`` along the intermediate
        axis (dim=1), producing ``(E, 2*I, H)``. The concat is cached
        against the pair of input data_ptrs so reusing the same
        weights across calls (the production-inference pattern)
        reuses the cached result; passing different weights triggers
        a re-concat.
        """
        import torch

        key = (int(W_gate.data_ptr()), int(W_up.data_ptr()))
        if self._gu_concat is not None and self._gu_concat_key == key:
            return self._gu_concat
        self._gu_concat = torch.cat([W_gate, W_up], dim=1).contiguous()
        self._gu_concat_key = key
        return self._gu_concat

    def _ensure_gu_interleaved(self, W_gate: Any, W_up: Any) -> Any:
        """Build / return interleaved gate-up packed weights.

        Shape ``(E, 2*I, H)`` with columns ``gate_i, up_i`` adjacent:
        ``out[:, 2*i, :] = W_gate[:, i, :]`` and
        ``out[:, 2*i + 1, :] = W_up[:, i, :]``.
        """
        import torch

        key = (int(W_gate.data_ptr()), int(W_up.data_ptr()))
        if self._gu_interleaved is not None and self._gu_interleaved_key == key:
            return self._gu_interleaved
        self._gu_interleaved = (
            torch.stack((W_gate, W_up), dim=2)
            .reshape(W_gate.shape[0], 2 * W_gate.shape[1], W_gate.shape[2])
            .contiguous()
        )
        self._gu_interleaved_key = key
        return self._gu_interleaved

    def _ensure_compiled(self) -> None:
        """Compile all components up front (idempotent).

        Useful for benchmark harnesses that want to keep the compile
        cost out of the timed region. :meth:`forward` will lazy-compile
        on first call regardless.
        """
        self._ensure_topk_launcher()
        self._sort_launcher._ensure_launchers()
        self._fused_moe_launcher._ensure_launchers()
        self._ensure_batched_gemm_launcher()
        if self.spec.preshuffle_w_down or self.spec.preshuffle_w_gate_up_packed:
            self._ensure_batched_gemm_preshuffle_b_launcher()
        use_fused = bool(self.spec.use_experimental_fused_gate_up_silu)
        use_interleaved = bool(self.spec.use_experimental_interleaved_gate_up_silu)
        if use_fused:
            self._ensure_gate_up_silu_launcher()
        elif use_interleaved:
            if self.spec.preshuffle_w_gate_up_interleaved:
                self._ensure_interleaved_gate_up_silu_preshuffle_launcher()
            else:
                self._ensure_interleaved_gate_up_silu_launcher()
        else:
            self._ensure_silu_mul_packed_launcher()
        if self.spec.use_experimental_fused_down_reduce:
            self._ensure_down_reduce_launcher()
        self._ensure_static_scatter_gather_launcher()

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def _workspace_specs(self, device: Any) -> Tuple[WorkspaceSpec, ...]:
        s = self.spec
        import torch

        i32 = torch.int32
        f32 = torch.float32
        act = _torch_dtype_for(s.dtype)
        return (
            WorkspaceSpec("TopkIds", (s.tokens, s.topk), i32, device),
            WorkspaceSpec("TopkWeights", (s.tokens, s.topk), f32, device),
            WorkspaceSpec("Hist", (s.experts,), i32, device),
            WorkspaceSpec("Counter", (s.experts,), i32, device),
            WorkspaceSpec("Offsets", (s.experts,), i32, device),
            WorkspaceSpec("Counts", (s.experts,), i32, device),
            WorkspaceSpec("SortedTokenIds", (s.total_pairs,), i32, device),
            WorkspaceSpec("SortedTopkIds", (s.total_pairs,), i32, device),
            WorkspaceSpec("SortedWeights", (s.total_pairs,), f32, device),
            WorkspaceSpec("GroupedInput", (s.total_pairs, s.hidden), act, device),
            WorkspaceSpec("GateOut", (s.total_pairs, s.intermediate), act, device),
            WorkspaceSpec("UpOut", (s.total_pairs, s.intermediate), act, device),
            WorkspaceSpec("Hidden", (s.total_pairs, s.intermediate), act, device),
            WorkspaceSpec("DownOut", (s.total_pairs, s.hidden), act, device),
            WorkspaceSpec("Y_f32", (s.tokens, s.hidden), f32, device),
        )

    def _build_per_expert_problems(
        self,
        *,
        counts_cpu: Any,
        offsets_cpu: Any,
        a_buf: Any,
        b_weights: Any,
        c_buf: Any,
        a_inner_dim: int,
        b_inner_dim: int,
        c_inner_dim: int,
        elem_bytes: int,
    ):
        """Build a list of :class:`GroupedGemmProblem` for one stage.

        ``a_buf`` and ``c_buf`` are flat ``(total_pairs, ...)`` tensors;
        per-expert offsets address them via pointer arithmetic.
        ``b_weights`` is a ``(experts, ...)`` tensor; per-expert ``B``
        addresses are computed as ``b_weights[e].data_ptr()``.
        """
        problems = []
        for e in range(self.spec.experts):
            count = int(counts_cpu[e])
            if count <= 0:
                continue
            offset = int(offsets_cpu[e])
            problems.append(
                GroupedGemmProblem(
                    M=count,
                    N=c_inner_dim,
                    K=b_inner_dim,
                    A_ptr=int(a_buf.data_ptr()) + offset * a_inner_dim * elem_bytes,
                    B_ptr=int(b_weights[e].data_ptr()),
                    C_ptr=int(c_buf.data_ptr()) + offset * c_inner_dim * elem_bytes,
                )
            )
        return problems

    # ------------------------------------------------------------------
    # Grouped sorted-token GEMM dispatch (de-padded dynamic path)
    # ------------------------------------------------------------------

    def _dispatch_grouped_gemm(
        self,
        *,
        counts_cpu: Any,
        offsets_cpu: Any,
        grouped_input: Any,
        sorted_token_ids: Any,
        sorted_weights: Any,
        y_f32: Any,
        W_gate: Any,
        W_up: Any,
        W_down: Any,
        Y: Any,
        device: Any,
        stream: int,
    ) -> bool:
        """De-padded grouped GEMM for the dynamic path.

        Packs the routed tokens into a dense, ``tile_m``-aligned per-expert
        layout (each expert's ``count[e]`` rows rounded up to ``tile_m``,
        concatenated) and dispatches the ``grouped=True`` interleaved
        gate+up+silu and down-reduce kernels over a flat M-block grid. Each
        M-block looks up its expert from a host-built ``BlockExpertIds``
        array; total GEMM rows = ``num_m_blocks * tile_m`` (~total routed
        tokens rounded to ``tile_m``) instead of ``E * MAX_PADDED_M``.

        Returns True when it handled the forward (the caller returns),
        False to fall back to the batched padded path (e.g. degenerate
        empty routing).
        """
        import torch

        from ...runtime.launcher import WorkspaceSpec as _WS

        s = self.spec
        tile_m = s.gemm_tile.tile_m
        # Only the M (token) axis is packed here; the N axis (hidden /
        # intermediate columns) is tiled inside the GEMM kernel's grid, so
        # tile_n is not needed in this host-side de-padding packer.

        # Per-expert block counts + dense packed block layout.
        blocks_per_expert = [
            (int(counts_cpu[e]) + tile_m - 1) // tile_m for e in range(s.experts)
        ]
        num_m_blocks = sum(blocks_per_expert)
        if num_m_blocks == 0:
            # No tokens routed anywhere; output stays zero.
            Y.copy_(y_f32.to(Y.dtype))
            return True
        total_packed = num_m_blocks * tile_m

        act_dtype = grouped_input.dtype
        grouped_input_packed = self._pool.get_spec(
            _WS("GroupedInputPacked", (total_packed, s.hidden), act_dtype, device)
        )
        hidden_packed = self._pool.get_spec(
            _WS("HiddenPacked", (total_packed, s.intermediate), act_dtype, device)
        )
        sorted_token_ids_packed = self._pool.get_spec(
            _WS("SortedTokenIdsPacked", (total_packed,), torch.int32, device)
        )
        sorted_weights_packed = self._pool.get_spec(
            _WS("SortedWeightsPacked", (total_packed,), torch.float32, device)
        )
        block_expert_ids = self._pool.get_spec(
            _WS("BlockExpertIds", (num_m_blocks,), torch.int32, device)
        )

        # Padded tail rows of partial tiles must read zero A (so their
        # gate/up GEMM contributes nothing) and carry token id -1 (so the
        # down-reduce skips their atomic write). ``hidden_packed`` is fully
        # overwritten by the gate-up kernel, so it needs no pre-zero.
        grouped_input_packed.zero_()
        sorted_token_ids_packed.fill_(-1)
        sorted_weights_packed.zero_()

        # Host-build BlockExpertIds and the dense per-expert copies. The
        # copies reuse the existing per-expert host loop pattern (one
        # contiguous slice copy per non-empty expert).
        block_expert_cpu = torch.empty(num_m_blocks, dtype=torch.int32)
        blk = 0
        for e in range(s.experts):
            be = blocks_per_expert[e]
            if be == 0:
                continue
            block_expert_cpu[blk : blk + be] = e
            ce = int(counts_cpu[e])
            uoff = int(offsets_cpu[e])
            poff = blk * tile_m
            grouped_input_packed[poff : poff + ce].copy_(
                grouped_input[uoff : uoff + ce]
            )
            sorted_token_ids_packed[poff : poff + ce].copy_(
                sorted_token_ids[uoff : uoff + ce]
            )
            sorted_weights_packed[poff : poff + ce].copy_(
                sorted_weights[uoff : uoff + ce]
            )
            blk += be
        block_expert_ids.copy_(block_expert_cpu)

        gate_up_launcher = self._grouped_gate_up_silu_launcher()
        down_launcher = self._grouped_down_reduce_launcher()

        gate_up_grid = moe_gate_up_silu_gemm_grouped_grid(
            num_m_blocks, s.intermediate, self._grouped_gate_up_spec()
        )
        down_grid = moe_down_reduce_gemm_grouped_grid(
            num_m_blocks, s.hidden, self._grouped_down_spec()
        )
        gemm_block = (self.spec.to_batched_gemm_spec().block_size, 1, 1)

        gate_up_callable = make_kernel(
            gate_up_launcher,
            {
                "A": grouped_input_packed,
                "WGate": W_gate,
                "WUp": W_up,
                "Hidden": hidden_packed,
                "M": total_packed,
                "N": s.intermediate,
                "K": s.hidden,
                "stride_a": 0,  # dense packed A (no per-expert stride)
                "stride_b": s.intermediate * s.hidden,
                "stride_c": 0,  # dense packed Hidden
                "BlockExpertIds": block_expert_ids,
            },
            gate_up_grid,
            gemm_block,
        )
        down_callable = make_kernel(
            down_launcher,
            {
                "A": hidden_packed,
                "WDown": W_down,
                "SortedTokenIds": sorted_token_ids_packed,
                "SortedWeights": sorted_weights_packed,
                "Y": y_f32,
                "M": total_packed,
                "N": s.hidden,
                "K": s.intermediate,
                "stride_a": 0,  # dense packed Hidden
                "stride_b": s.hidden * s.intermediate,
                "slot_size": 0,  # unused in grouped mode
                "tokens": s.tokens,
                "BlockExpertIds": block_expert_ids,
            },
            down_grid,
            gemm_block,
        )
        launch_kernel(
            StreamConfig(stream_id=int(stream)),
            gate_up_callable,
            down_callable,
        )
        Y.copy_(y_f32.to(Y.dtype))
        return True

    def _grouped_down_spec(self) -> FusedDownReduceGemmSpec:
        return FusedDownReduceGemmSpec(
            name=f"{self.spec.name}_down_reduce",
            tile=self.spec.gemm_tile,
            trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
            dtype=_gemm_dtype_to_universal(self.spec.dtype),
            grouped=True,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        routing_logits: Any,  # (T, E) f32
        X: Any,  # (T, H) act dtype
        W_gate: Any,  # (E, I, H) act dtype, row-major
        W_up: Any,  # (E, I, H) act dtype, row-major
        W_down: Any,  # (E, H, I) act dtype, row-major
        Y: Any,  # (T, H) act dtype output, populated in-place
        stream: int = 0,
    ) -> None:
        if self._use_static_offsets:
            self._forward_static(
                routing_logits=routing_logits,
                X=X,
                W_gate=W_gate,
                W_up=W_up,
                W_down=W_down,
                Y=Y,
                stream=stream,
            )
            return
        self._forward_dynamic(
            routing_logits=routing_logits,
            X=X,
            W_gate=W_gate,
            W_up=W_up,
            W_down=W_down,
            Y=Y,
            stream=stream,
        )

    def _forward_dynamic(
        self,
        *,
        routing_logits: Any,
        X: Any,
        W_gate: Any,
        W_up: Any,
        W_down: Any,
        Y: Any,
        stream: int = 0,
    ) -> None:
        """Run one fused-MoE forward, in-place into ``Y``.

        Issues 5 streaming-kernel launches plus 3*E grouped-GEMM
        launches on ``stream``. Includes a small device-to-host copy
        of ``(experts,)`` i32 ``Counts`` and ``Offsets`` arrays after
        the sort completes (needed to construct the per-expert GEMM
        argument lists). Returns after the final cast ``Y_f32 -> Y``;
        the caller is responsible for any further sync via
        ``torch.cuda.synchronize()`` if reading ``Y`` on the host.

        Per-expert offset alignment
        ---------------------------
        The per-expert grouped-GEMM kernel uses ``pad_m=True`` (since
        ``count[e]`` is a runtime-variable that need not divide
        ``tile_m``). On AMDGPU MFMA the kernel writes to all
        ``tile_m`` rows of the destination tile regardless of whether
        a row is "real" or "padded" -- the masking is on the
        accumulator side, not the store side. To prevent expert
        ``e``'s padded write from clobbering expert ``e+1``'s slot,
        the orchestrator allocates each expert's slot at a
        ``tile_m``-aligned offset and pads the gap by zero-init +
        a re-gather step that copies the count[e] real rows from the
        sort's contiguous output into the padded layout. The
        downstream ``silu_mul`` / ``topk_reduce`` kernels then run
        on the padded layout: padded rows have all-zero inputs (so
        their silu output is 0), and ``topk_reduce`` skips entries
        where ``SortedTokenIds[b] == -1`` (which we set on the
        padded slots).
        """
        import torch

        self._ensure_compiled()
        s = self.spec
        tile_m = self.spec.gemm_tile.tile_m

        device = X.device
        ws = self._pool.prepare(self._workspace_specs(device))
        topk_ids = ws["TopkIds"]
        topk_weights = ws["TopkWeights"]
        hist = ws["Hist"]
        counter = ws["Counter"]
        offsets = ws["Offsets"]
        counts = ws["Counts"]
        sorted_token_ids = ws["SortedTokenIds"]
        sorted_topk_ids = ws["SortedTopkIds"]
        sorted_weights = ws["SortedWeights"]
        grouped_input = ws["GroupedInput"]
        gate_out = ws["GateOut"]
        up_out = ws["UpOut"]
        hidden_buf = ws["Hidden"]
        down_out = ws["DownOut"]
        y_f32 = ws["Y_f32"]

        # Pre-zero the buffers that act as accumulators (atomic-add
        # targets in the histogram + scatter counters and the topk-
        # reduce output).
        hist.zero_()
        counter.zero_()
        y_f32.zero_()
        # Pre-zero per-expert GEMM A / C buffers; zeroing the input
        # buffer makes the ``pad_m=True`` GEMM kernel's reads of
        # padded rows contribute zero, and zeroing the output buffer
        # keeps the inter-expert padded-write region clean across
        # subsequent expert dispatches.
        grouped_input.zero_()
        gate_out.zero_()
        up_out.zero_()
        hidden_buf.zero_()
        down_out.zero_()

        topk_launcher = self._ensure_topk_launcher()
        # The grouped (single-launch-per-expert) GEMM launcher is
        # warmed up here so the first real forward call doesn't pay
        # the compile cost. It's referenced only via
        # ``self._gemm_launcher`` further down (the dynamic path
        # currently uses the batched single-launch dispatch instead,
        # but we keep the warm-up so a future toggle back to the
        # grouped launcher pays no compile penalty mid-loop).
        if self.spec.use_grouped_gemm:
            self._ensure_gemm_launcher()

        # ---------------- Stage 1 + 2: router + sort ----------------
        # 1 launch (topk-softmax) + 3 launches (sort) chained on ``stream``
        # via :func:`launch_kernel`. The closures capture the input /
        # output pointers; HIP's same-stream FIFO ordering preserves
        # the data flow (topk_ids -> hist -> offsets -> sorted arrays).
        router_grid = topk_softmax_grid(s.tokens, s.to_topk_softmax_spec())
        router_block = (s.router_block_size, 1, 1)
        sort_callables = self._sort_launcher.make_callables(
            {
                "histogram": {
                    "TopkIds": topk_ids,
                    "Hist": hist,
                    "num_pairs": s.total_pairs,
                    "num_experts": s.experts,
                },
                "scan": {
                    "Hist": hist,
                    "Offsets": offsets,
                    "Counts": counts,
                    "num_experts": s.experts,
                },
                "scatter": {
                    "TopkIds": topk_ids,
                    "TopkWeights": topk_weights,
                    "Offsets": offsets,
                    "Counter": counter,
                    "SortedTokenIds": sorted_token_ids,
                    "SortedTopkIds": sorted_topk_ids,
                    "SortedWeights": sorted_weights,
                    "tokens": s.tokens,
                    "topk": s.topk,
                    "num_experts": s.experts,
                },
            }
        )
        router_callable = make_kernel(
            topk_launcher,
            {
                "X": routing_logits,
                "Y": topk_weights,
                "Idx": topk_ids,
                "M": s.tokens,
                "N": s.experts,
            },
            router_grid,
            router_block,
        )
        # ---------------- Stage 2.5: gather (unpadded layout) ----------------
        # Chained directly with router + sort under the same
        # ``launch_kernel`` call (5 callables: router, hist, scan,
        # scatter, gather) -- amortizes the Python wrapping overhead
        # of separate ``launch_kernel`` calls. Same-stream FIFO
        # ordering still guarantees gather sees the sort's writes.
        gather_callable = make_kernel(
            self._fused_moe_launcher._ensure_launchers()["gather"],
            {
                "X": X,
                "SortedTokenIds": sorted_token_ids,
                "GroupedInput": grouped_input,
                "tokens": s.tokens,
                "hidden": s.hidden,
            },
            moe_gather_grid(s.to_fused_moe_spec()),
            (s.streaming_block_size, 1, 1),
        )
        launch_kernel(
            StreamConfig(stream_id=int(stream)),
            router_callable,
            *sort_callables,
            gather_callable,
        )

        # The per-expert GEMM dispatch needs ``Counts`` and ``Offsets``
        # on the host. Copy back the two ``(experts,)`` i32 buffers
        # and synchronize on the stream so the values are valid
        # before we read them.
        counts_cpu = counts.cpu()
        offsets_cpu = offsets.cpu()
        torch.cuda.synchronize()

        if self.spec.use_grouped_gemm:
            done = self._dispatch_grouped_gemm(
                counts_cpu=counts_cpu,
                offsets_cpu=offsets_cpu,
                grouped_input=grouped_input,
                sorted_token_ids=sorted_token_ids,
                sorted_weights=sorted_weights,
                y_f32=y_f32,
                W_gate=W_gate,
                W_up=W_up,
                W_down=W_down,
                Y=Y,
                device=device,
                stream=stream,
            )
            if done:
                return

        # ---------------- Uniform per-expert padded layout ----------------
        # Each expert's GEMM slot has a uniform ``MAX_PADDED_M`` rows
        # so the per-expert dispatch collapses to a single batched
        # GEMM with ``stride_X = MAX_PADDED_M * dim`` per batch.
        # ``MAX_PADDED_M`` = ``max_e ceil(count[e]/tile_m) * tile_m``,
        # rounded up to ``tile_m`` so the GEMM kernel's tile geometry
        # is a no-op.
        padded_counts_per_expert = [
            ((int(counts_cpu[e]) + tile_m - 1) // tile_m) * tile_m
            for e in range(s.experts)
        ]
        max_padded_m = max(padded_counts_per_expert)
        if max_padded_m == 0:
            # Degenerate case: every expert has count=0 (no tokens
            # routed). Output stays at zero, return early.
            Y.copy_(y_f32.to(Y.dtype))
            return
        if max_padded_m < tile_m:
            max_padded_m = tile_m
        total_padded = s.experts * max_padded_m

        from ...runtime.launcher import WorkspaceSpec as _WS  # local

        act_dtype = grouped_input.dtype
        grouped_input_padded = self._pool.get_spec(
            _WS(
                "GroupedInputPaddedUniform",
                (total_padded, s.hidden),
                act_dtype,
                device,
            )
        )
        gate_out_padded = self._pool.get_spec(
            _WS(
                "GateOutPaddedUniform",
                (total_padded, s.intermediate),
                act_dtype,
                device,
            )
        )
        up_out_padded = self._pool.get_spec(
            _WS(
                "UpOutPaddedUniform",
                (total_padded, s.intermediate),
                act_dtype,
                device,
            )
        )
        hidden_padded = self._pool.get_spec(
            _WS(
                "HiddenPaddedUniform",
                (total_padded, s.intermediate),
                act_dtype,
                device,
            )
        )
        down_out_padded = self._pool.get_spec(
            _WS(
                "DownOutPaddedUniform",
                (total_padded, s.hidden),
                act_dtype,
                device,
            )
        )
        sorted_token_ids_padded = self._pool.get_spec(
            _WS(
                "SortedTokenIdsPaddedUniform",
                (total_padded,),
                torch.int32,
                device,
            )
        )
        sorted_weights_padded = self._pool.get_spec(
            _WS(
                "SortedWeightsPaddedUniform",
                (total_padded,),
                torch.float32,
                device,
            )
        )

        grouped_input_padded.zero_()
        gate_out_padded.zero_()
        up_out_padded.zero_()
        hidden_padded.zero_()
        down_out_padded.zero_()
        sorted_token_ids_padded.fill_(-1)
        sorted_weights_padded.zero_()

        # Per-expert copy from unpadded gather output into the
        # uniform layout. Each expert e copies ``count[e]`` rows from
        # ``GroupedInput[unpadded_offsets[e]:...+count[e]]`` to
        # ``GroupedInputPaddedUniform[e*MAX:e*MAX+count[e]]`` (rest
        # stays zero from the ``zero_()`` above). Same logic
        # populates the padded ``SortedTokenIds`` and
        # ``SortedWeights`` -- the topk-reduce kernel skips entries
        # with ``SortedTokenIds == -1``.
        for e in range(s.experts):
            ce = int(counts_cpu[e])
            if ce == 0:
                continue
            u = int(offsets_cpu[e])
            p = e * max_padded_m
            grouped_input_padded[p : p + ce].copy_(grouped_input[u : u + ce])
            sorted_token_ids_padded[p : p + ce].copy_(sorted_token_ids[u : u + ce])
            sorted_weights_padded[p : p + ce].copy_(sorted_weights[u : u + ce])

        batched_gemm_launcher = self._ensure_batched_gemm_launcher()
        tile_n = self.spec.gemm_tile.tile_n
        gemm_block = (self.spec.to_batched_gemm_spec().block_size, 1, 1)

        # ---------------- Stages 3-6 chained in one launch_kernel ----------------
        # Build all 5 callables (gate / up GEMMs, silu_mul, down GEMM,
        # topk-reduce) and submit them in ONE
        # :func:`launch_kernel` call. Same-stream FIFO already gives
        # the data flow ordering; chaining them in one call amortizes
        # Python wrapping (one ``no_fence`` context, one args-pack
        # bucket walk) compared to 5 separate launch_kernel calls.
        gate_grid = (
            (s.intermediate + tile_n - 1) // tile_n,
            (max_padded_m + tile_m - 1) // tile_m,
            s.experts,
        )
        if s.active_tile_skip_gemms:
            gate_up_dyn_launcher = self._moe_batched_gemm_launcher(
                preshuffle_b=False,
                active_tile_skip=True,
            )
        else:
            gate_up_dyn_launcher = batched_gemm_launcher

        def _gate_up_args(B_tensor, C_tensor):
            args = {
                "A": grouped_input_padded,
                "B": B_tensor,
                "C": C_tensor,
                "M": max_padded_m,
                "N": s.intermediate,
                "K": s.hidden,
                "stride_a": max_padded_m * s.hidden,
                "stride_b": s.intermediate * s.hidden,
                "stride_c": max_padded_m * s.intermediate,
            }
            if s.active_tile_skip_gemms:
                args["SortedTokenIds"] = sorted_token_ids_padded
                args["slot_size"] = max_padded_m
            return args

        gate_callable = make_kernel(
            gate_up_dyn_launcher,
            _gate_up_args(W_gate, gate_out_padded),
            gate_grid,
            gemm_block,
        )
        up_callable = make_kernel(
            gate_up_dyn_launcher,
            _gate_up_args(W_up, up_out_padded),
            gate_grid,
            gemm_block,
        )
        silu_mul_callable = make_kernel(
            self._fused_moe_launcher._ensure_launchers()["silu_mul"],
            {
                "GateOut": gate_out_padded,
                "UpOut": up_out_padded,
                "Hidden": hidden_padded,
                "total_pairs": total_padded,
                "intermediate": s.intermediate,
            },
            (total_padded, 1, 1),
            (s.streaming_block_size, 1, 1),
        )
        down_grid = (
            (s.hidden + tile_n - 1) // tile_n,
            (max_padded_m + tile_m - 1) // tile_m,
            s.experts,
        )
        if s.active_tile_skip_gemms:
            down_b_launcher = self._moe_batched_gemm_launcher(
                preshuffle_b=s.preshuffle_w_down,
                active_tile_skip=True,
            )
            down_b_tensor = (
                self._ensure_w_down_preshuffled(W_down)
                if s.preshuffle_w_down
                else W_down
            )
        elif s.preshuffle_w_down:
            down_b_launcher = self._ensure_batched_gemm_preshuffle_b_launcher()
            down_b_tensor = self._ensure_w_down_preshuffled(W_down)
        else:
            down_b_launcher = batched_gemm_launcher
            down_b_tensor = W_down
        down_args = {
            "A": hidden_padded,
            "B": down_b_tensor,
            "C": down_out_padded,
            "M": max_padded_m,
            "N": s.hidden,
            "K": s.intermediate,
            "stride_a": max_padded_m * s.intermediate,
            "stride_b": s.hidden * s.intermediate,
            "stride_c": max_padded_m * s.hidden,
        }
        if s.active_tile_skip_gemms:
            down_args["SortedTokenIds"] = sorted_token_ids_padded
            down_args["slot_size"] = max_padded_m
        down_callable = make_kernel(
            down_b_launcher,
            down_args,
            down_grid,
            gemm_block,
        )
        reduce_callable = make_kernel(
            self._fused_moe_launcher._ensure_launchers()["topk_reduce"],
            {
                "DownOut": down_out_padded,
                "SortedTokenIds": sorted_token_ids_padded,
                "SortedWeights": sorted_weights_padded,
                "Y": y_f32,
                "total_pairs": total_padded,
                "hidden": s.hidden,
                "tokens": s.tokens,
            },
            (total_padded, 1, 1),
            (s.streaming_block_size, 1, 1),
        )
        launch_kernel(
            StreamConfig(stream_id=int(stream)),
            gate_callable,
            up_callable,
            silu_mul_callable,
            down_callable,
            reduce_callable,
        )

        # ---------------- Stage 7: dtype cast ----------------
        # The reduce kernel atomic-adds into f32; cast to the user's
        # output dtype. Inside the timed region so the benchmark
        # number reflects what an end-to-end caller observes.
        Y.copy_(y_f32.to(Y.dtype))

    # ------------------------------------------------------------------
    # HIP graph capture (opt-in, benchmark / fixed-input use)
    # ------------------------------------------------------------------

    def capture_graph(
        self,
        *,
        routing_logits: Any,
        X: Any,
        W_gate: Any,
        W_up: Any,
        W_down: Any,
        Y: Any,
        warmup_iters: int = 2,
    ) -> None:
        """Capture a HIP graph of the static-mode forward.

        Only valid in static-offset mode (``self._use_static_offsets``
        is True; raises otherwise). Runs ``warmup_iters`` forward
        calls on a side stream to populate the
        :class:`~rocke.runtime.launcher.WorkspacePool` and let the
        torch caching allocator settle, then captures one forward
        into a :class:`torch.cuda.CUDAGraph`. Subsequent
        :meth:`replay_graph` calls replay this graph in ~one
        ``hipGraphLaunch`` instead of N separate
        ``hipModuleLaunchKernel`` dispatches.

        Caller contract: the input tensor pointers (``routing_logits``,
        ``X``, ``W_gate``, ``W_up``, ``W_down``, ``Y``) must be the
        same on every :meth:`replay_graph` call. Callers that need
        to swap inputs should either re-capture (cheap; ~5 ms
        warmup) or copy fresh data into the captured tensors before
        replay.

        This is an inference-time optimization; production paths that
        legitimately get a fresh tensor each call should stay on
        :meth:`forward`.
        """
        if not self._use_static_offsets:
            raise RuntimeError(
                "capture_graph requires static-offset mode "
                f"(T*K*E={self.spec.tokens * self.spec.topk * self.spec.experts} > 256)"
            )
        import torch

        self._ensure_compiled()
        # Eagerly build the packed gate/up weights for paths that need
        # a cached packed B tensor so the packing doesn't happen inside
        # the captured region.
        if self.spec.use_experimental_fused_gate_up_silu:
            pass
        elif self.spec.use_experimental_interleaved_gate_up_silu:
            if self.spec.preshuffle_w_gate_up_interleaved:
                self._ensure_gu_interleaved_preshuffled(W_gate, W_up)
            else:
                self._ensure_gu_interleaved(W_gate, W_up)
        else:
            if self.spec.preshuffle_w_gate_up_packed:
                self._ensure_gu_concat_preshuffled(W_gate, W_up)
            else:
                self._ensure_gu_concat(W_gate, W_up)
        if self.spec.preshuffle_w_down:
            self._ensure_w_down_preshuffled(W_down)

        # Warmup on a side stream. The first warmup populates the
        # pool (allocations); subsequent warmups run on the same
        # pool (no allocations) so the captured stream is
        # allocation-free.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(max(1, warmup_iters)):
                self._forward_static(
                    routing_logits=routing_logits,
                    X=X,
                    W_gate=W_gate,
                    W_up=W_up,
                    W_down=W_down,
                    Y=Y,
                    stream=int(capture_stream.cuda_stream),
                )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self._forward_static(
                routing_logits=routing_logits,
                X=X,
                W_gate=W_gate,
                W_up=W_up,
                W_down=W_down,
                Y=Y,
                stream=int(capture_stream.cuda_stream),
            )
        self._captured_graph = graph
        self._captured_graph_stream = capture_stream

    def replay_graph(self) -> None:
        """Replay the graph captured by :meth:`capture_graph`.

        Raises if :meth:`capture_graph` has not been called yet.
        """
        if self._captured_graph is None:
            raise RuntimeError(
                "replay_graph called before capture_graph; call "
                "capture_graph(...) once with the inputs you intend to "
                "replay against, then replay_graph() in the timed loop"
            )
        self._captured_graph.replay()

    # ------------------------------------------------------------------
    # Static-offset fast path (decode-shape oriented)
    # ------------------------------------------------------------------

    def _forward_static(
        self,
        *,
        routing_logits: Any,
        X: Any,
        W_gate: Any,
        W_up: Any,
        W_down: Any,
        Y: Any,
        stream: int = 0,
    ) -> None:
        """End-to-end fused-MoE forward with static per-expert offsets.

        Eliminates the dynamic histogram + scan kernels and the host
        roundtrip on ``Counts`` / ``Offsets`` by pre-computing fixed
        offsets ``[0, S, 2S, ..., (E-1)*S]`` for ``S =
        ceil(T*K / tile_m) * tile_m`` (the worst-case per-expert
        count, all tokens to one expert). Each expert's GEMM slot is
        always ``S`` rows; padded rows have ``SortedTokenIds == -1``
        sentinels and zero-init weights, so they contribute nothing.

        The entire forward runs as a single
        :func:`~rocke.runtime.launcher.launch_kernel` chain (no
        host roundtrip, no per-expert Python loop). Used when
        ``T * K * E <= 256`` (decode + small-batch shapes); the
        full dynamic-histogram path stays the default for chunked
        prefill where ``T*K`` is large enough that the static
        over-padding becomes meaningful.
        """
        import torch

        self._ensure_compiled()
        s = self.spec
        slot_size = self._static_slot_size
        total_padded = s.experts * slot_size
        device = X.device

        # ---- Workspace ----
        ws = self._pool.prepare(self._workspace_specs(device))
        topk_ids = ws["TopkIds"]
        topk_weights = ws["TopkWeights"]
        counter = ws["Counter"]
        y_f32 = ws["Y_f32"]

        from ...runtime.launcher import WorkspaceSpec as _WS  # local

        act_dtype = _torch_dtype_for(s.dtype)
        sorted_token_ids_padded = self._pool.get_spec(
            _WS(
                "StaticSortedTokenIdsPadded",
                (total_padded,),
                torch.int32,
                device,
            )
        )
        sorted_topk_ids_padded = self._pool.get_spec(
            _WS(
                "StaticSortedTopkIdsPadded",
                (total_padded,),
                torch.int32,
                device,
            )
        )
        sorted_weights_padded = self._pool.get_spec(
            _WS(
                "StaticSortedWeightsPadded",
                (total_padded,),
                torch.float32,
                device,
            )
        )
        grouped_input_padded = self._pool.get_spec(
            _WS(
                "StaticGroupedInputPadded",
                (total_padded, s.hidden),
                act_dtype,
                device,
            )
        )
        gate_up_packed = self._pool.get_spec(
            _WS(
                "StaticGateUpPacked",
                (total_padded, 2 * s.intermediate),
                act_dtype,
                device,
            )
        )
        hidden_padded = self._pool.get_spec(
            _WS(
                "StaticHiddenPadded",
                (total_padded, s.intermediate),
                act_dtype,
                device,
            )
        )
        down_out_padded = self._pool.get_spec(
            _WS(
                "StaticDownOutPadded",
                (total_padded, s.hidden),
                act_dtype,
                device,
            )
        )

        # ---- One-shot init ----
        if self._static_offsets is None:
            self._static_offsets = (
                torch.arange(s.experts, dtype=torch.int32, device=device) * slot_size
            )

        # Reset per-call state. These are torch ops on the current
        # stream; they're cheap (~5us each) and needed because
        # ``Counter`` is the per-expert atomic-counter target for
        # scatter, ``SortedTokenIds`` defaults to -1 for "skip"
        # (only valid scatter writes overwrite that), ``Y_f32`` is
        # the topk-reduce f32 accumulator. The other padded buffers
        # do NOT need pre-zero (each kernel writes them in full).
        counter.zero_()
        sorted_token_ids_padded.fill_(-1)
        sorted_weights_padded.zero_()
        # The fused static scatter+gather kernel writes only real
        # routed rows. Padded rows must be zero because the subsequent
        # batched GEMM reads the full static slot_size per expert.
        grouped_input_padded.zero_()
        y_f32.zero_()

        topk_launcher = self._ensure_topk_launcher()
        batched_gemm_launcher = self._ensure_batched_gemm_launcher()
        use_experimental_fused = bool(s.use_experimental_fused_gate_up_silu)
        use_experimental_interleaved = (
            bool(s.use_experimental_interleaved_gate_up_silu)
            and not use_experimental_fused
        )
        use_experimental_down_reduce = bool(s.use_experimental_fused_down_reduce)
        use_experimental_static_sg = bool(s.use_experimental_static_scatter_gather)
        gate_up_silu_launcher = (
            self._ensure_gate_up_silu_launcher() if use_experimental_fused else None
        )
        if use_experimental_interleaved:
            if s.active_tile_skip_gemms:
                interleaved_gate_up_silu_launcher = (
                    self._moe_interleaved_gate_up_silu_launcher(
                        preshuffle_b=s.preshuffle_w_gate_up_interleaved,
                        active_tile_skip=True,
                    )
                )
            elif s.preshuffle_w_gate_up_interleaved:
                interleaved_gate_up_silu_launcher = (
                    self._ensure_interleaved_gate_up_silu_preshuffle_launcher()
                )
            else:
                interleaved_gate_up_silu_launcher = (
                    self._ensure_interleaved_gate_up_silu_launcher()
                )
        else:
            interleaved_gate_up_silu_launcher = None
        down_reduce_launcher = (
            self._ensure_down_reduce_launcher()
            if use_experimental_down_reduce
            else None
        )
        silu_mul_packed_launcher = (
            None if use_experimental_fused else self._ensure_silu_mul_packed_launcher()
        )
        static_scatter_gather_launcher = self._ensure_static_scatter_gather_launcher()
        sort_launchers = (
            None
            if use_experimental_static_sg
            else self._sort_launcher._ensure_launchers()
        )
        fmoe_launchers = self._fused_moe_launcher._ensure_launchers()
        gu_concat = (
            None
            if (use_experimental_fused or use_experimental_interleaved)
            else self._ensure_gu_concat(W_gate, W_up)
        )
        if use_experimental_interleaved:
            if s.preshuffle_w_gate_up_interleaved:
                gu_interleaved = self._ensure_gu_interleaved_preshuffled(W_gate, W_up)
            else:
                gu_interleaved = self._ensure_gu_interleaved(W_gate, W_up)
        else:
            gu_interleaved = None

        # ---- Build callables ----
        router_grid = topk_softmax_grid(s.tokens, s.to_topk_softmax_spec())
        router_block = (s.router_block_size, 1, 1)
        router_callable = make_kernel(
            topk_launcher,
            {
                "X": routing_logits,
                "Y": topk_weights,
                "Idx": topk_ids,
                "M": s.tokens,
                "N": s.experts,
            },
            router_grid,
            router_block,
        )

        fmoe_spec = s.to_fused_moe_spec()
        if use_experimental_static_sg:
            # Static scatter+gather fusion: reads topk_ids / weights,
            # claims static slots via Counter, writes SortedTokenIds /
            # SortedWeights, and directly copies X[t, :] into
            # GroupedInputPadded[slot, :]. Correct but currently
            # slower than the two-kernel default.
            route_stage_callables = [
                make_kernel(
                    static_scatter_gather_launcher,
                    {
                        "TopkIds": topk_ids,
                        "TopkWeights": topk_weights,
                        "Counter": counter,
                        "X": X,
                        "SortedTokenIds": sorted_token_ids_padded,
                        "SortedWeights": sorted_weights_padded,
                        "GroupedInput": grouped_input_padded,
                        "tokens": s.tokens,
                        "topk": s.topk,
                        "num_experts": s.experts,
                        "hidden": s.hidden,
                        "slot_size": slot_size,
                    },
                    moe_static_scatter_gather_grid(fmoe_spec),
                    (s.streaming_block_size, 1, 1),
                )
            ]
        else:
            sort_spec = s.to_sort_spec()
            scatter_callable = make_kernel(
                sort_launchers["scatter"],
                {
                    "TopkIds": topk_ids,
                    "TopkWeights": topk_weights,
                    "Offsets": self._static_offsets,
                    "Counter": counter,
                    "SortedTokenIds": sorted_token_ids_padded,
                    "SortedTopkIds": sorted_topk_ids_padded,
                    "SortedWeights": sorted_weights_padded,
                    "tokens": s.tokens,
                    "topk": s.topk,
                    "num_experts": s.experts,
                },
                (sort_spec.block_size, 1, 1),
                (sort_spec.block_size, 1, 1),
            )
            gather_callable = make_kernel(
                fmoe_launchers["gather"],
                {
                    "X": X,
                    "SortedTokenIds": sorted_token_ids_padded,
                    "GroupedInput": grouped_input_padded,
                    "tokens": s.tokens,
                    "hidden": s.hidden,
                },
                (total_padded, 1, 1),
                (s.streaming_block_size, 1, 1),
            )
            route_stage_callables = [scatter_callable, gather_callable]

        tile_m = self.spec.gemm_tile.tile_m
        tile_n = self.spec.gemm_tile.tile_n
        gemm_block = (self.spec.to_batched_gemm_spec().block_size, 1, 1)
        if use_experimental_fused:
            gate_up_silu_spec = FusedGateUpSiluGemmSpec(
                name=f"{self.spec.name}_gate_up_silu",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
                dtype=_gemm_dtype_to_universal(s.dtype),
            )
            # True CK Tile-style gate+up fusion: one MFMA kernel keeps
            # gate and up accumulators in registers and writes
            # Hidden = silu(gate) * up directly. This removes the
            # GateUpPacked HBM round-trip and the separate silu_mul
            # launch. It is experimental because the current dual-B
            # kernel is correctness-clean but still slower than the
            # graph-captured packed path on MI355X.
            gate_stage_callables = [
                make_kernel(
                    gate_up_silu_launcher,
                    {
                        "A": grouped_input_padded,
                        "WGate": W_gate,
                        "WUp": W_up,
                        "Hidden": hidden_padded,
                        "M": slot_size,
                        "N": s.intermediate,
                        "K": s.hidden,
                        "stride_a": slot_size * s.hidden,
                        "stride_b": s.intermediate * s.hidden,
                        "stride_c": slot_size * s.intermediate,
                    },
                    moe_gate_up_silu_gemm_grid(
                        s.experts, slot_size, s.intermediate, gate_up_silu_spec
                    ),
                    (gate_up_silu_spec.block_size, 1, 1),
                )
            ]
        elif use_experimental_interleaved:
            inter_spec = FusedInterleavedGateUpSiluGemmSpec(
                name=f"{self.spec.name}_interleaved_gate_up_silu",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(
                    pad_m=True,
                    pad_n=True,
                    epilogue="default",
                    active_tile_skip=s.active_tile_skip_gemms,
                ),
                dtype=_gemm_dtype_to_universal(s.dtype),
            )
            gate_up_args = {
                "A": grouped_input_padded,
                "WGateUp": gu_interleaved,
                "Hidden": hidden_padded,
                "M": slot_size,
                "N": s.intermediate,
                "K": s.hidden,
                "stride_a": slot_size * s.hidden,
                "stride_b": (2 * s.intermediate) * s.hidden,
                "stride_c": slot_size * s.intermediate,
            }
            if s.active_tile_skip_gemms:
                gate_up_args["SortedTokenIds"] = sorted_token_ids_padded
                gate_up_args["slot_size"] = slot_size
            gate_stage_callables = [
                make_kernel(
                    interleaved_gate_up_silu_launcher,
                    gate_up_args,
                    moe_interleaved_gate_up_silu_gemm_grid(
                        s.experts, slot_size, s.intermediate, inter_spec
                    ),
                    (inter_spec.block_size, 1, 1),
                )
            ]
        else:
            # Fast default path: one batched GEMM with N=2*I writes a
            # G1U1-packed GateUp buffer, then a vectorized streaming
            # kernel computes Hidden = silu(gate) * up. This still uses
            # an HBM intermediate, but current measurements show it is
            # faster than the first experimental dual-B fused kernel.
            gate_up_n = 2 * s.intermediate
            gate_up_grid = (
                (gate_up_n + tile_n - 1) // tile_n,
                (slot_size + tile_m - 1) // tile_m,
                s.experts,
            )
            if s.active_tile_skip_gemms:
                gate_up_b_launcher = self._moe_batched_gemm_launcher(
                    preshuffle_b=s.preshuffle_w_gate_up_packed,
                    active_tile_skip=True,
                )
                gate_up_b_tensor = (
                    self._ensure_gu_concat_preshuffled(W_gate, W_up)
                    if s.preshuffle_w_gate_up_packed
                    else gu_concat
                )
            elif s.preshuffle_w_gate_up_packed:
                gate_up_b_launcher = self._ensure_batched_gemm_preshuffle_b_launcher()
                gate_up_b_tensor = self._ensure_gu_concat_preshuffled(W_gate, W_up)
            else:
                gate_up_b_launcher = batched_gemm_launcher
                gate_up_b_tensor = gu_concat
            gate_up_args = {
                "A": grouped_input_padded,
                "B": gate_up_b_tensor,
                "C": gate_up_packed,
                "M": slot_size,
                "N": gate_up_n,
                "K": s.hidden,
                "stride_a": slot_size * s.hidden,
                "stride_b": gate_up_n * s.hidden,
                "stride_c": slot_size * gate_up_n,
            }
            if s.active_tile_skip_gemms:
                gate_up_args["SortedTokenIds"] = sorted_token_ids_padded
                gate_up_args["slot_size"] = slot_size
            gate_up_callable = make_kernel(
                gate_up_b_launcher,
                gate_up_args,
                gate_up_grid,
                gemm_block,
            )
            silu_mul_callable = make_kernel(
                silu_mul_packed_launcher,
                {
                    "GateUp": gate_up_packed,
                    "Hidden": hidden_padded,
                    "total_pairs": total_padded,
                    "intermediate": s.intermediate,
                },
                (total_padded, 1, 1),
                (s.streaming_block_size, 1, 1),
            )
            gate_stage_callables = [gate_up_callable, silu_mul_callable]
        down_grid = (
            (s.hidden + tile_n - 1) // tile_n,
            (slot_size + tile_m - 1) // tile_m,
            s.experts,
        )
        if use_experimental_down_reduce:
            down_reduce_spec = FusedDownReduceGemmSpec(
                name=f"{self.spec.name}_down_reduce",
                tile=self.spec.gemm_tile,
                trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
                dtype=_gemm_dtype_to_universal(s.dtype),
            )
            down_stage_callables = [
                make_kernel(
                    down_reduce_launcher,
                    {
                        "A": hidden_padded,
                        "WDown": W_down,
                        "SortedTokenIds": sorted_token_ids_padded,
                        "SortedWeights": sorted_weights_padded,
                        "Y": y_f32,
                        "M": slot_size,
                        "N": s.hidden,
                        "K": s.intermediate,
                        "stride_a": slot_size * s.intermediate,
                        "stride_b": s.hidden * s.intermediate,
                        "slot_size": slot_size,
                        "tokens": s.tokens,
                    },
                    moe_down_reduce_gemm_grid(
                        s.experts, slot_size, s.hidden, down_reduce_spec
                    ),
                    (down_reduce_spec.block_size, 1, 1),
                )
            ]
        else:
            if s.active_tile_skip_gemms:
                down_b_launcher = self._moe_batched_gemm_launcher(
                    preshuffle_b=s.preshuffle_w_down,
                    active_tile_skip=True,
                )
                down_b_tensor = (
                    self._ensure_w_down_preshuffled(W_down)
                    if s.preshuffle_w_down
                    else W_down
                )
            elif s.preshuffle_w_down:
                down_b_launcher = self._ensure_batched_gemm_preshuffle_b_launcher()
                down_b_tensor = self._ensure_w_down_preshuffled(W_down)
            else:
                down_b_launcher = batched_gemm_launcher
                down_b_tensor = W_down
            down_args = {
                "A": hidden_padded,
                "B": down_b_tensor,
                "C": down_out_padded,
                "M": slot_size,
                "N": s.hidden,
                "K": s.intermediate,
                "stride_a": slot_size * s.intermediate,
                "stride_b": s.hidden * s.intermediate,
                "stride_c": slot_size * s.hidden,
            }
            if s.active_tile_skip_gemms:
                down_args["SortedTokenIds"] = sorted_token_ids_padded
                down_args["slot_size"] = slot_size
            down_callable = make_kernel(
                down_b_launcher,
                down_args,
                down_grid,
                gemm_block,
            )
            reduce_callable = make_kernel(
                fmoe_launchers["topk_reduce"],
                {
                    "DownOut": down_out_padded,
                    "SortedTokenIds": sorted_token_ids_padded,
                    "SortedWeights": sorted_weights_padded,
                    "Y": y_f32,
                    "total_pairs": total_padded,
                    "hidden": s.hidden,
                    "tokens": s.tokens,
                },
                (total_padded, 1, 1),
                (s.streaming_block_size, 1, 1),
            )
            down_stage_callables = [down_callable, reduce_callable]

        # ---- Single launch_kernel for the whole forward ----
        # The gate/up stage contributes either one experimental fused
        # MFMA callable (gate+up+silu) or two fast-default callables
        # (packed gate+up GEMM + packed silu_mul). No host roundtrip. The
        # streaming path's same-stream FIFO + the static offsets
        # carry the data flow:
        #   router writes TopkIds / TopkWeights
        #   scatter reads them, writes SortedTokenIds (+others)
        #   gather reads SortedTokenIds, writes GroupedInputPadded
        #   gate/up stage reads GroupedInputPadded and writes Hidden
        #   down GEMM reads Hidden, writes DownOut
        #   topk_reduce reads DownOut + SortedTokenIds + Weights,
        #     atomic-adds into Y_f32
        launch_kernel(
            StreamConfig(stream_id=int(stream)),
            router_callable,
            *route_stage_callables,
            *gate_stage_callables,
            *down_stage_callables,
        )
        Y.copy_(y_f32.to(Y.dtype))
