# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The identity of an AOT convolution kernel and the shapes it can serve.

* :class:`KernelIdentity` -- the compile-time tuple that uniquely identifies
  one HSACO. Only configuration and capability fields belong here; no problem
  extents. See the class docstring for why the distinction matters.
* :func:`identity_supports_problem` -- whether a binary with a given identity
  can be launched on a concrete problem.

Both live with the kernels rather than with the benchmark tooling that first
needed them, because two layers consume them: the AOT cache
(``benchmarks.common.kernel_cache``) files binaries under their identity and
offers them to shapes, and the grouped-convolution dispatcher
(``dispatch.grouped_convolution``) registers its instances by identity and
admits a request with the same predicate -- one rule, so the dispatcher can
never select an instance the cache would refuse to launch, or the reverse.
Names are spelled by :mod:`kernels.conv.kernel_label`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

__all__ = [
    "DIRECT_PROBLEM_CAPABILITY_ATTRS",
    "IDENTITY_CAPABILITY_FIELDS",
    "KernelIdentity",
    "current_llvm_flavor",
    "dgrad_sub_gemms",
    "identity_capability_key",
    "identity_supports_problem",
    "problem_capability_key",
]


# The per-axis workgroup-count cap the validators enforce (gridDim y/z on
# AMD; x is capped the same way for symmetry with is_valid_spec_for_problem).
_MAX_GRID_DIM = 65535


def current_llvm_flavor() -> str:
    """LLVM IR flavor this process lowers to (the one ``compile_kernel`` uses).

    The platform's resolver owns the choice -- the ``ROCKE_LLVM_FLAVOR``
    override and COMGR detection -- and caches the loaded compiler's evidence,
    so this neither reads the environment itself nor memoises. Host tooling
    (the AOT cache, ``--compile-all``) asks it; dispatch never does: a
    request carries its own ``llvm_flavor``.
    """
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


# ---------------------------------------------------------------------------
# KernelIdentity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelIdentity:
    """Compile-time tuple that uniquely identifies an AOT kernel.

    No problem *extents* here — that is the whole point of AOT. Capability
    fields that genuinely constrain which shapes the binary accepts do belong,
    and are checked against the problem by :func:`identity_supports_problem`.

    Two kinds of field live here, and the distinction matters:

    * **Configuration** — tile sizes, pipeline, epilogue, vector widths. These
      change the ISA but not what the kernel can be *launched on*.
    * **Capability** — ``is_3d``, ``filter_h/w``, ``stride``, ``pad``,
      ``cpg``/``kpg`` for direct conv, ``max_sub_gemms`` for dgrad. These bound
      the set of problems the binary can serve, and
      :func:`identity_supports_problem` checks them before a kernel is offered
      for a shape.

    Every field that changes the emitted ISA must be here. A field that is
    missing does not merely weaken the cache key — two different kernels hash to
    the same name and silently overwrite each other on disk.
    """

    arch: str
    direction: str
    algorithm: str
    dtype_a: str
    dtype_b: str
    dtype_d: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    pipeline: str
    epilogue: str
    wave_size: int
    vector_size_a: int
    vector_size_b: int
    vector_size_c: int
    # ---- configuration that changes the emitted ISA ----
    async_dma: bool = False
    unroll_k: bool = False
    chiplet_swizzle: bool = False
    lds_layout: str = "default"
    lds_k_pad: int = 0
    lds_k_outer: bool = False
    waves_per_eu: int | None = None
    acc_epilogue: str = "none"
    split_k: int = 1
    two_stage: bool = False
    # Two-stage scratch slabs per group (WgradConvSpec.ws_replicas). The host
    # sizes the scratch and builds Stage 2 from it, so it has to come from the
    # binary's identity rather than a default the two could drift apart on.
    # 0 = no scratch (not two-stage).
    ws_replicas: int = 0
    # wgrad stream-K (WgradConvSpec.streamk / streamk_reduction): the launch
    # mode ("dp_sk" or "persistent") and the partial-sum reduction are baked
    # into the ISA; the CTA pool is a kernarg, so it is not here. "off" / ""
    # for every other kernel.
    streamk: str = "off"
    streamk_reduction: str = ""
    group_merge: int = 1
    # wgrad only (WgradConvSpec.prefetch_addr): carried K-loop byte offsets.
    # Left out of the hash while False, like ws_replicas, so identities that
    # predate the field keep their hash and their cache entries.
    prefetch_addr: bool = False
    # The register-prefetch K loop. wgrad: WgradConvSpec.prefetch_regs
    # (1: one LDS tile, 2: two, 3: two tiles ahead); fwd: the bool
    # ImplicitGemmConvSpec.prefetch_regs, recorded as 1. Left out of the hash
    # at 0, like prefetch_addr.
    prefetch_regs: int = 0
    # wgrad only (WgradConvSpec.lds_kpack): K steps packed per LDS store on
    # the M-outer transpose-on-store. Left out of the hash at 1.
    lds_kpack: int = 1
    # wgrad (WgradConvSpec.xcd_remap: whole-grid remap), fwd
    # (ImplicitGemmConvSpec.xcd_remap: XCD-contiguous tile order) and dgrad
    # (DgradConvSpec.xcd_remap: the XCD tile order for ungrouped launches
    # too): the XCD count, 0 = launch order / the target default. Left out
    # of the hash at 0.
    xcd_remap: int = 0
    # wgrad only (WgradConvSpec.sep_addr): separable (1) / carried (2)
    # operand addressing. Left out of the hash at 0.
    sep_addr: int = 0
    # wgrad only (WgradConvSpec.pfr_sched): issue schedule of the prefetch
    # loop's reads (1 early, N>1 interleave). Left out of the hash at 0.
    pfr_sched: int = 0
    # wgrad only (WgradConvSpec.lds_bswz): pad-free XOR-swizzled B tile.
    # Left out of the hash at 0.
    lds_bswz: int = 0
    # wgrad only (WgradConvSpec.kp_line): line-major decode of the k-packed
    # reads. Left out of the hash at 0.
    kp_line: int = 0
    # wgrad only (WgradConvSpec.xwin): X-window reuse over the x taps; the
    # value is the baked filter width. Left out of the hash at 0.
    xwin: int = 0
    # wgrad only (WgradConvSpec.xwin_rows): filter rows per xwin B tile.
    # Left out of the hash at 1.
    xwin_rows: int = 1
    # wgrad only (WgradConvSpec.xwin_groups / xwin_pp): xwin sub-tiles per
    # workgroup (stacked along split-K z) and the block ping-pong K loop.
    # Left out of the hash at 1 / 0.
    xwin_groups: int = 1
    xwin_pp: int = 0
    # wgrad only (WgradConvSpec.xwin_mrow): output rows per xwin K tile.
    # Left out of the hash at 1.
    xwin_mrow: int = 1
    # wgrad only (WgradConvSpec.xwin_gn): groups over consecutive flat tiles.
    # Left out of the hash at 0.
    xwin_gn: int = 0
    # dgrad (DgradConvSpec.uniform_k_decode) and fwd
    # (ImplicitGemmConvSpec.uniform_k_decode): the per-tile reduction decode
    # K loop. Left out of the hash while False, like prefetch_addr.
    uniform_k_decode: bool = False
    # fwd only (ImplicitGemmConvSpec.reg_prefetch): the register-prefetch
    # K loop, 1 = one LDS buffer, 2 = two. Left out of the hash at 0.
    reg_prefetch: int = 0
    # fwd (ImplicitGemmConvSpec.lds_xor) and dgrad x-halo
    # (DgradConvSpec.lds_xor): XOR-swizzled, unpadded A/B LDS tiles. Left out
    # of the hash while False.
    lds_xor: bool = False
    # fwd only (ImplicitGemmConvSpec.a_direct). Left out of the hash while False.
    a_direct: bool = False
    # fwd only (ImplicitGemmConvSpec.uk_tap_fast). Left out of the hash while False.
    uk_tap_fast: bool = False
    # dgrad only (DgradConvSpec.patch): a patch embedding as one GEMM over the
    # filter's (y, x, c). Changes the store and the problems served (see
    # identity_supports_problem). Left out of the hash while False.
    patch: bool = False
    # dgrad only (DgradConvSpec.tap_outer_k): the stride-1 tap-outer K loop.
    # Serves only problems whose kpg is a multiple of tile_k (see
    # identity_supports_problem). Left out of the hash while False.
    tap_outer_k: bool = False
    # dgrad only (DgradConvSpec.tapk_prefetch): the software-pipelined form of
    # the tap-outer K loop (1: register prefetch, 2: LDS ping-pong). Serves
    # the same problems as its tap_outer_k binary. Left out of the hash at 0.
    tapk_prefetch: int = 0
    # dgrad only (DgradConvSpec.tapk_xhalo): dY reuse across the x taps of the
    # tap-outer K loop. Bakes the filter width (filter_w) and serves pW == 0
    # only (see identity_supports_problem). Left out of the hash while False.
    tapk_xhalo: bool = False
    # dgrad only (DgradConvSpec.b_kpack): k values per B LDS write. Left out
    # of the hash at 0.
    b_kpack: int = 0
    # dgrad only (DgradConvSpec.tapk_ilv / xhalo_exact / cshuffle_m_chunks):
    # x-halo loop load interleave, exact halo rows, and the C staging slabs
    # of the cshuffle epilogue. Left out of the hash at 0 / False / 1.
    tapk_ilv: int = 0
    xhalo_exact: bool = False
    cshuffle_m_chunks: int = 1
    # dgrad only (DgradConvSpec.tapk_lds_ahead): LDS fragment reads that many
    # MFMA steps ahead in the x-halo loop. Left out of the hash at 0.
    tapk_lds_ahead: int = 0
    # dgrad only (DgradConvSpec.tapk_lds_barrier). Left out of the hash at False.
    tapk_lds_barrier: bool = False
    # dgrad only (DgradConvSpec.tapk_setprio). Left out of the hash at 0.
    tapk_setprio: int = 0
    # dgrad only (DgradConvSpec.xh_b_wide). Left out of the hash at False.
    xh_b_wide: bool = False
    # dgrad only (DgradConvSpec.xh_b_kpad). Left out of the hash at 0.
    xh_b_kpad: int = 0
    # dgrad only (DgradConvSpec.tapk_pingpong). Left out of the hash at 0.
    tapk_pingpong: int = 0
    # dgrad only (DgradConvSpec.tapk_pp_sched). Left out of the hash at 0.
    tapk_pp_sched: int = 0
    # dgrad only (DgradConvSpec.tapk_pp_epi). Left out of the hash at 0.
    tapk_pp_epi: int = 0
    # dgrad only (DgradConvSpec.xh_a_k4 / xh_order / xh_defer). Left out of
    # the hash at False / 0 / 0.
    xh_a_k4: bool = False
    xh_order: int = 0
    xh_defer: int = 0
    # fwd only (ImplicitGemmConvSpec.split_addr_k_decode): the per-tile
    # reduction decode of the 3-D split-addressing path. Serves only problems
    # whose cpg is a multiple of tile_k (see identity_supports_problem). Left
    # out of the hash while False.
    split_addr_k_decode: bool = False
    # fwd split-K binaries only (ImplicitGemmConvSpec.split_tail): split only
    # the tiles past the ``full_tiles`` kernarg. Left out of the hash at False.
    split_tail: bool = False
    # fwd only (ImplicitGemmConvSpec.uk_nopad): binaries without the per-tap
    # bounds checks serve unpadded problems only. Left out of the hash at False.
    uk_nopad: bool = False
    # fwd only (ImplicitGemmConvSpec.defer_mfma_per_write). Out of the hash at 0.
    defer_mfma_per_write: int = 0
    # fwd only (ImplicitGemmConvSpec.lds_ahead / defer_mfma /
    # vmem_interleave): K-loop instruction scheduling of the reg_prefetch /
    # prefetch_regs loops. Left out of the hash at False / 0.
    lds_ahead: bool = False
    defer_mfma: int = 0
    vmem_interleave: int = 0
    # fwd only (ImplicitGemmConvSpec.lds_kpack_read). Left out of the hash
    # while False.
    lds_kpack_read: bool = False
    # fwd only (ImplicitGemmConvSpec.uk_chunk_group). Left out of the hash at 0.
    uk_chunk_group: int = 0
    # fwd only (ImplicitGemmConvSpec.rpf_lds_barrier). Left out of the hash while False.
    rpf_lds_barrier: bool = False
    # fwd only (ImplicitGemmConvSpec.rpf_fetch_late). Left out of the hash while False.
    rpf_fetch_late: bool = False
    # fwd only (ImplicitGemmConvSpec.uk_xhalo). Left out of the hash at 0.
    uk_xhalo: int = 0
    # fwd only (ImplicitGemmConvSpec.xh_dpp). Left out of the hash while False.
    xh_dpp: bool = False
    # fwd only (ImplicitGemmConvSpec.rpf_setprio). Left out of the hash at 0.
    rpf_setprio: int = 0
    # fwd only (ImplicitGemmConvSpec.rpf_pingpong). Left out of the hash at 0.
    rpf_pingpong: int = 0
    # fwd only (ImplicitGemmConvSpec.pp_pre_mfma). Left out of the hash at 0.
    pp_pre_mfma: int = 0
    # fwd only (ImplicitGemmConvSpec.pp_b_double). Left out of the hash while False.
    pp_b_double: bool = False
    # fwd only (ImplicitGemmConvSpec.lds_xor_rowkey). Left out of the hash while False.
    lds_xor_rowkey: bool = False
    # fwd only (ImplicitGemmConvSpec.xh_wait_smem). Left out of the hash while False.
    xh_wait_smem: bool = False
    # fwd only (ImplicitGemmConvSpec.split_atomic_agent). Left out of the hash while False.
    split_atomic_agent: bool = False
    # fwd (ImplicitGemmConvSpec) and dgrad (DgradConvSpec) split_atomic_pairs.
    # Left out of the hash while False.
    split_atomic_pairs: bool = False
    # fwd only (ImplicitGemmConvSpec.split_xcd_remap). Left out of the hash at 0.
    split_xcd_remap: int = 0
    # fwd only (ImplicitGemmConvSpec.split_atomic_fast). Left out of the hash while False.
    split_atomic_fast: bool = False
    # fwd only (ImplicitGemmConvSpec.epi_store_fast). Left out of the hash while False.
    epi_store_fast: bool = False
    # fwd only (ImplicitGemmConvSpec.split_zero_first). Left out of the hash while False.
    split_zero_first: bool = False
    # fwd only (ImplicitGemmConvSpec.split_streamk). Left out of the hash while False.
    split_streamk: bool = False
    # fwd only (ImplicitGemmConvSpec.split_flag_store). Left out of the hash while False.
    split_flag_store: bool = False
    # fwd only (ImplicitGemmConvSpec.xh_subtile). Left out of the hash while False.
    xh_subtile: bool = False
    # fwd only (ImplicitGemmConvSpec.pp_fetch_deep). Left out of the hash while False.
    pp_fetch_deep: bool = False
    # fwd only (ImplicitGemmConvSpec.pp_split_fetch). Left out of the hash at 0.
    pp_split_fetch: int = 0
    # fwd only (ImplicitGemmConvSpec.fb_skip_empty). Left out of the hash while False.
    fb_skip_empty: bool = False
    # fwd only (ImplicitGemmConvSpec.pp_end_guard). Left out of the hash while False.
    pp_end_guard: bool = False
    # fwd only (ImplicitGemmConvSpec.epi_lds_barriers). Left out of the hash while False.
    epi_lds_barriers: bool = False
    # fwd only (ImplicitGemmConvSpec.split_hybrid). Left out of the hash at 0.
    split_hybrid: int = 0
    # fwd only (ImplicitGemmConvSpec.sched_strategy). Left out of the hash at 0.
    sched_strategy: int = 0
    # fwd only (ImplicitGemmConvSpec.split_tail_compact). Left out of the hash at False.
    split_tail_compact: bool = False
    # fwd only (ImplicitGemmConvSpec.epi_quad_xpose). Left out of the hash at 0.
    epi_quad_xpose: int = 0
    # fwd only (ImplicitGemmConvSpec.no_agpr). Left out of the hash while False.
    no_agpr: bool = False
    # fwd and dgrad (ImplicitGemmConvSpec.m_band, DgradConvSpec.m_band). Left out of the hash at 0.
    m_band: int = 0
    # fwd only (ImplicitGemmConvSpec.xh_stride_m). Left out of the hash while False.
    xh_stride_m: bool = False
    # dgrad only (DgradConvSpec.dy_halo / dy_halo_setprio /
    # dy_halo_kouter_pad): the dY halo mode of the tap-outer K loop and its
    # knobs. A halo binary also bakes its filter (filter_h/w) and the widest
    # Wo it serves (halo_max_wo, below). Left out of the hash at 0.
    dy_halo: int = 0
    dy_halo_setprio: int = 0
    dy_halo_kouter_pad: int = 0
    num_load_waves: int = 0
    cshuffle_no_alias: bool = False
    # ---- capability: bounds which problems this binary can serve ----
    is_3d: bool = False
    is_pointwise: bool = False
    # Direct conv bakes the filter geometry and per-group channel counts into
    # the unrolled MFMA chain and the LDS row layout; implicit GEMM leaves
    # them 0 (runtime).
    filter_h: int = 0
    filter_w: int = 0
    filter_d: int = 0
    stride_h: int = 0
    stride_w: int = 0
    dilation_h: int = 0
    dilation_w: int = 0
    pad_h: int = 0
    pad_w: int = 0
    cpg: int = 0
    kpg: int = 0
    # Grouped convolution takes a different code path in every direction (the
    # contraction index only spans one group, so the channel decode gains an
    # embed and the epilogue gains a per-group k_out fold). A binary built one
    # way cannot serve the other, and the difference is invisible at launch --
    # it produces wrong numbers rather than an error -- so it is a capability.
    grouped: bool = False
    # Direct conv only: rows per block (the H-loop trip count).
    block_h: int = 0
    # Dgrad only: the largest tilde sub-GEMM count the record table (and the
    # binary search's runtime trip count) was built to support.
    max_sub_gemms: int = 0
    # Dgrad dY halo only (DgradConvSpec.halo_max_wo): the staged halo is
    # sized for this output width, so the binary serves Wo <= halo_max_wo.
    # 0 = not a halo binary. Left out of the hash at 0.
    halo_max_wo: int = 0
    # async_dma only: elements per DRAM->LDS chunk of the A and B loaders. The
    # width is picked from the build-time cpg so a chunk never straddles a
    # filter position, and is baked into the ISA -- a binary is only correct
    # for problems whose contiguous run it divides.
    async_chunk_a: int = 0
    async_chunk_b: int = 0
    # Direct conv only: the spec's tuning kwargs (block_q, block_w, waves...)
    # as canonical JSON. Direct kernels have no GEMM tile, so their knobs do
    # not fit the tile/warp fields above; ``algorithm`` names the kernel
    # variant and this string carries the rest.
    knobs: str = ""
    # ---- provenance: the LLVM flavor the binary was lowered for ----
    # It changes the HSACO without showing up in the configuration, so a
    # binary for another flavor has to miss. Defaults to this process's
    # flavor, so a lookup only matches binaries lowered the same way.
    #
    # The emitter version is deliberately NOT part of the identity: an
    # identity names a kernel configuration, and the same configuration keeps
    # its entry across emitter changes. Whether its binary is still current is
    # tracked per entry (the build-time emitter digest and the binary's
    # content key in its metadata), which is what lets --compile-all rebuild
    # only the kernels whose code changed.
    llvm_flavor: str = field(default_factory=current_llvm_flavor)

    @property
    def is_direct(self) -> bool:
        """Direct-conv kernels: every capability field is baked, even a 0."""
        return self.algorithm.startswith("direct")

    def stable_hash(self) -> str:
        """Deterministic SHA-1 hex digest (40 chars) of the identity.

        Suitable as a filesystem-safe filename component. Deterministic
        across Python versions (sorted JSON keys, no randomization).
        """
        fields = asdict(self)
        # Left out at its "no scratch" value so identities that predate the
        # field keep their hash (and their cache entries).
        if not fields["ws_replicas"]:
            del fields["ws_replicas"]
        if not fields["prefetch_addr"]:
            del fields["prefetch_addr"]
        if not fields["prefetch_regs"]:
            del fields["prefetch_regs"]
        if fields["lds_kpack"] == 1:
            del fields["lds_kpack"]
        if not fields["xcd_remap"]:
            del fields["xcd_remap"]
        if not fields["sep_addr"]:
            del fields["sep_addr"]
        if not fields["pfr_sched"]:
            del fields["pfr_sched"]
        if not fields["lds_bswz"]:
            del fields["lds_bswz"]
        if not fields["kp_line"]:
            del fields["kp_line"]
        if not fields["xwin"]:
            del fields["xwin"]
        if fields["xwin_rows"] == 1:
            del fields["xwin_rows"]
        if fields["xwin_groups"] == 1:
            del fields["xwin_groups"]
        if not fields["xwin_pp"]:
            del fields["xwin_pp"]
        if fields["xwin_mrow"] == 1:
            del fields["xwin_mrow"]
        if not fields["xwin_gn"]:
            del fields["xwin_gn"]
        if not fields["uniform_k_decode"]:
            del fields["uniform_k_decode"]
        if not fields["reg_prefetch"]:
            del fields["reg_prefetch"]
        if not fields["lds_xor"]:
            del fields["lds_xor"]
        if not fields["a_direct"]:
            del fields["a_direct"]
        if not fields["uk_tap_fast"]:
            del fields["uk_tap_fast"]
        if not fields["patch"]:
            del fields["patch"]
        if not fields["tap_outer_k"]:
            del fields["tap_outer_k"]
        if not fields["tapk_prefetch"]:
            del fields["tapk_prefetch"]
        if not fields["tapk_xhalo"]:
            del fields["tapk_xhalo"]
        if not fields["b_kpack"]:
            del fields["b_kpack"]
        if not fields["tapk_ilv"]:
            del fields["tapk_ilv"]
        if not fields["xhalo_exact"]:
            del fields["xhalo_exact"]
        if fields["cshuffle_m_chunks"] == 1:
            del fields["cshuffle_m_chunks"]
        if not fields["tapk_lds_ahead"]:
            del fields["tapk_lds_ahead"]
        if not fields["tapk_lds_barrier"]:
            del fields["tapk_lds_barrier"]
        if not fields["tapk_setprio"]:
            del fields["tapk_setprio"]
        if not fields["xh_b_wide"]:
            del fields["xh_b_wide"]
        if not fields["xh_b_kpad"]:
            del fields["xh_b_kpad"]
        if not fields["tapk_pingpong"]:
            del fields["tapk_pingpong"]
        if not fields["tapk_pp_sched"]:
            del fields["tapk_pp_sched"]
        if not fields["tapk_pp_epi"]:
            del fields["tapk_pp_epi"]
        for name in ("xh_a_k4", "xh_order", "xh_defer"):
            if not fields[name]:
                del fields[name]
        if not fields["split_addr_k_decode"]:
            del fields["split_addr_k_decode"]
        for name in (
            "lds_ahead", "defer_mfma", "vmem_interleave", "split_tail", "uk_nopad",
            "defer_mfma_per_write",
            "lds_kpack_read", "uk_chunk_group", "rpf_lds_barrier", "rpf_fetch_late",
            "uk_xhalo", "xh_dpp", "rpf_setprio", "rpf_pingpong", "pp_pre_mfma", "xh_wait_smem", "pp_b_double", "lds_xor_rowkey",
            "split_atomic_agent", "split_atomic_pairs", "split_xcd_remap",
            "split_streamk", "split_flag_store", "xh_subtile", "split_atomic_fast",
            "epi_store_fast", "split_zero_first", "pp_fetch_deep",
            "pp_split_fetch", "fb_skip_empty", "pp_end_guard",
            "epi_lds_barriers", "split_hybrid", "sched_strategy", "split_tail_compact",
            "epi_quad_xpose", "no_agpr", "m_band", "xh_stride_m",
        ):
            if not fields[name]:
                del fields[name]
        for name in ("dy_halo", "dy_halo_setprio", "dy_halo_kouter_pad", "halo_max_wo"):
            if not fields[name]:
                del fields[name]
        # Same for the stream-K pair at "no stream-K".
        if fields["streamk"] == "off" and not fields["streamk_reduction"]:
            del fields["streamk"], fields["streamk_reduction"]
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()

    def label(self) -> str:
        """The kernel's name in logs, benchmark tables and instance lists.

        Lossless: it spells every field, so :meth:`from_label` turns it back
        into an equal identity and ``reproduce_kernel.py`` rebuilds the kernel
        from it. Format: :mod:`kernels.conv.kernel_label`. Not the
        HSACO's entry point -- the AOT cache records that in its metadata under
        ``kernel_name``.
        """
        from kernels.conv.kernel_label import encode

        return encode(self)

    @classmethod
    def from_label(cls, name: str) -> "KernelIdentity":
        """The identity :meth:`label` named ``name``.

        Launch parameters a benchmark table appends after whitespace
        (``@split_k=4``) are ignored. Raises ``ValueError`` on a malformed
        name.
        """
        from kernels.conv.kernel_label import decode

        return decode(name, cls)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KernelIdentity":
        """Rebuild from ``to_dict`` output, tolerating older cache entries.

        Unknown keys are dropped rather than raising: a cache written by an
        older build should be ignorable, not fatal. Entries missing a field
        that has since been added fall back to its default, which is also why
        the hash changes and they simply never match again.
        """
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        # An entry that predates the provenance field was lowered for an
        # unknown flavor; "" never equals the current one, so it never matches.
        kw.setdefault("llvm_flavor", "")
        return cls(**kw)


# ---------------------------------------------------------------------------
# Which problems an identity can serve
# ---------------------------------------------------------------------------

# ``DgradConvSpec.max_sub_gemms``'s default; identities written before the
# field existed recorded 0 and were built with it.
_DGRAD_DEFAULT_MAX_SUB_GEMMS = 64


def dgrad_sub_gemms(identity: KernelIdentity, problem: object) -> list:
    """The tilde sub-GEMMs ``problem`` decomposes into for this binary's tile.

    An implicit-GEMM dgrad binary reads them at run time, as the record table
    :func:`~kernels.common.conv_implicit_gemm_dgrad.pack_sub_gemm_buffer`
    packs; its grid is their total tile count.
    """
    from kernels.common.conv_implicit_gemm_dgrad import (
        compute_tilde,
        enumerate_sub_gemms,
        patch_sub_gemms,
    )

    if identity.patch:
        return patch_sub_gemms(
            problem, identity.tile_m, identity.tile_n, tile_k=identity.tile_k
        )
    return enumerate_sub_gemms(
        problem,
        compute_tilde(problem),
        identity.tile_m,
        identity.tile_n,
        tile_k=identity.tile_k,
        split_k=max(1, identity.split_k),
    )


def _group_merge_fits(
    identity: KernelIdentity, problem: object, gm: int
) -> tuple[bool, str]:
    """Can a binary merging ``gm`` groups serve ``problem``?

    Mirrors the shape rules of ``wgrad_group_merge_available`` /
    ``fwd_group_merge_available``: merging is depthwise only, ``gm`` must
    divide the group count, and the merged GEMM must fit one tile -- the
    filter extent is a runtime value, so it is checked here, per problem.
    wgrad's merged GEMM is ``kpg*Gm`` x ``[Z*]Y*X*cpg*Gm``; fwd's only
    merged free axis is GEMM-N, ``kpg*Gm``.
    """
    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    groups = int(getattr(problem, "groups", 1))
    if cpg != 1 or kpg != 1:
        return False, "group-merged kernel needs depthwise (cpg=kpg=1)"
    if groups % gm:
        return False, f"group_merge {gm} does not divide groups {groups}"
    if identity.direction == "fwd":
        if kpg * gm > identity.tile_n:
            return False, f"merged GEMM-N {kpg * gm} exceeds tile_n {identity.tile_n}"
        return True, "ok"
    spatial = int(getattr(problem, "Y", 1)) * int(getattr(problem, "X", 1))
    if bool(getattr(problem, "is_3d", False)):
        spatial *= int(getattr(problem, "Z", 1))
    if kpg * gm > identity.tile_m:
        return False, f"merged GEMM-M {kpg * gm} exceeds tile_m {identity.tile_m}"
    if spatial * cpg * gm > identity.tile_n:
        return False, (
            f"merged GEMM-N {spatial * cpg * gm} exceeds tile_n {identity.tile_n}"
        )
    return True, "ok"


def _grid_fits(identity: KernelIdentity, problem: object) -> tuple[bool, str]:
    """Does the launch grid for ``problem`` fit the hardware limits?

    The tile counts are runtime values of an AOT kernel, so a small tile on
    a large problem can need more workgroups along y (or x) than the
    65535 cap -- the launch then fails or leaves tiles unwritten. Mirrors
    the grid check of ``is_valid_spec_for_problem``; split-K degrees, the
    third axis of wgrad, are capped where the benchmark picks them.
    """
    cap = _MAX_GRID_DIM
    groups = max(1, int(getattr(problem, "groups", 1)))
    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    tm, tn = max(1, identity.tile_m), max(1, identity.tile_n)
    if identity.direction == "fwd":
        gm = max(1, identity.group_merge)
        gy = -(-int(getattr(problem, "M", 0)) // tm)
        gx = -(-(kpg * gm) // tn)
        axes = (
            ("y", gy, f"M={getattr(problem, 'M', 0)} tile_m={tm}"),
            ("x", gx, f"kpg*Gm={kpg * gm} tile_n={tn}"),
            ("z", groups // gm, f"groups/Gm={groups // gm}"),
        )
    elif identity.direction == "wgrad":
        gm = max(1, identity.group_merge)
        spatial = int(getattr(problem, "Y", 1)) * int(getattr(problem, "X", 1))
        if bool(getattr(problem, "is_3d", False)):
            spatial *= int(getattr(problem, "Z", 1))
        gy = -(-(kpg * gm) // tm)
        gx = -(-(spatial * cpg * gm) // tn)
        axes = (
            ("y", gy, f"wg_M={kpg * gm} tile_m={tm}"),
            ("x", gx, f"wg_N={spatial * cpg * gm} tile_n={tn}"),
            ("z", groups // gm, f"groups/Gm={groups // gm}"),
        )
    else:
        axes = (("y", groups, f"groups={groups}"),)
    for axis, extent, what in axes:
        if extent > cap:
            return False, f"grid {axis} {extent} > {cap} (hardware cap): {what}"
    return True, "ok"


def identity_supports_problem(
    identity: KernelIdentity, problem: object, llvm_flavor: str | None = None
) -> tuple[bool, str]:
    """Can a binary built for ``identity`` be launched on ``problem``?

    ``problem`` is a ``ConvProblem`` for the implicit-GEMM identities and a
    ``DirectConvProblem`` for the direct ones; both are read by attribute.
    Returns ``(ok, reason)``; the reason makes an empty candidate list
    diagnosable instead of just "no compatible kernels".

    ``llvm_flavor`` is the flavor to check provenance against; ``None`` (the
    default) falls back to :func:`current_llvm_flavor` (environment/COMGR
    detection), preserving this function's standalone behavior. Callers that
    have a flavor on hand -- a dispatch request, a test -- should pass it
    explicitly instead of relying on process environment.

    Provenance is checked first: a binary lowered for another LLVM flavor is
    never offered. (Emitter changes are not checked here: most leave most
    kernels' code alone, and only a ``--compile-all`` can tell which, by
    re-emitting and comparing content keys.) Then two classes of constraint:

    * **Vector alignment.** The load/store widths are baked into the ISA,
      so each operand's contiguous per-group extent has to stay divisible
      by its width. Which extent that is depends on the direction: fwd
      loads X and W along ``cpg`` and stores Y along ``kpg``; wgrad and
      dgrad load dY along ``kpg`` and run their B operand (X / W) and
      output (dW / dX) along ``cpg``.
    * **Capability.** A kernel that unrolled a 3x3 filter, or that baked
      ``cpg``/``kpg`` into its MFMA chain, simply cannot run another
      geometry. The identity records those, so a mismatch is a hard no
      rather than a silent wrong answer.

    The direct kernels' tuning knobs carry shape rules of their own (groups
    divisible by block_groups, ...), which this capability check does not
    see: :func:`kernels.conv.direct_variants.direct_spec_for_problem`
    re-validates those against the real shape.
    """
    flavor = current_llvm_flavor() if llvm_flavor is None else llvm_flavor
    if identity.llvm_flavor != flavor:
        return False, (
            f"built for LLVM flavor {identity.llvm_flavor or '<unknown>'}, "
            f"this process lowers to {flavor}"
        )

    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    # The contiguous extent of A / B / D. fwd: X (cpg), W (cpg), Y (kpg).
    # wgrad: dY (kpg), X (cpg), dW (cpg). dgrad: dY (kpg), W (cpg),
    # dX (cpg). Direct conv has no vector-width fields (they are 0); its
    # load widths follow the baked cpg/kpg, which are checked exactly below.
    if identity.direction in ("wgrad", "dgrad"):
        extents = ((kpg, "kpg"), (cpg, "cpg"), (cpg, "cpg"))
    else:
        extents = ((cpg, "cpg"), (cpg, "cpg"), (kpg, "kpg"))
    if identity.patch:
        # One GEMM over the filter's (y, x, c): dY along K, W along Y*X*C,
        # dX along the X*C of one filter row, which starts at a multiple of
        # sW*C and of the row Wi*C.
        from kernels.common.conv_implicit_gemm_dgrad import is_patch_problem

        if not is_patch_problem(problem):
            return False, (
                "patch kernel serves patch embeddings only (2-D, ungrouped, "
                "no padding or dilation, stride >= filter)"
            )
        C, X, Y = int(problem.C), int(problem.X), int(problem.Y)
        run = math.gcd(X * C, int(problem.sW) * C, int(problem.Wi) * C)
        extents = ((kpg, "kpg"), (Y * X * C, "Y*X*C"), (run, "dX run"))
    gm = max(1, int(identity.group_merge))
    if gm > 1:
        ok, why = _group_merge_fits(identity, problem, gm)
        if not ok:
            return False, why
        if identity.direction == "fwd":
            # A group-merged fwd kernel (depthwise only) loads X and stores
            # Y along the Gm merged channels; W stays per group (its merged
            # load is per-element masked).
            extents = ((cpg * gm, "cpg*Gm"), (cpg, "cpg"), (kpg * gm, "kpg*Gm"))
        else:
            # A group-merged wgrad kernel (depthwise only) loads dY and X
            # along the Gm merged channels; dW stays per group.
            extents = ((kpg * gm, "kpg*Gm"), (cpg * gm, "cpg*Gm"), (cpg, "cpg"))
    for name, (extent, extent_name) in zip(
        ("vector_size_a", "vector_size_b", "vector_size_c"), extents
    ):
        vec = getattr(identity, name)
        if vec and extent and extent % vec != 0:
            return False, f"{extent_name}={extent} not divisible by {name}={vec}"

    if identity.is_3d != bool(getattr(problem, "is_3d", False)):
        return False, "3-D capability mismatch"
    if identity.is_pointwise and not bool(getattr(problem, "is_pointwise", False)):
        return False, "kernel is pointwise-only"

    # Grouped convolution is a different code path in every direction. An
    # ungrouped binary on a grouped problem gives wrong numbers with no
    # error, so it is checked before anything else that could mask it. The
    # grouped path computes groups == 1 too (group 0, cpg == C), so a
    # grouped binary serves both; the AOT grid only builds those, and
    # ungrouped entries of older caches still serve ungrouped problems.
    # Direct kernels take the group count as a kernarg and index channels
    # per group in every variant, so they carry no grouped/ungrouped split.
    problem_grouped = int(getattr(problem, "groups", 1)) > 1
    if not identity.is_direct and problem_grouped and not identity.grouped:
        return False, "kernel is ungrouped, problem is grouped"

    # Split-K wgrad without two-stage atomic-adds straight into dW. For a
    # 16-bit dW that is a packed <2 x dtype> atomic, which needs an even
    # dW row (wg_N = [Z*]Y*X*cpg) and store width; on an odd row it
    # misaddresses silently. Ask the same predicate the spec validator uses.
    if (
        identity.direction == "wgrad"
        and not identity.is_direct
        and identity.split_k > 1
        and not identity.two_stage
        and hasattr(problem, "Y")
    ):
        from kernels.common.conv_implicit_gemm_wgrad import (
            wgrad_atomic_epilogue_available,
        )

        ok, why = wgrad_atomic_epilogue_available(
            problem, identity.dtype_d, identity.vector_size_c or None
        )
        if not ok:
            return False, why

    # Split-K fwd atomic-adds into D with a packed <2 x dtype> atomic for a
    # 16-bit D, which pairs adjacent output channels of one group: kpg must
    # be even (ImplicitGemmConvSpec's fwd_split_k_conflict).
    if (
        identity.direction == "fwd"
        and not identity.is_direct
        and identity.split_k > 1
        and identity.dtype_d != "fp32"
        and hasattr(problem, "kpg")
        and int(problem.kpg) % 2
    ):
        return False, f"fwd split-K with a 16-bit D needs even kpg (kpg={problem.kpg})"

    # The wgrad X-window kernel bakes the filter width and walks one output
    # row per K tile: stride/dilation 1 along W, whole channel blocks of
    # tile_n / X, ungrouped.
    if identity.direction == "wgrad" and identity.xwin:
        xw = identity.xwin
        xr = max(1, identity.xwin_rows)
        cb = max(1, identity.tile_n // (xw * xr))
        zy = int(getattr(problem, "Z", 1) or 1) * int(getattr(problem, "Y", 1))
        if zy % xr:
            return False, f"xwin_rows={xr} must divide the filter rows Z*Y={zy}"
        if int(getattr(problem, "X", 0)) != xw:
            return False, f"xwin bakes filter width {xw} (problem X={getattr(problem, 'X', None)})"
        if int(getattr(problem, "sW", 1)) != 1 or int(getattr(problem, "dW", 1)) != 1:
            return False, "xwin serves stride-1, dilation-1 W only"
        if int(getattr(problem, "groups", 1)) != 1:
            return False, "xwin serves groups == 1 only"
        if cpg % cb:
            return False, f"xwin needs cpg % {cb} == 0 (cpg={cpg})"
        mr = max(1, getattr(identity, "xwin_mrow", 1) or 1)
        if mr > 1:
            if int(getattr(problem, "Wo", 0)) * mr != identity.tile_k:
                return False, f"xwin_mrow={mr} needs Wo == tile_k / {mr} (Wo={getattr(problem, 'Wo', None)})"
            if int(getattr(problem, "Ho", 0)) % mr:
                return False, f"xwin_mrow={mr} needs Ho % {mr} == 0"
        elif int(getattr(problem, "Wo", 0)) % max(1, identity.tile_k):
            return False, f"xwin needs Wo % tile_k == 0 (Wo={getattr(problem, 'Wo', None)})"

    # uk_nopad drops the per-tap bounds checks: unpadded problems only.
    if (
        identity.direction == "fwd"
        and identity.uk_nopad
        and any(int(getattr(problem, a, 0) or 0) for a in ("pH", "pW", "pD"))
    ):
        return False, "uk_nopad serves unpadded problems only"

    # m_band (fwd / dgrad): a band at least as wide as the output plane's
    # tile row is the launch order, i.e. the same binary as without it --
    # leave it out instead of offering a duplicate candidate.
    if (
        identity.m_band
        and not identity.is_direct
        and identity.direction in ("fwd", "dgrad")
        and identity.tile_m
    ):
        if identity.direction == "fwd":
            _plane = int(getattr(problem, "Ho", 0) or 0) * int(getattr(problem, "Wo", 0) or 0)
        else:
            _plane = int(getattr(problem, "Hi", 0) or 0) * int(getattr(problem, "Wi", 0) or 0)
        _R = max(1, (_plane + identity.tile_m // 2) // identity.tile_m)
        if _plane and _R <= identity.m_band:
            return False, (
                f"m_band {identity.m_band} >= the {_R} tiles of an output plane: "
                "the launch order (same as without m_band)"
            )

    # The split-addressing per-tile reduction decode has no fallback: a K
    # tile must lie inside one filter tap, i.e. cpg % tile_k == 0. (The
    # uniform_k_decode loop keeps the plain loop for the rest.)
    if (
        identity.direction == "fwd"
        and not identity.is_direct
        and identity.split_addr_k_decode
        and hasattr(problem, "cpg")
        and int(problem.cpg) % max(1, identity.tile_k)
    ):
        return False, (
            f"fwd split_addr_k_decode needs cpg % tile_k == 0; cpg={problem.cpg}, "
            f"tile_k={identity.tile_k}"
        )

    # Baked filter geometry (direct conv), and the stride/dilation older
    # implicit-GEMM dgrad caches recorded per binary (current ones record
    # 0: the tilde kernel takes them at run time). For implicit GEMM 0
    # means "runtime"; direct conv bakes all of them, and there 0 is a real
    # value (PAD=0 for a 1x1 filter) that has to match too.
    for name, attrs in (
        ("filter_h", ("KH", "Y")),
        ("filter_w", ("KW", "X")),
        ("stride_h", ("stride", "sH")),
        ("stride_w", ("stride", "sW")),
        ("dilation_h", ("dilation", "dH", "dil_h")),
        ("dilation_w", ("dilation", "dW", "dil_w")),
        ("pad_h", ("PAD", "pH")),
        ("pad_w", ("PAD", "pW")),
    ):
        baked = getattr(identity, name)
        if not baked and not identity.is_direct:
            continue
        actual = next((getattr(problem, a) for a in attrs if hasattr(problem, a)), None)
        if actual is not None and int(actual) != baked:
            return False, f"{name}={baked} but problem has {actual}"

    # A baked stride/dilation (the stride-1 dgrad binary) holds along depth
    # too: on a 3-D problem it only serves sD == stride_h and dD == dilation_h.
    if identity.is_3d and not identity.is_direct:
        for name, attr in (("stride_h", "sD"), ("dilation_h", "dD")):
            baked = getattr(identity, name)
            actual = getattr(problem, attr, None)
            if baked and actual is not None and int(actual) != baked:
                return False, f"{name}={baked} but problem has {attr}={actual}"

    # 0 means runtime here, for direct conv too: the non-grouped direct
    # kernel takes its channel counts as kernargs (cpg == 0 is never a
    # real shape, unlike PAD == 0 above).
    if identity.cpg and identity.cpg != cpg:
        return False, f"kernel baked cpg={identity.cpg}, problem has {cpg}"
    if identity.kpg and identity.kpg != kpg:
        return False, f"kernel baked kpg={identity.kpg}, problem has {kpg}"

    # The async loaders fetch fixed-width chunks along a run that is only
    # contiguous over the per-group channels; a chunk that does not divide
    # it silently reads across filter positions.
    if identity.async_dma:
        if not (identity.async_chunk_a and identity.async_chunk_b):
            return False, "async_dma binary without a recorded chunk width"
        # (A, B) contiguous runs: fwd reads NHWC and KYXC along c; wgrad
        # reads dY along k_out and X along c.
        runs = {"wgrad": (kpg, cpg)}.get(identity.direction, (cpg, cpg))
        for chunk, run, operand in (
            (identity.async_chunk_a, runs[0], "A"),
            (identity.async_chunk_b, runs[1], "B"),
        ):
            if run % chunk != 0:
                return False, (
                    f"async {operand} chunk of {chunk} elements does not "
                    f"divide the problem's contiguous run of {run}"
                )

    # Implicit-GEMM dgrad dispatches CTAs with a binary search whose trip
    # count is runtime, but the sub-GEMM record table the search walks is
    # sized at build time for ``max_sub_gemms``; a problem that decomposes
    # into more would be rejected only at launch.
    if identity.direction == "dgrad" and not identity.is_direct:
        n_sub = len(dgrad_sub_gemms(identity, problem))
        bound = identity.max_sub_gemms or _DGRAD_DEFAULT_MAX_SUB_GEMMS
        if n_sub > bound:
            return False, (
                f"problem needs {n_sub} tilde sub-GEMMs, kernel's record "
                f"table supports {bound} (max_sub_gemms)"
            )
        # Mirrors is_valid_dgrad_spec: the grouped implicit-GEMM dgrad has
        # no depthwise (cpg == 1) path, and the grouped binary is offered
        # to every grouped problem otherwise.
        if int(getattr(problem, "groups", 1)) > 1 and cpg == 1:
            return False, (
                "depthwise dgrad (cpg == 1) is not supported by the "
                "implicit-GEMM grouped path"
            )
        # The tap-outer K loop has no fallback: every K tile must lie inside
        # one filter tap, and the tap must be the plain (y, x). The
        # stride/dilation fields above already pin a stride-1 binary; this
        # states the rule for the loop itself.
        if identity.tap_outer_k:
            strided = any(
                int(getattr(problem, a, 1)) != 1 for a in ("sH", "sW", "dH", "dW")
            )
            if strided:
                return False, "tap_outer_k serves stride 1 / dilation 1 only"
            if kpg % max(1, identity.tile_k):
                return False, (
                    f"tap_outer_k needs kpg % tile_k == 0; kpg={kpg}, "
                    f"tile_k={identity.tile_k}"
                )
        # tapk_pingpong 4 keeps one bit per filter tap in a 32-bit row mask.
        if identity.tapk_pingpong >= 4 and any(
            int(getattr(problem, a, 1) or 1) > 3 for a in ("Y", "X", "Z")
        ):
            return False, "tapk_pingpong 4/5 serves filters of up to 3 per axis"
        # The x-halo stage holds tap x = 0 of every pixel; a pixel with
        # wi < x reads zero, which is right only without a W pad. The
        # filter width it unrolls is baked (filter_w, matched above).
        if identity.tapk_xhalo and int(getattr(problem, "pW", 0)) != 0:
            return False, "tapk_xhalo serves pW == 0 only"
        # x-halo split-K accumulates through a packed-atomic store that
        # treats dX as one dense (M, C) matrix: ungrouped problems only.
        if (
            (identity.tapk_xhalo or identity.tapk_pingpong)
            and identity.split_k > 1
            and int(getattr(problem, "groups", 1)) != 1
        ):
            return False, "x-halo split-K serves groups == 1 only"
        # The dY halo loop shifts the linear pixel index by whole output
        # rows, so the output must be the input's size, and its LDS tile
        # is sized for halo_max_wo. The filter (filter_h/w, baked) is
        # matched above; with stride 1 and Ho == Hi it fixes the pad.
        if identity.dy_halo:
            ho = int(getattr(problem, "Ho", -1))
            wo = int(getattr(problem, "Wo", -1))
            if (ho, wo) != (
                int(getattr(problem, "Hi", -2)),
                int(getattr(problem, "Wi", -2)),
            ):
                return False, "dy_halo needs Ho == Hi and Wo == Wi"
            if wo > identity.halo_max_wo:
                return False, (
                    f"dy_halo binary serves Wo <= {identity.halo_max_wo}, "
                    f"problem has Wo={wo}"
                )

    if not identity.is_direct:
        ok, why = _grid_fits(identity, problem)
        if not ok:
            return False, why

    return True, "ok"


# ---------------------------------------------------------------------------
# Capability keys
# ---------------------------------------------------------------------------
#
# :func:`identity_supports_problem` is a pure function of the identity and the
# problem, but it reads only a part of each: most of an identity is tuning
# (tile shapes it does read, but also pipeline, LDS layout, scheduling) and
# most of a direct problem is extents the baked kernel takes as kernargs. A
# caller filtering one cache of ~10^4 entries per shape over ~10^3 shapes pays
# for that asymmetry ~10^7 times, so both sides get a *capability key*: the
# projection onto what the predicate actually reads. Two identities with the
# same key answer alike for every problem, and two problems with the same key
# are answered alike by every identity -- so the predicate can be evaluated
# once per (identity key, problem key) pair instead of once per entry and
# shape.
#
# Both keys are deliberately conservative: a field listed here that the
# predicate does not read only costs reuse, while one it reads and the key
# omits is a wrong answer. ``library/tests/test_capability_keys.py`` holds the
# two sides to that by perturbing the fields the keys leave out and asserting
# the predicate does not move.

#: Fields of :class:`KernelIdentity` that :func:`identity_supports_problem`
#: reads (directly, or through ``_grid_fits`` / ``_group_merge_fits`` /
#: :func:`dgrad_sub_gemms`). ``is_direct`` is a property over ``algorithm``
#: and stands in for it.
IDENTITY_CAPABILITY_FIELDS = (
    "llvm_flavor",
    "direction",
    "is_direct",
    "is_3d",
    "is_pointwise",
    "grouped",
    "patch",
    "group_merge",
    "tile_m",
    "tile_n",
    "tile_k",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "dtype_d",
    "split_k",
    "two_stage",
    "filter_h",
    "filter_w",
    "stride_h",
    "stride_w",
    "dilation_h",
    "dilation_w",
    "pad_h",
    "pad_w",
    "cpg",
    "kpg",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "max_sub_gemms",
    "tap_outer_k",
    "tapk_xhalo",
    "dy_halo",
    "halo_max_wo",
    "split_addr_k_decode",
    "uk_nopad",
    "xwin",
    "xwin_rows",
    "xwin_mrow",
    "m_band",
)

#: Problem attributes the predicate reads when the identity is a *direct* one.
#: A direct binary bakes its whole geometry and takes batch, image size and
#: group count as kernargs, so the predicate never looks at an extent: it
#: stops at the baked-geometry equalities. Every branch that reads one
#: (``_grid_fits``, the dgrad sub-GEMM table, the patch and dy-halo rules, the
#: wgrad atomic epilogue) is behind ``not identity.is_direct`` or behind a
#: field only an implicit-GEMM identity sets. That is what makes this key
#: coarse enough to be worth having: thousands of shapes collapse onto the few
#: hundred geometries the cache was built for.
DIRECT_PROBLEM_CAPABILITY_ATTRS = (
    "cpg",
    "kpg",
    "groups",
    "is_3d",
    "is_pointwise",
    "Y",
    "X",
    "Z",
    "KH",
    "KW",
    "stride",
    "sH",
    "sW",
    "dilation",
    "dH",
    "dW",
    "dil_h",
    "dil_w",
    "PAD",
    "pH",
    "pW",
)


def identity_capability_key(identity: KernelIdentity) -> tuple:
    """``identity`` projected onto :data:`IDENTITY_CAPABILITY_FIELDS`.

    Identities sharing a key are indistinguishable to
    :func:`identity_supports_problem` -- it is told the same thing about all
    of them -- so a filter over a cache can answer once per key.
    """
    return tuple(getattr(identity, name) for name in IDENTITY_CAPABILITY_FIELDS)


def problem_capability_key(problem: object, *, is_direct: bool) -> tuple | None:
    """``problem`` projected onto what the predicate reads, or ``None``.

    ``is_direct`` says which projection applies: a direct identity reads only
    :data:`DIRECT_PROBLEM_CAPABILITY_ATTRS`, an implicit-GEMM one reads
    extents too (the launch grid, the dgrad sub-GEMM decomposition, the
    dy-halo and patch rules), and rather than track which of those a future
    rule adds, the implicit key is the whole problem. It is then exact rather
    than coarse, which still collapses a shape that repeats across dtypes,
    directions or runs.

    ``None`` means "not keyable" -- an implicit-GEMM problem that is not a
    dataclass, or one whose fields are not hashable. Callers fall back to
    evaluating the predicate.
    """
    if is_direct:
        return tuple(getattr(problem, a, None) for a in DIRECT_PROBLEM_CAPABILITY_ATTRS)
    if not dataclasses.is_dataclass(problem) or isinstance(problem, type):
        return None
    key = tuple(
        (f.name, getattr(problem, f.name, None)) for f in dataclasses.fields(problem)
    )
    try:
        hash(key)
    except TypeError:
        return None
    return key
