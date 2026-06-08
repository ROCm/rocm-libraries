// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace ck_dsl_provider {

/// Performance knobs the provider selects for the ck_dsl implicit-GEMM
/// convolution kernel (``ImplicitGemmConvSpec`` in
/// ``instances/conv_implicit_gemm/conv_implicit_gemm.py``).
///
/// This is the interface the scorer-driven selection produces and the
/// adapter consumes when it overlays per-knob fields onto the base
/// ``ConvImplicitGemmSpec`` before handing the spec to the Python compile
/// path. Pure value type (no owning pointers, no allocation) so it can be
/// enumerated, copied, and compared freely.
///
/// **Phase 1 scope**: the knobs here cover the axes the ck_dsl
/// ``ImplicitGemmConvSpec`` already exposes. CK-Tile's grouped-conv
/// training set also varies ``wave_mode`` / ``has_dsb`` / ``has_si``
/// (the suffix-aware features that lifted top-1 from ~5% to ~28%), but
/// ck_dsl has no spec field for them yet. Phase 1 pins those features to
/// constants when extracting; Phase 2 will add them as enumerated knobs
/// once the DSL side grows the matching spec lanes.
///
/// All training data is bf16 / gfx950 only, so the scorer-driven path
/// is bf16-only. fp16 / fp32 fall through to the analytic fallback.
///
/// The training data covers 10 ``(tile_m, tile_n, tile_k)`` triples
/// (TILE_TO_WAVE / TILE_TO_WARP in ``grouped_config_rules.py``); the
/// candidate enumerator emits exactly those triples, each pinned to its
/// table-prescribed ``(warp_m, warp_n)`` (CK wave grid) and
/// ``(warp_tile_m, warp_tile_n, warp_tile_k)`` (MFMA atom). ``tile_*`` is
/// the only continuous axis enumerated independently here; the wave grid
/// and MFMA atom are derived from it.
struct ConvImplicitGemmPerfKnobs {
    // --- Block tile (scored continuous axis) -------------------------

    /// Block GEMM-M tile. Valid in the trained TILE_TO_WAVE table:
    /// {16, 32, 64, 128}. Mirrors ``ImplicitGemmConvSpec.tile_m``.
    std::int32_t tile_m{64};

    /// Block GEMM-N tile. Valid in the trained TILE_TO_WAVE table:
    /// {64, 128}. Mirrors ``ImplicitGemmConvSpec.tile_n``.
    std::int32_t tile_n{64};

    /// Block GEMM-K tile. Valid in the trained TILE_TO_WAVE table:
    /// {64, 128}. Mirrors ``ImplicitGemmConvSpec.tile_k``.
    std::int32_t tile_k{64};

    // --- Wave grid (derived from tile via TILE_TO_WAVE table) --------

    /// Number of waves along block-M. The enumerator sets this from the
    /// TILE_TO_WAVE table for the chosen ``(tile_m, tile_n, tile_k)``
    /// triple; it is not independently varied. Mirrors
    /// ``ImplicitGemmConvSpec.warp_m`` (DSL uses "warp" where CK uses
    /// "wave" -- same quantity).
    std::int32_t warp_m{2};

    /// Number of waves along block-N (TILE_TO_WAVE-derived).
    std::int32_t warp_n{2};

    // --- MFMA atom (derived from tile via TILE_TO_WARP table) --------

    /// MFMA atom M. Valid in the trained TILE_TO_WARP table: {16, 32}.
    std::int32_t warp_tile_m{32};

    /// MFMA atom N. Valid in the trained TILE_TO_WARP table: {16, 32}.
    std::int32_t warp_tile_n{32};

    /// MFMA atom K. Valid in the trained TILE_TO_WARP table: {16, 32}.
    std::int32_t warp_tile_k{16};

    // --- Pipeline (scored categorical axis) --------------------------

    /// Pipeline name. Valid in VARIANT_PIPELINES["forward"]:
    /// {basic_v1, mem, compv3, compv4, compv5, compv6,
    ///  comp_async, basic_async_v1}.
    ///
    /// Of those, the model's training set (PIPELINE_MAP in
    /// feature_engine.py) only encodes six categorical IDs: compv3=0,
    /// compv4=1, compv5=2, mem=3, preshufflev2=4, basic_v1=5, compv6=6.
    /// ``comp_async`` / ``basic_async_v1`` get encoded as 0 (the
    /// PIPELINE_MAP default) -- they fall onto the ``compv3`` split, so
    /// they are not OOV but they are not distinguishable from compv3 in
    /// the score either. The enumerator emits them anyway for build-side
    /// coverage; the scorer simply cannot rank them apart from compv3
    /// and the deterministic tie-break decides between them.
    std::string pipeline{"mem"};

    // --- Wave size ---------------------------------------------------

    /// AMDGPU wave size. Fixed at 64 on gfx950 (the only architecture
    /// with conv training data). Kept here so the derived helpers don't
    /// need to know the arch.
    std::int32_t wave_size{64};

    /// Derived block_size = warp_m * warp_n * wave_size. This is the
    /// CTA thread count fed to the feature engine as the ``block_size``
    /// kernel feature (NOT the same as ``tile_k`` -- the feature engine
    /// uses ``block_size`` to estimate ``num_warps = block_size / 4.0``,
    /// to compute ``num_tiles_k = ceil(gemm_k / block_size)``, and as
    /// the K-stride in the LDS estimate).
    [[nodiscard]] std::int32_t block_size() const {
        return warp_m * warp_n * wave_size;
    }
};

}  // namespace ck_dsl_provider
