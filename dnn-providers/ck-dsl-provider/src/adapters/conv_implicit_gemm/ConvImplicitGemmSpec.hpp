// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ck_dsl_provider {

/// C++ mirror of ``ck_dsl.instances.conv_implicit_gemm.ConvProblem``.
///
/// All 13 fields are graph-derived (no constexpr defaults at the spec
/// level -- they come from the hipDNN conv-fwd node + its tensors).
/// The dataclass on the Python side does default `sH/sW/pH/pW/dH/dW`
/// to (1,1,0,0,1,1); we keep those defaults here too so a partially-
/// initialised problem is still well-formed for the common case.
///
/// Layouts (per the dataclass docstring):
///   A: NHWC fp16, shape [N, Hi, Wi, C]
///   B: KRSC fp16, shape [K, R, S, C]
///   D: NHWK fp16, shape [N, Ho, Wo, K]
///
/// Implicit-GEMM packing:
///   M     = N * Ho * Wo
///   N_gemm = K
///   K_gemm = R * S * C
struct ConvProblem {
    std::int32_t N{0};
    std::int32_t Hi{0};
    std::int32_t Wi{0};
    std::int32_t C{0};
    std::int32_t K{0};
    std::int32_t R{0};
    std::int32_t S{0};

    std::int32_t sH{1};
    std::int32_t sW{1};
    std::int32_t pH{0};
    std::int32_t pW{0};
    std::int32_t dH{1};
    std::int32_t dW{1};

    /// Output height. Mirrors the Python @property.
    /// Hi + 2*pH - dH*(R-1) - 1) / sH + 1
    constexpr std::int32_t Ho() const noexcept {
        return (Hi + 2 * pH - dH * (R - 1) - 1) / sH + 1;
    }

    /// Output width. Mirrors the Python @property.
    constexpr std::int32_t Wo() const noexcept {
        return (Wi + 2 * pW - dW * (S - 1) - 1) / sW + 1;
    }

    /// Flattened GEMM M dimension.
    constexpr std::int64_t M() const noexcept {
        return static_cast<std::int64_t>(N) * Ho() * Wo();
    }

    constexpr std::int32_t Ngemm() const noexcept {
        return K;
    }

    constexpr std::int64_t Kgemm() const noexcept {
        return static_cast<std::int64_t>(R) * S * C;
    }
};

/// C++ mirror of ``ck_dsl.instances.conv_implicit_gemm.ImplicitGemmConvSpec``.
///
/// Mirrors the Python dataclass for the knobs this struct carries, and
/// their defaults match the dataclass defaults -- so a spec the adapter
/// builds without overrides matches what the DSL constructs from its own
/// defaults. Three dataclass fields are intentionally omitted because
/// the M1 path always leaves them at their dataclass defaults: those
/// defaults are reconstructed on the Python side when the payload
/// arrives:
///   * ``lds_layout`` -- a complex object, always ``None`` for M1
///     (re-derived from ``async_dma``/``lds_k_pad``/``tile_k``);
///   * ``k0_k1_split`` -- defaults ``False``;
///   * ``groups`` -- defaults ``1``.
///
/// The DSL's default is the gfx950-tuned config (tile 64x64x64, warp
/// 2x2, MFMA atom 32x32x16, ``mem`` pipeline, ``default`` epilogue,
/// wave64): valid on gfx950, but gfx942/gfx1151 need a different atom
/// and/or wave size.
/// The provider tests supply those per-arch example configs explicitly
/// rather than relying on these defaults (the cross-arch example config
/// is a test concern, not a production one). ``name`` keeps a
/// provider-specific prefix for kernel identification in profiles.
struct ConvImplicitGemmSpec {
    ConvProblem problem;

    std::string name{"ck_dsl_conv_igemm"};

    // Block tile (mirrors the dataclass defaults).
    std::int32_t tile_m{64};
    std::int32_t tile_n{64};
    std::int32_t tile_k{64};

    // Warp grid within the block (mirrors the dataclass defaults).
    std::int32_t warp_m{2};
    std::int32_t warp_n{2};

    // MFMA atom (mirrors the dataclass default: 32x32x16 f16). This is
    // gfx950-valid; gfx942/gfx1151 require the 16x16x16 atom instead.
    std::int32_t warp_tile_m{32};
    std::int32_t warp_tile_n{32};
    std::int32_t warp_tile_k{16};

    std::int32_t wave_size{64};

    // Pipeline / epilogue knobs (mirror the dataclass defaults).
    std::string pipeline{"mem"};
    std::string epilogue{"default"};
    bool async_dma{false};
    bool unroll_k{false};

    // Optional LDS K-pad override (None on the dataclass).
    std::optional<std::int32_t> lds_k_pad{std::nullopt};

    // Chiplet-aware grid swizzle (dataclass defaults to off; the
    // numeric knobs only take effect when the bool is on).
    bool chiplet_swizzle{false};
    std::int32_t chiplet_wgm{8};
    std::int32_t chiplet_num_xcds{8};
    std::int32_t chiplet_chunk_size{64};

    // AMDGPU waves_per_eu hint; None on the dataclass.
    std::optional<std::int32_t> waves_per_eu{std::nullopt};

    /// Block size = warp_m * warp_n * wave_size. Mirrors the Python
    /// @property; used by the test to cross-check warp_m/warp_n.
    constexpr std::int32_t block_size() const noexcept {
        return warp_m * warp_n * wave_size;
    }
};

}  // namespace ck_dsl_provider
