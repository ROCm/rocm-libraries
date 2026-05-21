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
/// Field-for-field with the Python dataclass except for ``lds_layout``,
/// which is a complex object on the Python side and is always ``None``
/// for the M1 path; we omit it here and rely on the dataclass default
/// (re-derived from ``async_dma``/``lds_k_pad``/``tile_k``) when the
/// payload reaches Python.
///
/// **Constexpr defaults follow the bake-off values, NOT the dataclass
/// defaults** -- per PREP_FINDINGS P-5 the bake-off overrides several
/// dataclass defaults and those overrides are the values we want at
/// JIT time. The deltas (dataclass -> bake-off):
///
///   tile_k:       128 -> 64
///   warp_tile_m:  16  -> 32
///   warp_tile_n:  16  -> 32
///   warp_tile_k:  32  -> 16
///   epilogue:     "default" -> "cshuffle"  (largest single perf lever)
///
/// All other fields keep their dataclass defaults.
struct ConvImplicitGemmSpec {
    ConvProblem problem;

    std::string name{"ck_dsl_conv_igemm"};

    // Block tile -- dataclass defaults 64/64/128. Bake-off: tile_k=64.
    std::int32_t tile_m{64};
    std::int32_t tile_n{64};
    std::int32_t tile_k{64};

    // Warp grid (within block).
    std::int32_t warp_m{2};
    std::int32_t warp_n{2};

    // MFMA atom -- dataclass defaults 16/16/32. Bake-off: 32/32/16
    // (uses the 32x32x16 MFMA atom rather than 16x16x32).
    std::int32_t warp_tile_m{32};
    std::int32_t warp_tile_n{32};
    std::int32_t warp_tile_k{16};

    std::int32_t wave_size{64};

    // Pipeline / epilogue knobs.
    std::string pipeline{"mem"};
    std::string epilogue{"cshuffle"};  // dataclass default is "default"
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
