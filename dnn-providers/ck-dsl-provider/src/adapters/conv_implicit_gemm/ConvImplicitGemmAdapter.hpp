// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <string>
#include <unordered_map>

#include "ConvImplicitGemmSpec.hpp"

namespace ck_dsl_provider {

/// Walks a single hipDNN ConvolutionFwd node plus the surrounding
/// tensor map and produces a fully-populated ``ConvImplicitGemmSpec``.
///
/// The ~15 graph-derived fields per plan §4 are:
///   * 13 ConvProblem fields: N, Hi, Wi, C, K, R, S, sH, sW, pH, pW,
///     dH, dW (extracted from X/W tensor dims + the conv-fwd attrs'
///     stride/padding/dilation)
///   * 2 top-level spec fields: ``name`` (kernel-name stem; we keep it
///     as the constexpr default "ck_dsl_conv_igemm" for M1 since
///     cache-key stability comes from the GraphSignature in I-7) and
///     the dtype (implicit -- M1 only handles FP16).
///
/// The codegen knobs that depend on the target arch (the warp-tile MMA
/// atom + wave size) are NOT fixed by ``buildSpec``: they are selected
/// per-arch by ``applyArchCodegenConfig`` once the device arch is known
/// (the plan builder detects the arch and applies the config before the
/// signature/compile step). The remaining knobs (tile_*, warp_*,
/// pipeline, epilogue, chiplet_*, etc.) keep their example constexpr
/// defaults from ``ConvImplicitGemmSpec``. Autotuning is M2+ work.
///
/// Validation (throws HipdnnPluginException on any failure -- callers
/// can catch + return ``false`` from ``isApplicable``):
///   * Spatial dimensions == 2 (M1 is 2D conv only)
///   * X/W dims size == 4 (NCHW logical)
///   * X / W / Y data_type == FLOAT16 (M1 is FP16 only)
///   * stride/dilation/pre_padding all present and size 2
///   * pre_padding == post_padding (asymmetric padding unsupported)
///   * conv_mode == CROSS_CORRELATION (true convolution unsupported)
///   * X.C == W.C (channel dim must match; the DSL uses input C for
///     the implicit-GEMM K_gemm = R*S*C calculation)
///
/// All extracted scalars are narrowed from int64_t to int32_t via
/// explicit cast; the validator first checks each value fits in
/// int32_t (the DSL's signature is i32 for shape scalars).
class ConvImplicitGemmAdapter {
   public:
    using ConvolutionFwdAttributes = hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes;
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorMap = std::unordered_map<std::int64_t, const TensorAttributes*>;

    /// Build the spec from a single conv-fwd node's attributes + the
    /// tensor map that resolves its X/W/Y UIDs.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` on any
    /// validation failure listed in the class docstring.
    static ConvImplicitGemmSpec buildSpec(const ConvolutionFwdAttributes& convAttr,
                                          const TensorMap& tensorMap);

    /// Overwrite the spec's arch-dependent codegen knobs -- the
    /// warp-tile MMA atom (``warp_tile_m/n/k``) and ``wave_size`` -- with
    /// the provider's per-arch default config for ``arch``. The problem
    /// geometry and every other knob (tile_*, warp_*, pipeline,
    /// epilogue, ...) are left untouched.
    ///
    /// Returns ``true`` when ``arch`` has a known config (the spec is
    /// updated), ``false`` when it does not (the spec is left unchanged
    /// and the caller should treat the graph as inapplicable on that
    /// device). ``arch`` is the bare gfx token from ``detectDeviceArch``
    /// (feature suffixes already stripped).
    ///
    /// Per-arch config (the minimal correct config the DSL validates per
    /// target; tuning is M2+):
    ///   * gfx950  -> 32x32x16 f16 MFMA atom, wave64 -- the historical
    ///     default, kept so gfx950 kernel selection is unchanged.
    ///   * gfx942  -> 16x16x16 f16 MFMA atom, wave64 (gfx942 lacks the
    ///     wider 32x32x16 f16 atom).
    ///   * gfx1151 -> 16x16x16 f16 WMMA atom, wave32 (wave32 RDNA).
    static bool applyArchCodegenConfig(ConvImplicitGemmSpec& spec, const std::string& arch);

   private:
    ConvImplicitGemmAdapter() = delete;
};

}  // namespace ck_dsl_provider
