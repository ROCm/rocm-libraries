// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <string_view>
#include <unordered_map>

#include "../runtime/JitCache.hpp"

namespace ck_dsl_provider {

/// Derives ``JitCache`` keys from hipDNN graph nodes.
///
/// Per plan §3.4 the cache key is a deterministic hash over
///   (op_kind_string, dtype_tuple, shape_tuple, stride_tuple,
///    layout_string, dsl_version_string).
///
/// For M1 we only need conv-fwd. The signature inputs are:
///
///   * ``opKind`` -- the per-op identifier
///     ("conv_implicit_gemm")
///   * X/W/Y tensor data_type
///   * 13 ConvProblem-equivalent fields lifted from X.dims, W.dims,
///     and the conv attrs' stride/padding/dilation. The adapter does
///     the same lift in I-6; we duplicate the read here so the
///     signature derivation can run on the hot path without
///     constructing a full ConvImplicitGemmSpec first.
///   * ``CK_DSL_PROVIDER_VERSION_STRING`` -- folded into the hash via
///     ``hashDslVersion()``. Bumping the provider version (which
///     embeds the git SHA of the DSL subtree, per
///     CkDslProviderVersion.cmake) invalidates every prior key, which
///     is the correct behaviour: a DSL change can silently produce a
///     different HSACO for the same logical shape, and we must not
///     hand a stale module back from the cache.
///
/// **Hash function:** FNV-1a 64-bit. Chosen for being well-understood,
/// stdlib-free, and deterministic across compilers; we don't need
/// cryptographic strength, just a low collision rate over the small
/// signature input.
class GraphSignature {
   public:
    using ConvolutionFwdAttributes = hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes;
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorMap = std::unordered_map<std::int64_t, const TensorAttributes*>;

    /// Compute a cache key for a single conv-fwd node + its tensors.
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` on any
    /// validation failure (missing tensor, malformed dims, missing
    /// spatial attr). Validation duplicates what the adapter would
    /// reject so a cache lookup that misses for a "shouldn't be
    /// applicable" graph fails the same way the buildPlan path
    /// would -- callers can rely on a successful signature meaning
    /// the adapter will also succeed.
    static SignatureHash computeForConvFwd(std::string_view opKind,
                                           const ConvolutionFwdAttributes& convAttr,
                                           const TensorMap& tensorMap);

   private:
    GraphSignature() = delete;
};

}  // namespace ck_dsl_provider
