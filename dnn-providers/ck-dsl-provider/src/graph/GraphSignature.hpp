// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string_view>

#include "../adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "../adapters/sdpa/SdpaBwdSpec.hpp"
#include "../adapters/sdpa/SdpaSpec.hpp"
#include "../runtime/JitCache.hpp"

namespace ck_dsl_provider {

/// Derives ``JitCache`` keys from hipDNN graph nodes.
///
/// Per plan §3.4 the cache key is a deterministic hash over
///   (op_kind_string, dtype_tuple, shape_tuple, stride_tuple,
///    layout_string, dsl_version_string).
///
/// For M1 we only need conv-fwd. The signature is derived from the
/// ``ConvImplicitGemmSpec`` the adapter already built, so the adapter
/// stays the single FlatBuffer reader. The folded inputs are:
///
///   * ``opKind`` -- the per-op identifier
///     ("conv_implicit_gemm")
///   * ``arch`` -- the target gfx token, passed as a separate argument
///     (an orthogonal compile target, not a spec field -- mirrors the
///     DSL). The HSACO is arch-specific, so builds for different arches
///     must not alias (a gfx950 module on gfx942 fails hipModuleLoadData)
///   * the 13 ``ConvProblem`` fields (N, Hi, Wi, C, K, R, S + sH, sW,
///     pH, pW, dH, dW) so any shape / stride / padding / dilation
///     change produces a distinct key
///   * every codegen knob on the spec (name, tile_*, warp_*,
///     warp_tile_*, wave_size, pipeline, epilogue, async_dma,
///     unroll_k, lds_k_pad, chiplet_*, waves_per_eu). These are
///     constexpr defaults in M1, but folding them keeps the key
///     correct once M2 autotuning starts varying them per shape/arch:
///     a different kernel for the same shape must not alias.
///   * ``CK_DSL_PROVIDER_VERSION_STRING`` -- folded into the hash.
///     Bumping the provider version (which
///     embeds the git SHA of the DSL subtree, per
///     CkDslProviderVersion.cmake) invalidates every prior key, which
///     is the correct behaviour: a DSL change can silently produce a
///     different HSACO for the same logical shape, and we must not
///     hand a stale module back from the cache.
///
/// **Intentionally NOT folded** (each is constant for the lifetime of
/// the current in-memory, single-process cache, so it carries no
/// entropy today -- but each becomes a *required* key input the moment
/// the noted feature lands, and the on-disk cache in particular cannot
/// ship without all of them):
///
///   * dtype -- M1 is FP16-only and dtype is not yet a spec field;
///     required when bf16/fp8 are added (alongside a dtype on the spec)
///   * toolchain version (ROCm / comgr) -- the folded version string is
///     the DSL SHA, not the build toolchain; a disk cache that outlives
///     the ROCm install that produced it must also fold the toolchain
///     version. (Target arch IS folded now -- see the ``arch`` argument
///     above.)
///   * physical tensor layout / memory strides -- M1 assumes the
///     canonical NHWC/KRSC/NHWK layouts; required if other layouts are
///     accepted
///   * Y output dims -- derived from the folded fields (Ho/Wo), so
///     they add no entropy; a malformed Y is the adapter's job to
///     reject, not the cache key's
///
/// **Hash function:** FNV-1a 64-bit. Chosen for being well-understood,
/// stdlib-free, and deterministic across compilers; we don't need
/// cryptographic strength, just a low collision rate over the small
/// signature input.
class GraphSignature {
   public:
    /// Compute a cache key directly from a built spec. This is the
    /// production path: the adapter runs once to build the spec, then
    /// the spec is folded into the hash here. Folding the spec rather
    /// than re-walking the FlatBuffer keeps the adapter as the single
    /// source of truth for what the spec fields mean.
    ///
    /// Folded inputs:
    ///   * ``CK_DSL_PROVIDER_VERSION_STRING`` (provider/DSL version)
    ///   * ``opKind``
    ///   * ``arch`` (target gfx token -- the HSACO is arch-specific)
    ///   * ``spec.problem`` fields (N, Hi, Wi, C, K, R, S + sH, sW,
    ///     pH, pW, dH, dW) so any shape / stride / padding / dilation
    ///     change produces a distinct key
    ///   * every codegen knob on the spec -- see the class docstring
    ///     for the full list and for what is intentionally omitted.
    ///
    /// ``arch`` is a separate argument rather than a spec field,
    /// mirroring the DSL where arch is an orthogonal compile target.
    static SignatureHash computeForSpec(std::string_view opKind, const ConvImplicitGemmSpec& spec,
                                        std::string_view arch);

    /// Compute a cache key directly from a built ``SdpaSpec``. Mirrors
    /// the conv overload's prologue (version string, opKind, arch) and
    /// then folds ONLY the codegen-relevant fields: the problem shape
    /// (B, Hq, Hkv, Sq, Skv, D), the dtype, the mask mode, and the
    /// kernel-name stem.
    ///
    /// The eight stride_* scalars and scale_log2 are NOT folded: they
    /// are launch-time kernel arguments and the compiled kernel + grid
    /// are identical across stride/scale changes, so folding them would
    /// thrash the cache without distinguishing any real codegen output.
    ///
    /// ``arch`` is a separate argument rather than a spec field,
    /// mirroring the DSL where arch is an orthogonal compile target.
    static SignatureHash computeForSpec(std::string_view opKind, const SdpaSpec& spec,
                                        std::string_view arch);

    /// Compute a cache key directly from a built ``SdpaBwdSpec`` for the
    /// FMHA-backward kernel. Mirrors the conv/fwd overloads' prologue
    /// (version string, opKind, arch) and then folds ONLY the
    /// codegen-relevant fields: the problem shape (B, Hq, Hkv, Sq, Skv,
    /// D), the dtype, the mask mode, and the kernel-name stem.
    ///
    /// The stride_* scalars and the scale_* values are NOT folded: they
    /// are launch-time kernel arguments and the compiled kernel + grid
    /// are identical across stride/scale changes, so folding them would
    /// thrash the cache without distinguishing any real codegen output.
    ///
    /// ``arch`` is a separate argument rather than a spec field,
    /// mirroring the DSL where arch is an orthogonal compile target.
    static SignatureHash computeForSpec(std::string_view opKind, const SdpaBwdSpec& spec,
                                        std::string_view arch);

    /// Compute a cache key for the LSE-prep kernel that precedes the
    /// FMHA-backward launch. The prep kernel depends ONLY on the
    /// version/opKind/arch prologue plus B, Hq, and Sq -- it neither
    /// reads K/V nor varies with head_size or the kv sequence length --
    /// so folding only those three shape fields keeps the prep module
    /// cached independently of the bwd module.
    static SignatureHash computeForSdpaLsePrep(std::string_view opKind, const SdpaBwdSpec& spec,
                                               std::string_view arch);

   private:
    GraphSignature() = delete;
};

}  // namespace ck_dsl_provider
