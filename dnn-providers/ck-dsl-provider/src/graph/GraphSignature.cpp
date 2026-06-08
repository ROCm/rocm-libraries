// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphSignature.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

#include "version.h"

namespace ck_dsl_provider {

namespace {

constexpr std::uint64_t kFnv1aOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnv1aPrime = 0x100000001b3ULL;

inline std::uint64_t fnv1aFold(std::uint64_t h, std::uint8_t byte) {
    return (h ^ static_cast<std::uint64_t>(byte)) * kFnv1aPrime;
}

inline std::uint64_t fnv1aBytes(std::uint64_t h, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h = fnv1aFold(h, p[i]);
    }
    return h;
}

inline std::uint64_t fnv1aString(std::uint64_t h, std::string_view s) {
    return fnv1aBytes(h, s.data(), s.size());
}

inline std::uint64_t fnv1aI32(std::uint64_t h, std::int32_t v) {
    return fnv1aBytes(h, &v, sizeof(v));
}

inline std::uint64_t fnv1aBool(std::uint64_t h, bool b) {
    return fnv1aFold(h, b ? 0x01 : 0x00);
}

// Fold an optional<i32> as a presence discriminator followed by the
// value when set. The discriminator keeps ``nullopt`` distinct from a
// present ``0`` (otherwise both would fold nothing / a zero and alias).
inline std::uint64_t fnv1aOptI32(std::uint64_t h, const std::optional<std::int32_t>& v) {
    h = fnv1aFold(h, v.has_value() ? 0x01 : 0x00);
    if (v.has_value()) {
        h = fnv1aI32(h, *v);
    }
    return h;
}

}  // namespace

SignatureHash GraphSignature::computeForSpec(std::string_view opKind,
                                             const ConvImplicitGemmSpec& spec,
                                             std::string_view arch) {
    std::uint64_t h = kFnv1aOffset;

    // Provider/DSL version string. Folding the macro contents
    // (including the git SHA suffix) means any DSL or provider change
    // invalidates the namespace.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
    h = fnv1aFold(h, 0x00);

    // Target GPU arch. The HSACO is arch-specific (a gfx950 code object
    // launched on gfx942 yields hipError 209 "no kernel image"), so the
    // key MUST distinguish builds for different arches -- otherwise a
    // multi-arch process (or a persisted disk cache) would alias them
    // and hand back the wrong module. arch is an orthogonal compile
    // target (not a spec field, mirroring the DSL), passed in by the
    // plan builder from the detected device.
    h = fnv1aString(h, arch);
    h = fnv1aFold(h, 0x00);

    // ConvProblem fields, in declaration order so a future field
    // addition to the spec extends the hash deterministically (just
    // add the new field at the bottom; the existing prefix stays
    // identical for unchanged shapes, so cache entries for
    // pre-extension keys remain reachable until the version-string
    // fold bumps them).
    const auto& p = spec.problem;
    h = fnv1aI32(h, p.N);
    h = fnv1aI32(h, p.Hi);
    h = fnv1aI32(h, p.Wi);
    h = fnv1aI32(h, p.C);
    h = fnv1aI32(h, p.K);
    h = fnv1aI32(h, p.R);
    h = fnv1aI32(h, p.S);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.sH);
    h = fnv1aI32(h, p.sW);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.pH);
    h = fnv1aI32(h, p.pW);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.dH);
    h = fnv1aI32(h, p.dW);
    h = fnv1aFold(h, 0x00);

    // Codegen knobs. Every field below changes the emitted HSACO (tile
    // shape, MFMA atom, pipeline/epilogue, occupancy hints, grid
    // swizzle, kernel name). They are all constexpr defaults in M1, so
    // folding them is behaviour-identical today -- but it makes the key
    // correct-by-construction for M2 autotuning, which will vary these
    // per shape/arch. Omitting them would let an autotuned kernel
    // collide with a default-tuned one of the same shape and hand back
    // the wrong module. New knobs append at the bottom, same as the
    // ConvProblem block.
    h = fnv1aString(h, spec.name);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, spec.tile_m);
    h = fnv1aI32(h, spec.tile_n);
    h = fnv1aI32(h, spec.tile_k);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, spec.warp_m);
    h = fnv1aI32(h, spec.warp_n);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, spec.warp_tile_m);
    h = fnv1aI32(h, spec.warp_tile_n);
    h = fnv1aI32(h, spec.warp_tile_k);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, spec.wave_size);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.pipeline);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.epilogue);
    h = fnv1aFold(h, 0x00);
    h = fnv1aBool(h, spec.async_dma);
    h = fnv1aBool(h, spec.unroll_k);
    h = fnv1aOptI32(h, spec.lds_k_pad);
    h = fnv1aFold(h, 0x00);
    h = fnv1aBool(h, spec.chiplet_swizzle);
    h = fnv1aI32(h, spec.chiplet_wgm);
    h = fnv1aI32(h, spec.chiplet_num_xcds);
    h = fnv1aI32(h, spec.chiplet_chunk_size);
    h = fnv1aFold(h, 0x00);
    h = fnv1aOptI32(h, spec.waves_per_eu);
    h = fnv1aFold(h, 0x00);

    // dtype appended at the bottom (append-only ordering preserves
    // pre-dtype keys for unchanged-shape entries). Today the adapter is
    // fp16-only so this folds a constant string into the hash; the slot
    // is here so a later widening doesn't have to bump the version
    // string to invalidate.
    h = fnv1aString(h, spec.dtype);

    return static_cast<SignatureHash>(h);
}

SignatureHash GraphSignature::computeForSpec(std::string_view opKind, const SdpaSpec& spec,
                                             std::string_view arch) {
    std::uint64_t h = kFnv1aOffset;

    // Same prologue as the conv overload: provider/DSL version string,
    // opKind, arch -- each terminated by a 0x00 separator so adjacent
    // fields cannot run together and alias.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, arch);
    h = fnv1aFold(h, 0x00);

    // SdpaProblem shape fields. These are the codegen inputs: any change
    // to batch / head counts / sequence lengths / head_size produces a
    // distinct kernel binary and grid.
    const auto& p = spec.problem;
    h = fnv1aI32(h, p.B);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Hq);
    h = fnv1aI32(h, p.Hkv);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Sq);
    h = fnv1aI32(h, p.Skv);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.D);
    h = fnv1aFold(h, 0x00);

    // dtype, mask mode, and the kernel-name stem all change the emitted
    // HSACO; fold them after the shape block.
    h = fnv1aString(h, spec.dtype);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.mask_mode);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.name);
    h = fnv1aFold(h, 0x00);

    // Opt-in forward stats (LSE) output. Codegen-relevant: stats-on emits
    // the 16-arg kernel (appends the LSE_out pointer) while stats-off
    // keeps the 15-arg kernel, so the two must get distinct cache keys.
    // Appended after the name so existing stats-off keys keep their
    // prefix (the version-string fold ultimately bumps stale entries).
    h = fnv1aBool(h, spec.generate_stats);
    h = fnv1aFold(h, 0x00);

    // Unified paged/varlen problem lanes. Each describes a distinct
    // marshalling path / KV layout the kernel sees (paged vs dense,
    // varlen vs fixed-length, windowed vs full causal, sinks vs none),
    // so each emits a distinct kernel binary and grid and must cache
    // distinctly. Same separator style as the conv overload.
    h = fnv1aBool(h, spec.is_paged);
    h = fnv1aI32(h, spec.block_size);
    h = fnv1aFold(h, 0x00);
    h = fnv1aBool(h, spec.is_varlen);
    h = fnv1aI32(h, spec.sliding_window);
    h = fnv1aFold(h, 0x00);
    h = fnv1aBool(h, spec.use_sinks);
    h = fnv1aFold(h, 0x00);

    // Chosen perf knobs. The scorer-driven selection writes these onto
    // the spec before the key is computed; distinct scored configs must
    // cache distinctly. The continuous axes (num_warps, block_m_per_warp,
    // tile_size, waves_per_eu) and the curated boolean flags all change
    // the emitted HSACO (tile shape, MFMA atom, schedule pipeline,
    // occupancy hint, paged-KV descriptor), so fold them all -- mirroring
    // how the conv overload folds its codegen knobs. Folding these changes
    // the hash for every SDPA spec (defaults included); that only
    // invalidates stale JIT cache entries, never correctness.
    const auto& k = spec.knobs;
    h = fnv1aI32(h, k.num_warps);
    h = fnv1aI32(h, k.block_m_per_warp);
    h = fnv1aI32(h, k.tile_size);
    h = fnv1aI32(h, k.waves_per_eu);
    h = fnv1aFold(h, 0x00);
    h = fnv1aBool(h, k.use_mfma_32x32);
    h = fnv1aBool(h, k.use_transposed_qk_32x32);
    h = fnv1aBool(h, k.use_register_pv);
    h = fnv1aBool(h, k.use_early_v_schedule);
    h = fnv1aBool(h, k.use_fast_paged_kv_desc);

    // Intentionally NOT folded: the eight stride_* scalars and
    // scale_log2. They are launch-time kernel arguments -- the compiled
    // kernel + grid are identical across stride/scale changes, so
    // folding them would thrash the cache without distinguishing any
    // real codegen output.

    return static_cast<SignatureHash>(h);
}

SignatureHash GraphSignature::computeForSpec(std::string_view opKind, const SdpaBwdSpec& spec,
                                             std::string_view arch) {
    std::uint64_t h = kFnv1aOffset;

    // Same prologue as the conv/fwd overloads: provider/DSL version
    // string, opKind, arch -- each terminated by a 0x00 separator so
    // adjacent fields cannot run together and alias.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, arch);
    h = fnv1aFold(h, 0x00);

    // SdpaBwdProblem shape fields. These are the codegen inputs: any
    // change to batch / head counts / sequence lengths / head_size
    // produces a distinct kernel binary and grid.
    const auto& p = spec.problem;
    h = fnv1aI32(h, p.B);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Hq);
    h = fnv1aI32(h, p.Hkv);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Sq);
    h = fnv1aI32(h, p.Skv);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.D);
    h = fnv1aFold(h, 0x00);

    // dtype, mask mode, and the kernel-name stem all change the emitted
    // HSACO; fold them after the shape block.
    h = fnv1aString(h, spec.dtype);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.mask_mode);
    h = fnv1aFold(h, 0x00);
    h = fnv1aString(h, spec.name);

    // Intentionally NOT folded: the stride_* scalars and the scale_*
    // values. They are launch-time kernel arguments -- the compiled
    // kernel + grid are identical across stride/scale changes, so
    // folding them would thrash the cache without distinguishing any
    // real codegen output.

    return static_cast<SignatureHash>(h);
}

SignatureHash GraphSignature::computeForSdpaLsePrep(std::string_view opKind,
                                                    const SdpaBwdSpec& spec,
                                                    std::string_view arch) {
    std::uint64_t h = kFnv1aOffset;

    // Same prologue as the other overloads.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, arch);
    h = fnv1aFold(h, 0x00);

    // The LSE-prep kernel only depends on B, Hq, and Sq -- it reads the
    // contiguous [B, Hq, Sq] stats buffer and writes the M/L scratch, and
    // is independent of head_size and the kv sequence length. Folding
    // only these three fields keeps the prep module cached independently
    // of the bwd module (which folds the full shape).
    const auto& p = spec.problem;
    h = fnv1aI32(h, p.B);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Hq);
    h = fnv1aFold(h, 0x00);
    h = fnv1aI32(h, p.Sq);

    return static_cast<SignatureHash>(h);
}

}  // namespace ck_dsl_provider
