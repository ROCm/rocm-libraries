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
                                             const ConvImplicitGemmSpec& spec) {
    std::uint64_t h = kFnv1aOffset;

    // Provider/DSL version string. Folding the macro contents
    // (including the git SHA suffix) means any DSL or provider change
    // invalidates the namespace.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
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

    return static_cast<SignatureHash>(h);
}

}  // namespace ck_dsl_provider
