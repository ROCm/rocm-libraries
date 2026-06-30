// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "harness/input_init/InputInitSpec.hpp"

namespace hipdnn_integration_tests
{

// User- and metadata-facing configuration for input synthesis.
// Two maps hold TensorInit entries at different priority levels;
// resolve(uid) walks them in order:
//
//   1. Code      — test C++ sets these via range/fixed/init.
//   2. Metadata  — harness loads bundle JSON here via metadataInit.
//   3. (fallback) — TensorInit{} (lo=-1, hi=1, kind=FREE).
//
// Op-specific defaults (e.g. variance in [0.5, 1.5]) live on the
// InputSynthesizer, not here — they are internal synthesis knowledge,
// not user or metadata configuration.
//
// Seeds are independent of ranges — setting a seed never stomps a range.
// Resolution (InputSynthesizer handles this):
//   1. Per-tensor seed  →  .seed(uid, value)
//   2. Fixed seed       →  .fixedSeedPerTensor(value)
//   3. Default          →  draw from rng seeded with seedEntropy (unique per tensor)
class SynthesisConfig
{
public:
    static constexpr unsigned int K_DEFAULT_SEED_ENTROPY = 42;

    // ── Code layer (highest priority) ────────────────────────────────────

    SynthesisConfig& range(int64_t uid, float lo, float hi)
    {
        _code[uid].kind = TensorInit::Kind::FREE;
        _code[uid].lo = lo;
        _code[uid].hi = hi;
        return *this;
    }

    SynthesisConfig& fixed(int64_t uid, float value)
    {
        _code[uid] = TensorInit::fixed(value);
        return *this;
    }

    SynthesisConfig& init(int64_t uid, const TensorInit& tensorInit)
    {
        _code[uid] = tensorInit;
        return *this;
    }

    SynthesisConfig& seed(int64_t uid, unsigned int value)
    {
        _seeds[uid] = value;
        return *this;
    }

    // ── Metadata layer ───────────────────────────────────────────────────

    SynthesisConfig& metadataInit(int64_t uid, const TensorInit& tensorInit)
    {
        _metadata[uid] = tensorInit;
        return *this;
    }

    SynthesisConfig& metadataSeedEntropy(unsigned int seed)
    {
        _seedEntropy = seed;
        return *this;
    }

    // ── Resolution ───────────────────────────────────────────────────────

    std::optional<TensorInit> resolve(int64_t uid) const
    {
        if(auto it = _code.find(uid); it != _code.end())
        {
            return it->second;
        }
        if(auto it = _metadata.find(uid); it != _metadata.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    // ── Seed config ──────────────────────────────────────────────────────

    SynthesisConfig& seedEntropy(unsigned int seed)
    {
        _seedEntropy = seed;
        return *this;
    }

    SynthesisConfig& fixedSeedPerTensor(unsigned int seed)
    {
        _fixedSeed = seed;
        _seedEntropy = seed;
        return *this;
    }

    std::optional<unsigned int> resolveSeed(int64_t uid) const
    {
        if(auto it = _seeds.find(uid); it != _seeds.end())
        {
            return it->second;
        }
        return _fixedSeed;
    }

    unsigned int getSeedEntropy() const
    {
        return _seedEntropy;
    }

private:
    std::unordered_map<int64_t, TensorInit> _code;
    std::unordered_map<int64_t, TensorInit> _metadata;
    std::unordered_map<int64_t, unsigned int> _seeds;
    std::optional<unsigned int> _fixedSeed;
    unsigned int _seedEntropy = K_DEFAULT_SEED_ENTROPY;
};

} // namespace hipdnn_integration_tests
