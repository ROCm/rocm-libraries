// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "harness/input_init/InputInitSpec.hpp"

namespace hipdnn_integration_tests
{

// Configuration for input tensor synthesis.
//
// One map holds TensorInit entries keyed by tensor uid. Two write modes:
//
//   set(uid, t)        — operator[], always overwrites. Tests and metadata
//                        use this to force a specific init for a tensor.
//   setDefault(uid, t) — try_emplace, no-op if uid already present.
//                        Declaration functions use this to register
//                        op-specific defaults without stomping overrides.
//
// Both metadata and test code call set(), so priority between them is
// purely temporal: setBundle() runs before the test body, so test calls
// overwrite metadata. Nothing in the type enforces this ordering.
//
// Seeds are independent of init values — setting a seed never stomps a range.
class SynthesisConfig
{
public:
    static constexpr unsigned int K_DEFAULT_SEED_ENTROPY = 42;

    // ── Write (override) — tests and metadata ───────────────────────────

    SynthesisConfig& set(int64_t uid, TensorInit t)
    {
        _inits[uid] = t;
        return *this;
    }

    SynthesisConfig& set(flatbuffers::Optional<int64_t> uid, TensorInit t)
    {
        if(uid.has_value())
        {
            set(*uid, t);
        }
        return *this;
    }

    SynthesisConfig& range(int64_t uid, float lo, float hi)
    {
        return set(uid, TensorInit::free(lo, hi));
    }

    SynthesisConfig& fixed(int64_t uid, float value)
    {
        return set(uid, TensorInit::fixed(value));
    }

    // ── Write (default) — declaration functions ─────────────────────────

    SynthesisConfig& setDefault(int64_t uid, TensorInit t)
    {
        _inits.try_emplace(uid, t);
        return *this;
    }

    SynthesisConfig& setDefault(flatbuffers::Optional<int64_t> uid, TensorInit t)
    {
        if(uid.has_value())
        {
            setDefault(*uid, t);
        }
        return *this;
    }

    // ── Read ────────────────────────────────────────────────────────────

    const std::unordered_map<int64_t, TensorInit>& inits() const
    {
        return _inits;
    }

    const std::unordered_map<int64_t, unsigned int>& seeds() const
    {
        return _seeds;
    }

    TensorInit get(int64_t uid) const
    {
        auto it = _inits.find(uid);
        return it != _inits.end() ? it->second : TensorInit{};
    }

    std::vector<int64_t> unfilled(const std::vector<int64_t>& ownedUids) const
    {
        std::vector<int64_t> result;
        for(const int64_t uid : ownedUids)
        {
            const auto init = get(uid);
            if(init.kind != TensorInit::Kind::FREE && init.kind != TensorInit::Kind::FIXED)
            {
                result.push_back(uid);
            }
        }
        return result;
    }

    // ── Seed config ─────────────────────────────────────────────────────

    SynthesisConfig& seed(int64_t uid, unsigned int value)
    {
        _seeds[uid] = value;
        return *this;
    }

    SynthesisConfig& seedEntropy(unsigned int s)
    {
        _seedEntropy = s;
        return *this;
    }

    SynthesisConfig& fallbackSeed(unsigned int s)
    {
        _fixedSeed = s;
        _seedEntropy = s;
        return *this;
    }

    unsigned int getSeedEntropy() const
    {
        return _seedEntropy;
    }

    std::optional<unsigned int> resolveSeed(int64_t uid) const
    {
        if(auto it = _seeds.find(uid); it != _seeds.end())
        {
            return it->second;
        }
        return _fixedSeed;
    }

private:
    std::unordered_map<int64_t, TensorInit> _inits;
    std::unordered_map<int64_t, unsigned int> _seeds;
    std::optional<unsigned int> _fixedSeed;
    unsigned int _seedEntropy = K_DEFAULT_SEED_ENTROPY;
};

} // namespace hipdnn_integration_tests
