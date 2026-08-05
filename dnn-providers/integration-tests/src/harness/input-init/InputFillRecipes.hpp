// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <nlohmann/json.hpp>

#include "harness/input-init/FillRecipe.hpp"

namespace hipdnn_integration_tests
{

// The fill recipes for a graph's input tensors: one FillRecipe per tensor,
// plus seed overrides. fillInputs() consults this to fill each input.
//
// Two maps for two independent concerns:
//   _fills  — how to fill each tensor (kind + params)
//   _seeds  — per-tensor seed overrides
//
// One scalar:
//   _globalSeed — seeds the RNG that generates per-tensor seeds
//
// Write modes:
//   set(uid, f)        — operator[], always overwrites. Tests and metadata
//                        use this to force a specific fill for a tensor.
//   setDefault(uid, f) — try_emplace, no-op if uid already present.
//                        Declaration functions use this to register
//                        op-specific defaults without stomping overrides.
//
// Seeds are independent of fills — setting a seed never creates a fill entry,
// and setting a fill never touches the seed map.
class InputFillRecipes
{
public:
    static constexpr unsigned int K_DEFAULT_GLOBAL_SEED = 42;

    // ── Write (override) — tests and metadata ───────────────────────────

    InputFillRecipes& set(int64_t uid, FillRecipe f)
    {
        _fills[uid] = f;
        return *this;
    }

    InputFillRecipes& set(flatbuffers::Optional<int64_t> uid, FillRecipe f)
    {
        if(uid.has_value())
        {
            set(*uid, f);
        }
        return *this;
    }

    InputFillRecipes& setRange(int64_t uid, float lo, float hi)
    {
        return set(uid, FillRecipe::free(lo, hi));
    }

    InputFillRecipes& setFixed(int64_t uid, float value)
    {
        return set(uid, FillRecipe::fixed(value));
    }

    // ── Write (default) — declaration functions ─────────────────────────

    InputFillRecipes& setDefault(int64_t uid, FillRecipe f)
    {
        _fills.try_emplace(uid, f);
        return *this;
    }

    InputFillRecipes& setDefault(flatbuffers::Optional<int64_t> uid, FillRecipe f)
    {
        if(uid.has_value())
        {
            setDefault(*uid, f);
        }
        return *this;
    }

    // ── Read ────────────────────────────────────────────────────────────

    const std::unordered_map<int64_t, FillRecipe>& fills() const
    {
        return _fills;
    }

    FillRecipe fill(int64_t uid) const
    {
        auto it = _fills.find(uid);
        return it != _fills.end() ? it->second : FillRecipe{};
    }

    std::vector<int64_t> unfilled(const std::vector<int64_t>& ownedUids) const
    {
        std::vector<int64_t> result;
        for(const int64_t uid : ownedUids)
        {
            const auto f = fill(uid);
            if(f.kind != FillRecipe::Kind::FREE && f.kind != FillRecipe::Kind::FIXED)
            {
                result.push_back(uid);
            }
        }
        return result;
    }

    // ── Seed config ─────────────────────────────────────────────────────

    InputFillRecipes& setSeed(int64_t uid, unsigned int value)
    {
        _seeds[uid] = value;
        return *this;
    }

    InputFillRecipes& setGlobalSeed(unsigned int s)
    {
        _globalSeed = s;
        return *this;
    }

    unsigned int globalSeed() const
    {
        return _globalSeed;
    }

    std::optional<unsigned int> resolveSeed(int64_t uid) const
    {
        if(auto it = _seeds.find(uid); it != _seeds.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    // ── JSON boundary (serialization for capture/load) ──────────────────

    nlohmann::json toJson() const
    {
        nlohmann::json inputs;
        for(const auto& [uid, fill] : _fills)
        {
            nlohmann::json j;
            j["kind"] = kindToString(fill.kind);
            if(fill.kind == FillRecipe::Kind::FREE)
            {
                j["lo"] = fill.lo;
                j["hi"] = fill.hi;
            }
            if(fill.kind == FillRecipe::Kind::FIXED)
            {
                j["value"] = fill.value;
            }
            if(auto it = _seeds.find(uid); it != _seeds.end())
            {
                j["seed"] = it->second;
            }
            inputs[std::to_string(uid)] = std::move(j);
        }
        for(const auto& [uid, seedVal] : _seeds)
        {
            auto key = std::to_string(uid);
            if(!inputs.contains(key))
            {
                inputs[key] = nlohmann::json{{"seed", seedVal}};
            }
        }
        return inputs;
    }

    void loadFromJson(const std::unordered_map<int64_t, nlohmann::json>& inputs)
    {
        for(const auto& [uid, j] : inputs)
        {
            if(j.contains("kind") && j["kind"].is_string())
            {
                FillRecipe f;
                f.kind = kindFromString(j["kind"].get<std::string>());
                if(j.contains("lo") && j["lo"].is_number())
                {
                    f.lo = j["lo"].get<float>();
                }
                if(j.contains("hi") && j["hi"].is_number())
                {
                    f.hi = j["hi"].get<float>();
                }
                if(j.contains("value") && j["value"].is_number())
                {
                    f.value = j["value"].get<float>();
                }
                set(uid, f);
            }

            if(j.contains("seed") && j["seed"].is_number_unsigned())
            {
                setSeed(uid, j["seed"].get<unsigned int>());
            }
        }
    }

private:
    std::unordered_map<int64_t, FillRecipe> _fills;
    std::unordered_map<int64_t, unsigned int> _seeds;
    unsigned int _globalSeed = K_DEFAULT_GLOBAL_SEED;

    static const char* kindToString(FillRecipe::Kind k)
    {
        switch(k)
        {
        case FillRecipe::Kind::FREE:
            return "free";
        case FillRecipe::Kind::FIXED:
            return "fixed";
        case FillRecipe::Kind::STRUCTURED:
            return "structured";
        case FillRecipe::Kind::DERIVED:
            return "derived";
        default:
            return "free";
        }
    }

    static FillRecipe::Kind kindFromString(const std::string& s)
    {
        if(s == "fixed")
        {
            return FillRecipe::Kind::FIXED;
        }
        if(s == "structured")
        {
            return FillRecipe::Kind::STRUCTURED;
        }
        if(s == "derived")
        {
            return FillRecipe::Kind::DERIVED;
        }
        return FillRecipe::Kind::FREE;
    }
};

} // namespace hipdnn_integration_tests
