// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <flatbuffers/flatbuffers.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "harness/input-init/FillRecipe.hpp"
#include "harness/input-init/TensorAliases.hpp"

namespace hipdnn_integration_tests
{

// One input's entry in metadata `inputs`. A spec without `kind` sets only a seed.
struct InputFillSpec
{
    std::optional<FillRecipe> recipe;
    std::optional<unsigned int> seed;
};

namespace input_fill_detail
{

inline FillRecipe parseValuesFillSpec(const nlohmann::json& spec)
{
    for(const auto& item : spec.items())
    {
        const auto& key = item.key();
        if(key != "kind" && key != "values" && key != "repeat" && key != "seed")
        {
            throw std::invalid_argument("kind values has unknown key \"" + key + "\"");
        }
    }

    if(!spec.contains("values") || !spec["values"].is_array() || spec["values"].empty())
    {
        throw std::invalid_argument("kind values needs a non-empty \"values\" array");
    }

    bool repeat = true;
    if(spec.contains("repeat"))
    {
        if(!spec["repeat"].is_boolean())
        {
            throw std::invalid_argument("kind values has a \"repeat\" that is not a boolean");
        }
        repeat = spec["repeat"].get<bool>();
    }

    const auto& numbers = spec["values"];
    bool allIntegers = true;
    for(std::size_t i = 0; i < numbers.size(); ++i)
    {
        const auto& number = numbers[i];
        const std::string where = "values[" + std::to_string(i) + "] ";
        if(number.is_number_unsigned())
        {
            if(number.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            {
                throw std::invalid_argument(where + number.dump()
                                            + " is above the largest 64-bit signed integer");
            }
        }
        else if(number.is_number_float())
        {
            if(!std::isfinite(number.get<double>()))
            {
                throw std::invalid_argument(where + "is not finite");
            }
            allIntegers = false;
        }
        else if(!number.is_number_integer())
        {
            throw std::invalid_argument(where + number.dump() + " is not a number");
        }
    }

    if(allIntegers)
    {
        return FillRecipe::values(numbers.get<std::vector<int64_t>>(), repeat);
    }
    return FillRecipe::values(numbers.get<std::vector<double>>(), repeat);
}

} // namespace input_fill_detail

// Parses one entry of metadata `inputs`. Throws std::invalid_argument for an unknown
// kind or a malformed spec.
inline InputFillSpec parseInputFillSpec(const nlohmann::json& spec)
{
    if(!spec.is_object())
    {
        throw std::invalid_argument("fill spec is not a JSON object");
    }

    InputFillSpec parsed;
    if(spec.contains("seed") && spec["seed"].is_number_unsigned())
    {
        parsed.seed = spec["seed"].get<unsigned int>();
    }

    if(!spec.contains("kind"))
    {
        tensor_alias_detail::rejectUnknownKeys(spec, "fill spec without a kind", {"seed"});
        return parsed;
    }
    if(!spec["kind"].is_string())
    {
        throw std::invalid_argument("fill kind " + spec["kind"].dump() + " is not a string");
    }

    const auto kind = spec["kind"].get<std::string>();
    if(kind == "values")
    {
        parsed.recipe = input_fill_detail::parseValuesFillSpec(spec);
        return parsed;
    }

    FillRecipe f;
    if(kind == "free")
    {
        f.kind = FillRecipe::Kind::FREE;
    }
    else if(kind == "fixed")
    {
        f.kind = FillRecipe::Kind::FIXED;
    }
    else
    {
        throw std::invalid_argument("unknown fill kind \"" + kind + "\"");
    }

    tensor_alias_detail::rejectUnknownKeys(
        spec, "kind " + kind, {"kind", "lo", "hi", "value", "seed"});

    if(spec.contains("lo") && spec["lo"].is_number())
    {
        f.lo = spec["lo"].get<float>();
    }
    if(spec.contains("hi") && spec["hi"].is_number())
    {
        f.hi = spec["hi"].get<float>();
    }
    if(spec.contains("value") && spec["value"].is_number())
    {
        f.value = spec["value"].get<float>();
    }
    parsed.recipe = std::move(f);
    return parsed;
}

// The fill recipes for a graph's input tensors: one FillRecipe per tensor,
// plus seed overrides. fillInputs() consults this to fill each input.
//
// Maps for independent concerns:
//   _fills      — how to fill each tensor (kind + params)
//   _typeFills  — data-type defaults, below everything in _fills
//   _seeds      — per-tensor seed overrides
//
// One scalar:
//   _globalSeed — seeds the RNG that generates per-tensor seeds
//
// Write modes:
//   set(uid, f)            — operator[], always overwrites. Tests and metadata
//                            use this to force a specific fill for a tensor.
//   setDefault(uid, f)     — try_emplace, no-op if uid already present.
//                            Declaration functions use this to register
//                            op-specific defaults without stomping overrides.
//   setTypeDefault(uid, f) — data-type default, consulted only when _fills
//                            has no entry. Kept in its own map so fill()
//                            resolves the same whichever is written first: an
//                            op default derived from fill(x) sees x's final
//                            range.
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
        _fills[uid] = std::move(f);
        _setUids.insert(uid);
        return *this;
    }

    InputFillRecipes& set(flatbuffers::Optional<int64_t> uid, FillRecipe f)
    {
        if(uid.has_value())
        {
            set(*uid, std::move(f));
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

    InputFillRecipes& setDefault(int64_t uid, const FillRecipe& f)
    {
        _fills.try_emplace(uid, f);
        return *this;
    }

    InputFillRecipes& setDefault(flatbuffers::Optional<int64_t> uid, const FillRecipe& f)
    {
        if(uid.has_value())
        {
            setDefault(*uid, f);
        }
        return *this;
    }

    // ── Write (data-type default) — fillInputs ──────────────────────────

    InputFillRecipes& setTypeDefault(int64_t uid, FillRecipe f)
    {
        _typeFills[uid] = std::move(f);
        return *this;
    }

    // ── Read ────────────────────────────────────────────────────────────

    // Every tensor with a recipe, each resolved as fill() would.
    std::unordered_map<int64_t, FillRecipe> fills() const
    {
        auto resolved = _fills;
        resolved.insert(_typeFills.begin(), _typeFills.end());
        return resolved;
    }

    // Whether set() wrote this uid's recipe, as opposed to a default or nothing.
    bool isSet(int64_t uid) const
    {
        return _setUids.count(uid) != 0;
    }

    FillRecipe fill(int64_t uid) const
    {
        if(auto it = _fills.find(uid); it != _fills.end())
        {
            return it->second;
        }
        auto it = _typeFills.find(uid);
        return it != _typeFills.end() ? it->second : FillRecipe{};
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
        for(const auto& [uid, fill] : fills())
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
            if(fill.kind == FillRecipe::Kind::VALUES)
            {
                j["values"]
                    = fill.list != nullptr
                          ? std::visit([](const auto& numbers) { return nlohmann::json(numbers); },
                                       *fill.list)
                          : nlohmann::json::array();
                j["repeat"] = fill.repeat;
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

    // Throws std::invalid_argument for an entry parseInputFillSpec() rejects.
    void loadFromJson(const std::unordered_map<int64_t, nlohmann::json>& inputs)
    {
        for(const auto& [uid, j] : inputs)
        {
            auto spec = parseInputFillSpec(j);
            if(spec.recipe.has_value())
            {
                set(uid, std::move(*spec.recipe));
            }
            if(spec.seed.has_value())
            {
                setSeed(uid, *spec.seed);
            }
        }
    }

private:
    std::unordered_map<int64_t, FillRecipe> _fills;
    std::unordered_map<int64_t, FillRecipe> _typeFills;
    std::unordered_map<int64_t, unsigned int> _seeds;
    std::unordered_set<int64_t> _setUids;
    unsigned int _globalSeed = K_DEFAULT_GLOBAL_SEED;

    static const char* kindToString(FillRecipe::Kind k)
    {
        switch(k)
        {
        case FillRecipe::Kind::FREE:
            return "free";
        case FillRecipe::Kind::FIXED:
            return "fixed";
        case FillRecipe::Kind::VALUES:
            return "values";
        default:
            return "free";
        }
    }
};

// The lowest metadata format_version that can carry these recipes: 2 when any tensor
// uses the VALUES kind, else 1.
inline int requiredFormatVersion(const InputFillRecipes& recipes)
{
    for(const auto& [uid, fill] : recipes.fills())
    {
        if(fill.kind == FillRecipe::Kind::VALUES)
        {
            return 2;
        }
    }
    return 1;
}

} // namespace hipdnn_integration_tests
