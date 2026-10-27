// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <flatbuffers/flatbuffers.h>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "harness/input-init/FillRecipe.hpp"

namespace hipdnn_integration_tests
{

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
//
// JSON, one object per tensor uid in a case's `metadata.inputs` (see toJson/loadFromJson):
//   {"kind": "free",      "lo": -1, "hi": 1}
//   {"kind": "fixed",     "value": 0.5}
//   {"kind": "nonzero",   "lo": -1, "hi": 1}
//   {"kind": "zeros"}
//   {"kind": "large",     "scale": 100}
//   {"kind": "subnormal"}
//   {"kind": "special",   "special": "nan" | "inf" | "-inf", "fraction": 0.25,
//                         "lo": -1, "hi": 1}
// plus an optional "seed" (unsigned 32-bit). Absent `lo`/`hi`/`value` take the FillRecipe
// defaults; `scale` (large) and `special`/`fraction` (special) are required. Parsing is
// strict: see loadFromJson.
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

    InputFillRecipes& setNonzero(int64_t uid, float lo, float hi)
    {
        return set(uid, FillRecipe::nonzero(lo, hi));
    }

    InputFillRecipes& setZeros(int64_t uid)
    {
        return set(uid, FillRecipe::zeros());
    }

    InputFillRecipes& setLarge(int64_t uid, float scale)
    {
        return set(uid, FillRecipe::large(scale));
    }

    InputFillRecipes& setSubnormal(int64_t uid)
    {
        return set(uid, FillRecipe::subnormal());
    }

    InputFillRecipes& setNonFinite(int64_t uid,
                                   FillRecipe::SpecialValue special,
                                   float fraction,
                                   float lo = FillRecipe::K_DEFAULT_LO,
                                   float hi = FillRecipe::K_DEFAULT_HI)
    {
        return set(uid, FillRecipe::nonFinite(special, fraction, lo, hi));
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

    // ── Write (data-type default) — fillInputs ──────────────────────────

    InputFillRecipes& setTypeDefault(int64_t uid, FillRecipe f)
    {
        _typeFills[uid] = f;
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

    // The kind's name in the JSON (and in messages that name a kind).
    static const char* kindToString(FillRecipe::Kind k)
    {
        switch(k)
        {
        case FillRecipe::Kind::FREE:
            return "free";
        case FillRecipe::Kind::FIXED:
            return "fixed";
        case FillRecipe::Kind::NONZERO:
            return "nonzero";
        case FillRecipe::Kind::ZEROS:
            return "zeros";
        case FillRecipe::Kind::LARGE:
            return "large";
        case FillRecipe::Kind::SUBNORMAL:
            return "subnormal";
        case FillRecipe::Kind::SPECIAL:
            return "special";
        default:
            break;
        }
        throw std::logic_error("InputFillRecipes: unhandled FillRecipe::Kind");
    }

    // ── JSON boundary (serialization for capture/load) ──────────────────

    nlohmann::json toJson() const
    {
        nlohmann::json inputs;
        for(const auto& [uid, fill] : fills())
        {
            nlohmann::json j;
            j["kind"] = kindToString(fill.kind);
            switch(fill.kind)
            {
            case FillRecipe::Kind::FREE:
            case FillRecipe::Kind::NONZERO:
                j["lo"] = fill.lo;
                j["hi"] = fill.hi;
                break;
            case FillRecipe::Kind::FIXED:
                j["value"] = fill.value;
                break;
            case FillRecipe::Kind::LARGE:
                j["scale"] = fill.scale;
                break;
            case FillRecipe::Kind::SPECIAL:
                j["special"] = specialToString(fill.special);
                j["fraction"] = fill.fraction;
                j["lo"] = fill.lo;
                j["hi"] = fill.hi;
                break;
            case FillRecipe::Kind::ZEROS:
            case FillRecipe::Kind::SUBNORMAL:
            default:
                break;
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

    /// Loads the entries of a case's `metadata.inputs`, keyed by tensor uid. `where` names
    /// the origin (bundle path or case) in error messages.
    ///
    /// Strict: throws FillRecipeError, naming the uid and the field, for an entry that
    /// is not an object, a `kind` that is not a string or not one of free, fixed, nonzero,
    /// zeros, large, subnormal, special (the constant fill is `fixed`), a numeric field the
    /// kind reads (`lo`, `hi`, `value`, `scale`, `fraction`) that is not a finite float-range
    /// number,
    /// a `seed` that is not an unsigned 32-bit integer, a nonzero or special entry with
    /// lo > hi, a large entry without a `scale` > 0, and a special entry without a
    /// `special` in {"nan", "inf", "-inf"} or without a `fraction` in [0, 1], and an entry
    /// with neither `kind` nor `seed` (a misspelled key). Nothing is loaded when any entry is
    /// rejected.
    ///
    /// Fields the kind does not read (a `value` next to `free`, `lo`/`hi` next to `fixed` or
    /// `zeros`) are ignored, whatever they hold.
    ///
    /// Whether a data type can hold a kind (subnormal on an integer tensor, inf on a type
    /// without one) is not knowable here; fillInputs() reports it as unsupported.
    void loadFromJson(const std::unordered_map<int64_t, nlohmann::json>& inputs,
                      const std::string& where = "metadata inputs")
    {
        std::vector<std::pair<int64_t, FillRecipe>> parsedFills;
        std::vector<std::pair<int64_t, unsigned int>> parsedSeeds;
        for(const auto& [uid, j] : inputs)
        {
            const std::string at = where + ", uid " + std::to_string(uid);
            if(!j.is_object())
            {
                throw FillRecipeError(at + ": the input spec must be a JSON object, got "
                                      + j.dump());
            }
            if(!j.contains("kind") && !j.contains("seed"))
            {
                // A misspelled key ("Kind") would otherwise load nothing and run on defaults.
                std::string keys;
                for(const auto& item : j.items())
                {
                    if(!keys.empty())
                    {
                        keys += ", ";
                    }
                    keys += '"';
                    keys += item.key();
                    keys += '"';
                }
                if(keys.empty())
                {
                    keys = "none";
                }
                std::string message = at;
                message += R"(: the input spec has neither "kind" nor "seed" (keys: )";
                message += keys;
                message += ")";
                throw FillRecipeError(message);
            }
            if(j.contains("kind"))
            {
                parsedFills.emplace_back(uid, parseFill(j, at));
            }
            if(j.contains("seed"))
            {
                parsedSeeds.emplace_back(uid, parseSeed(j["seed"], at));
            }
        }

        for(const auto& [uid, f] : parsedFills)
        {
            set(uid, f);
        }
        for(const auto& [uid, seed] : parsedSeeds)
        {
            setSeed(uid, seed);
        }
    }

private:
    std::unordered_map<int64_t, FillRecipe> _fills;
    std::unordered_map<int64_t, FillRecipe> _typeFills;
    std::unordered_map<int64_t, unsigned int> _seeds;
    unsigned int _globalSeed = K_DEFAULT_GLOBAL_SEED;

    static std::optional<FillRecipe::Kind> kindFromString(const std::string& s)
    {
        for(const auto kind : {FillRecipe::Kind::FREE,
                               FillRecipe::Kind::FIXED,
                               FillRecipe::Kind::NONZERO,
                               FillRecipe::Kind::ZEROS,
                               FillRecipe::Kind::LARGE,
                               FillRecipe::Kind::SUBNORMAL,
                               FillRecipe::Kind::SPECIAL})
        {
            if(s == kindToString(kind))
            {
                return kind;
            }
        }
        return std::nullopt;
    }

    static const char* specialToString(FillRecipe::SpecialValue v)
    {
        switch(v)
        {
        case FillRecipe::SpecialValue::QUIET_NAN:
            return "nan";
        case FillRecipe::SpecialValue::POS_INF:
            return "inf";
        case FillRecipe::SpecialValue::NEG_INF:
            return "-inf";
        default:
            break;
        }
        throw std::logic_error("InputFillRecipes: unhandled FillRecipe::SpecialValue");
    }

    static std::optional<FillRecipe::SpecialValue> specialFromString(const std::string& s)
    {
        for(const auto v : {FillRecipe::SpecialValue::QUIET_NAN,
                            FillRecipe::SpecialValue::POS_INF,
                            FillRecipe::SpecialValue::NEG_INF})
        {
            if(s == specialToString(v))
            {
                return v;
            }
        }
        return std::nullopt;
    }

    // A finite number that fits a float; anything else is the author's error.
    static float parseFloat(const nlohmann::json& v, const char* field, const std::string& at)
    {
        if(!v.is_number())
        {
            throw FillRecipeError(at + ": \"" + field + "\" must be a number, got " + v.dump());
        }
        const auto d = v.get<double>();
        constexpr auto MAX_FLOAT = static_cast<double>(std::numeric_limits<float>::max());
        if(!std::isfinite(d) || std::fabs(d) > MAX_FLOAT)
        {
            throw FillRecipeError(at + ": \"" + field + "\" = " + v.dump()
                                  + " is not a finite float-range number");
        }
        return static_cast<float>(d);
    }

    // Overwrites `out` with the numeric field `field` of `j`; an absent field leaves it.
    static void
        readFloat(const nlohmann::json& j, const char* field, const std::string& at, float& out)
    {
        if(const auto it = j.find(field); it != j.end())
        {
            out = parseFloat(*it, field, at);
        }
    }

    static unsigned int parseSeed(const nlohmann::json& v, const std::string& at)
    {
        if(!v.is_number_unsigned()
           || v.get<std::uint64_t>() > std::numeric_limits<unsigned int>::max())
        {
            throw FillRecipeError(at + ": \"seed\" must be an unsigned 32-bit integer, got "
                                  + v.dump());
        }
        return v.get<unsigned int>();
    }

    static FillRecipe parseFill(const nlohmann::json& j, const std::string& at)
    {
        const auto& kindJson = j.at("kind");
        if(!kindJson.is_string())
        {
            throw FillRecipeError(at + ": \"kind\" must be a string, got " + kindJson.dump());
        }
        const auto kindName = kindJson.get<std::string>();
        const auto kind = kindFromString(kindName);
        if(!kind.has_value())
        {
            throw FillRecipeError(
                at + R"(: "kind" is an unknown fill kind ")" + kindName
                + R"(" (expected free, fixed, nonzero, zeros, large, subnormal or special; )"
                  R"(the constant fill is "fixed"))");
        }

        FillRecipe f;
        f.kind = *kind;
        // Only the fields the kind reads are looked at; any others on the entry are ignored
        // whatever their value, since a space renders every leaf for every kind.
        switch(f.kind)
        {
        case FillRecipe::Kind::FREE:
        case FillRecipe::Kind::NONZERO:
            readFloat(j, "lo", at, f.lo);
            readFloat(j, "hi", at, f.hi);
            break;
        case FillRecipe::Kind::FIXED:
            readFloat(j, "value", at, f.value);
            break;
        case FillRecipe::Kind::LARGE:
            readFloat(j, "scale", at, f.scale);
            break;
        case FillRecipe::Kind::SPECIAL:
            readFloat(j, "lo", at, f.lo);
            readFloat(j, "hi", at, f.hi);
            readFloat(j, "fraction", at, f.fraction);
            break;
        case FillRecipe::Kind::ZEROS:
        case FillRecipe::Kind::SUBNORMAL:
        default:
            break;
        }

        if((f.kind == FillRecipe::Kind::NONZERO || f.kind == FillRecipe::Kind::SPECIAL)
           && f.lo > f.hi)
        {
            throw FillRecipeError(at + ": kind \"" + kindName + "\" needs lo <= hi, got lo="
                                  + std::to_string(f.lo) + " hi=" + std::to_string(f.hi));
        }

        if(f.kind == FillRecipe::Kind::LARGE)
        {
            if(!j.contains("scale"))
            {
                throw FillRecipeError(at + R"(: kind "large" requires "scale")");
            }
            if(f.scale <= 0.0f)
            {
                throw FillRecipeError(at + ": \"scale\" must be > 0, got "
                                      + std::to_string(f.scale));
            }
        }

        if(f.kind == FillRecipe::Kind::SPECIAL)
        {
            const auto special = j.find("special");
            if(special == j.end() || !special->is_string())
            {
                throw FillRecipeError(
                    at
                    + ": kind \"special\" requires \"special\" as one of \"nan\", \"inf\", "
                      "\"-inf\"");
            }
            const auto specialName = special->get<std::string>();
            const auto value = specialFromString(specialName);
            if(!value.has_value())
            {
                throw FillRecipeError(at
                                      + ": \"special\" must be one of \"nan\", \"inf\", "
                                        "\"-inf\", got \""
                                      + specialName + "\"");
            }
            f.special = *value;

            if(!j.contains("fraction"))
            {
                throw FillRecipeError(at + R"(: kind "special" requires "fraction")");
            }
            if(f.fraction < 0.0f || f.fraction > 1.0f)
            {
                throw FillRecipeError(at + ": \"fraction\" must be in [0, 1], got "
                                      + std::to_string(f.fraction));
            }
        }
        return f;
    }
};

} // namespace hipdnn_integration_tests
