// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <flatbuffers/flatbuffers.h>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

// An entry of metadata `inputs` that cannot be loaded. `at` names the entry in the message
// (empty: no prefix).
[[noreturn]] inline void failFill(const std::string& at, const std::string& message)
{
    throw FillRecipeError(at.empty() ? message : at + ": " + message);
}

// The kind's name in the JSON (and in messages that name a kind).
inline const char* kindName(FillRecipe::Kind k)
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
    case FillRecipe::Kind::VALUES:
        return "values";
    default:
        break;
    }
    throw std::logic_error("InputFillRecipes: unhandled FillRecipe::Kind");
}

// The kinds a JSON entry spells out with `lo`/`hi`/`value`/`scale`/`special`/`fraction`;
// VALUES has its own parser.
inline std::optional<FillRecipe::Kind> scalarKindFromName(const std::string& s)
{
    for(const auto kind : {FillRecipe::Kind::FREE,
                           FillRecipe::Kind::FIXED,
                           FillRecipe::Kind::NONZERO,
                           FillRecipe::Kind::ZEROS,
                           FillRecipe::Kind::LARGE,
                           FillRecipe::Kind::SUBNORMAL,
                           FillRecipe::Kind::SPECIAL})
    {
        if(s == kindName(kind))
        {
            return kind;
        }
    }
    return std::nullopt;
}

inline const char* specialName(FillRecipe::SpecialValue v)
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

inline std::optional<FillRecipe::SpecialValue> specialFromName(const std::string& s)
{
    for(const auto v : {FillRecipe::SpecialValue::QUIET_NAN,
                        FillRecipe::SpecialValue::POS_INF,
                        FillRecipe::SpecialValue::NEG_INF})
    {
        if(s == specialName(v))
        {
            return v;
        }
    }
    return std::nullopt;
}

// A finite number that fits a float; anything else is the author's error.
inline float parseFloat(const nlohmann::json& v, const char* field, const std::string& at)
{
    if(!v.is_number())
    {
        failFill(at, std::string("\"") + field + "\" must be a number, got " + v.dump());
    }
    const auto d = v.get<double>();
    constexpr auto MAX_FLOAT = static_cast<double>(std::numeric_limits<float>::max());
    if(!std::isfinite(d) || std::fabs(d) > MAX_FLOAT)
    {
        failFill(at,
                 std::string("\"") + field + "\" = " + v.dump()
                     + " is not a finite float-range number");
    }
    return static_cast<float>(d);
}

// Overwrites `out` with the numeric field `field` of `j`; an absent field leaves it.
inline void readFloat(const nlohmann::json& j, const char* field, const std::string& at, float& out)
{
    if(const auto it = j.find(field); it != j.end())
    {
        out = parseFloat(*it, field, at);
    }
}

inline unsigned int parseSeed(const nlohmann::json& v, const std::string& at)
{
    if(!v.is_number_unsigned() || v.get<std::uint64_t>() > std::numeric_limits<unsigned int>::max())
    {
        failFill(at, "\"seed\" must be an unsigned 32-bit integer, got " + v.dump());
    }
    return v.get<unsigned int>();
}

// A misspelled key would otherwise load as the default and the case would run without the
// feature it asked for.
inline void rejectUnknownKeys(const nlohmann::json& spec,
                              const std::string& at,
                              const std::string& what,
                              std::initializer_list<std::string_view> allowedKeys)
{
    for(const auto& item : spec.items())
    {
        const auto& key = item.key();
        bool known = false;
        for(const auto allowed : allowedKeys)
        {
            if(key == allowed)
            {
                known = true;
                break;
            }
        }
        if(!known)
        {
            std::string message = what;
            message += R"( has unknown key ")";
            message += key;
            message += '"';
            failFill(at, message);
        }
    }
}

// An entry with a `kind` other than "values".
inline FillRecipe
    parseScalarFill(const nlohmann::json& j, const std::string& kindText, const std::string& at)
{
    const auto kind = scalarKindFromName(kindText);
    if(!kind.has_value())
    {
        failFill(at,
                 R"("kind" is an unknown fill kind ")" + kindText
                     + R"(" (expected free, fixed, nonzero, zeros, large, subnormal, special )"
                       R"(or values; the constant fill is "fixed"))");
    }

    // A space renders every leaf for every kind, so each kind accepts the whole set; only
    // the fields the kind reads are looked at, and the others are ignored whatever they hold.
    rejectUnknownKeys(j,
                      at,
                      "kind \"" + kindText + "\"",
                      {"kind", "lo", "hi", "value", "scale", "special", "fraction", "seed"});

    FillRecipe f;
    f.kind = *kind;
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
    case FillRecipe::Kind::VALUES:
    default:
        break;
    }

    if((f.kind == FillRecipe::Kind::NONZERO || f.kind == FillRecipe::Kind::SPECIAL) && f.lo > f.hi)
    {
        failFill(at,
                 "kind \"" + kindText + "\" needs lo <= hi, got lo=" + std::to_string(f.lo)
                     + " hi=" + std::to_string(f.hi));
    }

    if(f.kind == FillRecipe::Kind::LARGE)
    {
        if(!j.contains("scale"))
        {
            failFill(at, R"(kind "large" requires "scale")");
        }
        if(f.scale <= 0.0f)
        {
            failFill(at, "\"scale\" must be > 0, got " + std::to_string(f.scale));
        }
    }

    if(f.kind == FillRecipe::Kind::SPECIAL)
    {
        const auto special = j.find("special");
        if(special == j.end() || !special->is_string())
        {
            failFill(at,
                     "kind \"special\" requires \"special\" as one of \"nan\", \"inf\", "
                     "\"-inf\"");
        }
        const auto specialText = special->get<std::string>();
        const auto value = specialFromName(specialText);
        if(!value.has_value())
        {
            failFill(
                at, R"("special" must be one of "nan", "inf", "-inf", got ")" + specialText + "\"");
        }
        f.special = *value;

        if(!j.contains("fraction"))
        {
            failFill(at, R"(kind "special" requires "fraction")");
        }
        if(f.fraction < 0.0f || f.fraction > 1.0f)
        {
            failFill(at, "\"fraction\" must be in [0, 1], got " + std::to_string(f.fraction));
        }
    }
    return f;
}

} // namespace input_fill_detail

/// Parses one entry of metadata `inputs`; `at` names the entry (bundle path or case, and
/// uid) in error messages and may be empty.
///
/// Strict: throws FillRecipeError (a std::invalid_argument) for an entry that is not an
/// object, a `kind` that is not a string or not one of free, fixed, nonzero, zeros, large,
/// subnormal, special, values (the constant fill is `fixed`), a numeric field the kind reads
/// (`lo`, `hi`, `value`, `scale`, `fraction`) that is not a finite float-range number, a
/// `seed` that is not an unsigned 32-bit integer, a nonzero or special entry with lo > hi, a
/// large entry without a `scale` > 0, a special entry without a `special` in {"nan", "inf",
/// "-inf"} or without a `fraction` in [0, 1], a malformed values entry, a key outside the
/// entry's kind's schema, and an entry with neither `kind` nor `seed` (a misspelled key).
///
/// Fields of the scalar kinds that the kind does not read (a `value` next to `free`,
/// `lo`/`hi` next to `fixed` or `zeros`) are ignored, whatever they hold.
///
/// Whether a data type can hold a kind (subnormal on an integer tensor, inf on a type
/// without one) is not knowable here; fillInputs() reports it as unsupported.
inline InputFillSpec parseInputFillSpec(const nlohmann::json& spec, const std::string& at = {})
{
    using namespace input_fill_detail;

    if(!spec.is_object())
    {
        failFill(at, "the input spec must be a JSON object, got " + spec.dump());
    }

    if(!spec.contains("kind") && !spec.contains("seed"))
    {
        // A misspelled key ("Kind") would otherwise load nothing and run on defaults.
        std::string keys;
        for(const auto& item : spec.items())
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
        std::string message = R"(the input spec has neither "kind" nor "seed" (keys: )";
        message += keys;
        message += ")";
        failFill(at, message);
    }

    InputFillSpec parsed;
    if(!spec.contains("kind"))
    {
        rejectUnknownKeys(spec, at, "a fill spec without a kind", {"seed"});
        parsed.seed = parseSeed(spec["seed"], at);
        return parsed;
    }

    const auto& kindJson = spec["kind"];
    if(!kindJson.is_string())
    {
        failFill(at, "\"kind\" must be a string, got " + kindJson.dump());
    }
    if(spec.contains("seed"))
    {
        parsed.seed = parseSeed(spec["seed"], at);
    }

    const auto kindText = kindJson.get<std::string>();
    if(kindText == kindName(FillRecipe::Kind::VALUES))
    {
        try
        {
            parsed.recipe = parseValuesFillSpec(spec);
        }
        catch(const std::invalid_argument& e)
        {
            failFill(at, e.what());
        }
        return parsed;
    }

    parsed.recipe = parseScalarFill(spec, kindText, at);
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

    // The kind's name in the JSON (and in messages that name a kind).
    static const char* kindToString(FillRecipe::Kind k)
    {
        return input_fill_detail::kindName(k);
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
                j["special"] = input_fill_detail::specialName(fill.special);
                j["fraction"] = fill.fraction;
                j["lo"] = fill.lo;
                j["hi"] = fill.hi;
                break;
            case FillRecipe::Kind::ZEROS:
            case FillRecipe::Kind::SUBNORMAL:
            default:
                break;
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

    /// Loads the entries of a case's `metadata.inputs`, keyed by tensor uid. `where` names
    /// the origin (bundle path or case) in error messages, each of which also names the uid.
    ///
    /// Every entry is parsed by parseInputFillSpec(), which documents what it rejects.
    /// Nothing is loaded when any entry is rejected.
    void loadFromJson(const std::unordered_map<int64_t, nlohmann::json>& inputs,
                      const std::string& where = "metadata inputs")
    {
        std::vector<std::pair<int64_t, InputFillSpec>> parsed;
        parsed.reserve(inputs.size());
        for(const auto& [uid, j] : inputs)
        {
            parsed.emplace_back(uid, parseInputFillSpec(j, where + ", uid " + std::to_string(uid)));
        }

        for(auto& [uid, spec] : parsed)
        {
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
