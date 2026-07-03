// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests
{

// How a single leaf input tensor should be initialized. A per-op fill function
// (see SynthesizeInputs.hpp) declares a sensible default for each input; a test
// that needs something different supplies a TensorInit for that tensor's uid.
//
//   Free       — uniform random values in [lo, hi].
//   Fixed      — every element set to `value`.
//   Structured — refuse to synthesize (data needs a specific format).
//   Derived    — refuse to synthesize (value must come from another op).
//
// seed, when set, overrides the global/automatic seed for this tensor.
// Distribution authors (adding gaussian, log-uniform, etc.) can ignore it —
// seed is resolved by the tracker, not by the fill pattern.
struct TensorInit
{
    enum class Kind
    {
        FREE,
        FIXED,
        STRUCTURED,
        DERIVED,
    };

    static constexpr float K_DEFAULT_LO = -1.0f;
    static constexpr float K_DEFAULT_HI = 1.0f;

    Kind kind = Kind::FREE;
    float lo = K_DEFAULT_LO;
    float hi = K_DEFAULT_HI;
    float value = 0.0f;
    std::optional<unsigned int> seed;

    static TensorInit free(float lo, float hi)
    {
        TensorInit t;
        t.kind = Kind::FREE;
        t.lo = lo;
        t.hi = hi;
        return t;
    }
    static TensorInit fixed(float v)
    {
        TensorInit t;
        t.kind = Kind::FIXED;
        t.value = v;
        return t;
    }
    static TensorInit structured()
    {
        TensorInit t;
        t.kind = Kind::STRUCTURED;
        return t;
    }
    static TensorInit derived()
    {
        TensorInit t;
        t.kind = Kind::DERIVED;
        return t;
    }

    static const char* kindToString(Kind k)
    {
        switch(k)
        {
        case Kind::FREE:
            return "free";
        case Kind::FIXED:
            return "fixed";
        case Kind::STRUCTURED:
            return "structured";
        case Kind::DERIVED:
            return "derived";
        default:
            return "free";
        }
    }

    static Kind kindFromString(const std::string& s)
    {
        if(s == "fixed")
            return Kind::FIXED;
        if(s == "structured")
            return Kind::STRUCTURED;
        if(s == "derived")
            return Kind::DERIVED;
        return Kind::FREE;
    }

    nlohmann::json toJson() const
    {
        nlohmann::json j;
        j["kind"] = kindToString(kind);
        if(kind == Kind::FREE)
        {
            j["lo"] = lo;
            j["hi"] = hi;
        }
        if(kind == Kind::FIXED)
        {
            j["value"] = value;
        }
        if(seed.has_value())
        {
            j["seed"] = *seed;
        }
        return j;
    }

    static TensorInit fromJson(const nlohmann::json& j)
    {
        TensorInit t;
        if(j.contains("kind") && j["kind"].is_string())
        {
            t.kind = kindFromString(j["kind"].get<std::string>());
        }
        if(j.contains("lo") && j["lo"].is_number())
        {
            t.lo = j["lo"].get<float>();
        }
        if(j.contains("hi") && j["hi"].is_number())
        {
            t.hi = j["hi"].get<float>();
        }
        if(j.contains("value") && j["value"].is_number())
        {
            t.value = j["value"].get<float>();
        }
        if(j.contains("seed") && j["seed"].is_number_unsigned())
        {
            t.seed = j["seed"].get<unsigned int>();
        }
        return t;
    }
};

} // namespace hipdnn_integration_tests
