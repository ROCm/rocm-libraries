// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

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
};

} // namespace hipdnn_integration_tests
