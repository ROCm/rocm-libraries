// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace hipdnn_integration_tests
{

// How a single leaf input tensor should be initialized, overriding the op's
// default. A per-op fill function (see SynthesizeInputs.hpp) declares a sensible
// default range for each input; a test that needs a different distribution for a
// specific tensor supplies a TensorInit for that tensor's uid instead of
// reimplementing initialization wholesale.
//
//   Free       — uniform random values in [lo, hi]. The everyday case; lo/hi
//                replace the op default's range.
//   Fixed      — every element set to `value`. For inputs a test needs pinned
//                (e.g. a constant scale) rather than randomized.
//   Structured — refuse to synthesize (data needs a specific format that random
//                values cannot satisfy). Routed to SynthesisTracker::markStructured.
//   Derived    — refuse to synthesize (value must come from another op's output).
//                Routed to SynthesisTracker::markDerived.
//
// Structured/Derived are available here for completeness so a spec can express
// the full role taxonomy, but the common use is Free with a custom range.
struct TensorInit
{
    enum class Kind
    {
        FREE,
        FIXED,
        STRUCTURED,
        DERIVED,
    };

    Kind kind = Kind::FREE;
    float lo = -1.0f; // FREE: inclusive lower bound
    float hi = 1.0f; // FREE: inclusive upper bound
    float value = 0.0f; // FIXED: the value every element takes

    static TensorInit free(float lo, float hi)
    {
        return {Kind::FREE, lo, hi, 0.0f};
    }
    static TensorInit fixed(float value)
    {
        return {Kind::FIXED, 0.0f, 0.0f, value};
    }
    static TensorInit structured()
    {
        return {Kind::STRUCTURED, 0.0f, 0.0f, 0.0f};
    }
    static TensorInit derived()
    {
        return {Kind::DERIVED, 0.0f, 0.0f, 0.0f};
    }
};

// A per-test initialization policy consulted by SynthesisTracker before it
// applies an op's default for a given leaf input. overrideFor(uid) returns the
// custom TensorInit for that tensor, or nullopt to use the op default.
//
// This is the extension point for "different kinds of initialization": the
// default (DefaultInitSpec) overrides nothing, so synthesis behaves exactly as
// the per-op fill functions specify; ExplicitInitSpec maps specific uids to
// custom inits; future kinds (golden replay, fuzz, fixed-from-file) subclass
// this without touching the 19 per-op fill functions.
class InputInitSpec
{
public:
    InputInitSpec() = default;
    virtual ~InputInitSpec() = default;

    InputInitSpec(const InputInitSpec&) = default;
    InputInitSpec& operator=(const InputInitSpec&) = default;
    InputInitSpec(InputInitSpec&&) = default;
    InputInitSpec& operator=(InputInitSpec&&) = default;

    virtual std::optional<TensorInit> overrideFor(int64_t /*uid*/) const
    {
        return std::nullopt;
    }
};

// The no-override policy: every tensor uses its op default. This is the implicit
// behavior when no spec is supplied, named here so call sites can be explicit.
class DefaultInitSpec : public InputInitSpec
{
};

// A spec backed by an explicit uid -> TensorInit map. A test populates it with
// only the tensors that need a non-default init; all other inputs fall through
// to the op default. Replaces a hand-written initializeBundle() override.
class ExplicitInitSpec : public InputInitSpec
{
public:
    ExplicitInitSpec& set(int64_t uid, TensorInit init)
    {
        _overrides[uid] = init;
        return *this;
    }

    std::optional<TensorInit> overrideFor(int64_t uid) const override
    {
        const auto it = _overrides.find(uid);
        if(it == _overrides.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

private:
    std::unordered_map<int64_t, TensorInit> _overrides;
};

} // namespace hipdnn_integration_tests
