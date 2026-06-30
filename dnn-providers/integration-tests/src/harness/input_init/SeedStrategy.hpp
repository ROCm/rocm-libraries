// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <random>

namespace hipdnn_integration_tests
{

// Controls how per-tensor seeds are derived during input synthesis.
// Each call to next() produces the seed passed to fillTensorWithRandomValues.
class SeedStrategy
{
public:
    SeedStrategy() = default;
    virtual ~SeedStrategy() = default;

    SeedStrategy(const SeedStrategy&) = default;
    SeedStrategy& operator=(const SeedStrategy&) = default;
    SeedStrategy(SeedStrategy&&) = default;
    SeedStrategy& operator=(SeedStrategy&&) = default;

    virtual unsigned int next(std::mt19937& rng) = 0;
};

// Each tensor gets a different seed drawn sequentially from the shared RNG.
// Deterministic: given the same initial seed, every tensor gets the same
// derived seed on every run. This is the default strategy.
class SequentialSeed : public SeedStrategy
{
public:
    unsigned int next(std::mt19937& rng) override
    {
        return static_cast<unsigned int>(rng());
    }
};

// Every tensor gets the same fixed seed regardless of visit order.
// Reproduces the pre-synthesis legacy behavior where initializeBundle()
// passed the raw test seed to every fillTensorWithRandomValues() call.
class FixedSeed : public SeedStrategy
{
public:
    explicit FixedSeed(unsigned int seed)
        : _seed(seed)
    {
    }

    unsigned int next(std::mt19937& /*rng*/) override
    {
        return _seed;
    }

private:
    unsigned int _seed;
};

} // namespace hipdnn_integration_tests
