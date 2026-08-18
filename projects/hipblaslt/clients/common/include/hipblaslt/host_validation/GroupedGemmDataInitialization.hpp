// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cstdint>
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <hipblaslt_datatype2string.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    template <typename InputA, typename InputB, typename Output>
    void initializeGroupedGemm(std::vector<InputA>&     a,
                               int64_t                  sizeA,
                               std::vector<InputB>&     b,
                               int64_t                  sizeB,
                               std::vector<Output>&     c,
                               int64_t                  sizeC,
                               std::vector<float>&      bias,
                               int64_t                  sizeBias,
                               hipblaslt_initialization initialization,
                               uint64_t                 seed = defaultInitializationSeed)
    {
        auto fillValues
            = [&](auto& values, int64_t size, GenerationPatternSpec pattern, uint64_t stream) {
                  if(size < 0)
                      throw std::invalid_argument("Grouped GEMM initialization size is negative.");
                  if(static_cast<size_t>(size) > values.size())
                      throw std::invalid_argument(
                          "Grouped GEMM initialization size exceeds destination storage.");
                  pattern.stream = stream;
                  initializeTensor(values.data(),
                                   Layout::contiguous(Shape{static_cast<size_t>(size)}),
                                   GenerationOptions{.seed = seed, .real = std::move(pattern)});
              };

        const auto fillAllOperands = [&](const GenerationPatternSpec& pattern) {
            fillValues(a, sizeA, pattern, 0);
            fillValues(b, sizeB, pattern, 1);
            fillValues(c, sizeC, pattern, 2);
            fillValues(bias, sizeBias, pattern, 3);
        };

        // HPL means High-Performance Linpack style: uniform values in [-0.5, 0.5].
        const GenerationPatternSpec hpl{
            .pattern    = GenerationPattern::UniformReal,
            .parameter0 = -0.5,
            .parameter1 = 0.5,
        };

        if(initialization == hipblaslt_initialization::rand_int)
        {
            const GenerationPatternSpec randomInteger{
                .pattern    = GenerationPattern::UniformInteger,
                .parameter0 = 1,
                .parameter1 = 10,
            };
            // Legacy grouped GEMM initialization alternates B's signs to limit reduction growth
            // for 16-bit inputs; A, C, and bias retain the positive integer recipe.
            GenerationPatternSpec alternatingRandom = randomInteger;
            alternatingRandom.alternatingDimensions = {0};
            fillValues(a, sizeA, randomInteger, 0);
            fillValues(b, sizeB, alternatingRandom, 1);
            fillValues(c, sizeC, randomInteger, 2);
            fillValues(bias, sizeBias, randomInteger, 3);
        }
        else if(initialization == hipblaslt_initialization::trig_float)
        {
            fillValues(a, sizeA, {.pattern = GenerationPattern::Sine}, 0);
            fillValues(b, sizeB, {.pattern = GenerationPattern::Cosine}, 1);
            fillValues(c, sizeC, {.pattern = GenerationPattern::Sine}, 2);
            fillValues(bias, sizeBias, {.pattern = GenerationPattern::Sine}, 3);
        }
        else if(initialization == hipblaslt_initialization::hpl)
        {
            fillAllOperands(hpl);
        }
        else if(initialization == hipblaslt_initialization::uniform_low_precision)
        {
            const GenerationPatternSpec uniform{
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -6.0,
                .parameter1 = 6.0,
            };
            fillAllOperands(uniform);
        }
        else if(initialization == hipblaslt_initialization::special)
        {
            // Legacy "special" uses the fixed binary16 edge-value pair from
            // hipblaslt_init_alt_impl_big/small for A and B; C and bias use HPL-style values.
            fillValues(
                a,
                sizeA,
                {.pattern = GenerationPattern::Constant, .parameter0 = specialInitializationAValue},
                0);
            fillValues(
                b,
                sizeB,
                {.pattern = GenerationPattern::Constant, .parameter0 = specialInitializationBValue},
                1);
            fillValues(c, sizeC, hpl, 2);
            fillValues(bias, sizeBias, hpl, 3);
        }
        else
        {
            fillAllOperands({});
        }
    }
} // namespace hipblaslt::host_validation
