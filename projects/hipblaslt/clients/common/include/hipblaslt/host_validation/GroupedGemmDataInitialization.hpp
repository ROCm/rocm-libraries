// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cstdint>
#include <hipblaslt_datatype2string.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <roc/host_validation/generation.hpp>
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
                               hipblaslt_initialization initialization)
    {
        auto fillValues
            = [&](auto& values, int64_t size, GenerationPatternSpec pattern, uint64_t stream) {
                  if(size < 0)
                      throw std::invalid_argument("Grouped GEMM initialization size is negative.");
                  if(static_cast<size_t>(size) > values.size())
                      throw std::invalid_argument(
                          "Grouped GEMM initialization size exceeds destination storage.");
                  pattern.stream = stream;
                  GenerationOptions options;
                  options.seed = 69069;
                  options.real = std::move(pattern);
                  generate(mutableTensorView(values.data(),
                                             values.size(),
                                             Layout::contiguous(Shape{static_cast<size_t>(size)})),
                           options);
              };

        if(initialization == hipblaslt_initialization::rand_int)
        {
            const GenerationPatternSpec randomInteger{
                .pattern    = GenerationPattern::UniformInteger,
                .parameter0 = 1,
                .parameter1 = 10,
            };
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
            const GenerationPatternSpec uniform{
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -0.5,
                .parameter1 = 0.5,
            };
            fillValues(a, sizeA, uniform, 0);
            fillValues(b, sizeB, uniform, 1);
            fillValues(c, sizeC, uniform, 2);
            fillValues(bias, sizeBias, uniform, 3);
        }
        else if(initialization == hipblaslt_initialization::uniform_low_precision)
        {
            const GenerationPatternSpec uniform{
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -6.0,
                .parameter1 = 6.0,
            };
            fillValues(a, sizeA, uniform, 0);
            fillValues(b, sizeB, uniform, 1);
            fillValues(c, sizeC, uniform, 2);
            fillValues(bias, sizeBias, uniform, 3);
        }
        else if(initialization == hipblaslt_initialization::special)
        {
            fillValues(
                a, sizeA, {.pattern = GenerationPattern::Constant, .parameter0 = 65280.0}, 0);
            fillValues(
                b,
                sizeB,
                {.pattern = GenerationPattern::Constant, .parameter0 = 0.0000607967376708984375},
                1);
            const GenerationPatternSpec uniform{
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -0.5,
                .parameter1 = 0.5,
            };
            fillValues(c, sizeC, uniform, 2);
            fillValues(bias, sizeBias, uniform, 3);
        }
        else
        {
            fillValues(a, sizeA, {}, 0);
            fillValues(b, sizeB, {}, 1);
            fillValues(c, sizeC, {}, 2);
            fillValues(bias, sizeBias, {}, 3);
        }
    }
} // namespace hipblaslt::host_validation
