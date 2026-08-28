// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cstdint>
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>
#include <hipblaslt_datatype2string.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hipblaslt::host_numerics
{
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
        const auto fillValues = [&](auto&                              values,
                                    int64_t                            size,
                                    const GenerationRecipe::Component& component,
                                    initialization::OperandSequence    sequence) {
            if(size < 0)
                throw std::invalid_argument("Grouped GEMM initialization size is negative.");
            if(static_cast<size_t>(size) > values.size())
                throw std::invalid_argument(
                    "Grouped GEMM initialization size exceeds destination storage.");
            const uint64_t recipeSeed
                = initialization::seedForSequence(seed, static_cast<uint64_t>(sequence));
            initializeTensor(
                values.data(),
                Layout::contiguousLastDimensionFastest(Shape{static_cast<size_t>(size)}),
                GenerationRecipe::realOnly(component, {.seed = recipeSeed}));
        };

        const auto fillAllOperands = [&](const GenerationRecipe::Component& component) {
            fillValues(a, sizeA, component, initialization::OperandSequence::MatrixA);
            fillValues(b, sizeB, component, initialization::OperandSequence::MatrixB);
            fillValues(c, sizeC, component, initialization::OperandSequence::MatrixC);
            fillValues(bias, sizeBias, component, initialization::OperandSequence::Bias);
        };

        // HPL means High-Performance Linpack style: uniform values in [-0.5, 0.5].
        const GenerationRecipe::Component hpl
            = GenerationRecipe::uniformReal({.lower = -0.5, .upper = 0.5});

        if(initialization == hipblaslt_initialization::rand_int)
        {
            const GenerationRecipe::Component randomInteger
                = GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
            // Grouped GEMM alternates B's signs to limit reduction growth for
            // 16-bit inputs; A, C, and bias retain positive integers.
            const GenerationRecipe::Component alternatingRandom
                = randomInteger.withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = false});
            fillValues(a, sizeA, randomInteger, initialization::OperandSequence::MatrixA);
            fillValues(b, sizeB, alternatingRandom, initialization::OperandSequence::MatrixB);
            fillValues(c, sizeC, randomInteger, initialization::OperandSequence::MatrixC);
            fillValues(bias, sizeBias, randomInteger, initialization::OperandSequence::Bias);
        }
        else if(initialization == hipblaslt_initialization::trig_float)
        {
            const GenerationRecipe::Component sine   = GenerationRecipe::sine();
            const GenerationRecipe::Component cosine = GenerationRecipe::cosine();
            fillValues(a, sizeA, sine, initialization::OperandSequence::MatrixA);
            fillValues(b, sizeB, cosine, initialization::OperandSequence::MatrixB);
            fillValues(c, sizeC, sine, initialization::OperandSequence::MatrixC);
            fillValues(bias, sizeBias, sine, initialization::OperandSequence::Bias);
        }
        else if(initialization == hipblaslt_initialization::hpl)
        {
            fillAllOperands(hpl);
        }
        else if(initialization == hipblaslt_initialization::uniform_low_precision)
        {
            const GenerationRecipe::Component uniform
                = GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0});
            fillAllOperands(uniform);
        }
        else if(initialization == hipblaslt_initialization::special)
        {
            // "special" uses the fixed binary16 edge-value pair from
            // hipblaslt_init_alt_impl_big/small for A and B; C and bias use
            // HPL-style values.
            fillValues(a,
                       sizeA,
                       GenerationRecipe::constant({.value = specialInitializationAValue}),
                       initialization::OperandSequence::MatrixA);
            fillValues(b,
                       sizeB,
                       GenerationRecipe::constant({.value = specialInitializationBValue}),
                       initialization::OperandSequence::MatrixB);
            fillValues(c, sizeC, hpl, initialization::OperandSequence::MatrixC);
            fillValues(bias, sizeBias, hpl, initialization::OperandSequence::Bias);
        }
        else if(initialization == hipblaslt_initialization::zero)
        {
            fillAllOperands(GenerationRecipe::zero());
        }
        else
        {
            throw std::invalid_argument(
                "Grouped GEMM does not support the requested initialization mode.");
        }
    }
} // namespace hipblaslt::host_numerics
