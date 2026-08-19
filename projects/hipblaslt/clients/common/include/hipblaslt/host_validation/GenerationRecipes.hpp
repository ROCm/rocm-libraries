// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipblaslt/host_validation/GenerationCompatibility.hpp>
#include <roc/host_validation/generation.hpp>
#include <utility>

namespace hipblaslt::host_validation
{
    using roc::host_validation::GenerationRecipe;
    using roc::host_validation::ScalarCategory;
    using roc::host_validation::ScalarType;
    using roc::host_validation::scalarTypeInfo;

    inline constexpr double specialInitializationAValue = 65'280.0;
    inline constexpr double specialInitializationBValue = 0.0000607967376708984375;
    inline constexpr double maximumFiniteFloat16Value   = 65'504.0;
    inline constexpr double fp16AccumulatorProbeStep    = 4.0;

    enum class TrigonometricComponent
    {
        Sine,
        Cosine,
    };

    enum class ComplexGenerationPolicy
    {
        RealOnly,
        Replicated,
        Cartesian,
    };

    struct RandomIntegerRecipeConfiguration
    {
        bool                    small         = false;
        bool                    alternating   = false;
        ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::Cartesian;
        uint64_t                seed          = defaultInitializationSeed;
    };

    struct HplRecipeConfiguration
    {
        bool                    positiveOnly  = false;
        bool                    alternating   = false;
        ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::Cartesian;
        uint64_t                seed          = defaultInitializationSeed;
    };

    inline GenerationRecipe bindComponentRecipe(ScalarType                  destinationType,
                                                GenerationRecipe::Component component,
                                                ComplexGenerationPolicy     policy,
                                                uint64_t                    seed)
    {
        if(scalarTypeInfo(destinationType).category != ScalarCategory::Complex
           || policy == ComplexGenerationPolicy::RealOnly)
            return GenerationRecipe::realOnly(std::move(component), {.seed = seed});
        if(policy == ComplexGenerationPolicy::Replicated)
            return GenerationRecipe::replicated(std::move(component), {.seed = seed});
        GenerationRecipe::Component imaginary = component;
        return GenerationRecipe::cartesian(
            std::move(component), std::move(imaginary), {.seed = seed});
    }

    inline GenerationRecipe bindComponentPairRecipe(ScalarType                  destinationType,
                                                    GenerationRecipe::Component real,
                                                    GenerationRecipe::Component imaginary,
                                                    ComplexGenerationPolicy     policy,
                                                    uint64_t                    seed)
    {
        if(scalarTypeInfo(destinationType).category != ScalarCategory::Complex
           || policy == ComplexGenerationPolicy::RealOnly)
            return GenerationRecipe::realOnly(std::move(real), {.seed = seed});
        if(policy == ComplexGenerationPolicy::Replicated)
            return GenerationRecipe::replicated(std::move(real), {.seed = seed});
        return GenerationRecipe::cartesian(std::move(real), std::move(imaginary), {.seed = seed});
    }

    inline GenerationRecipe::Component trigonometricComponent(TrigonometricComponent component,
                                                              bool absolute = false)
    {
        GenerationRecipe::Component result = component == TrigonometricComponent::Sine
                                                 ? GenerationRecipe::sine()
                                                 : GenerationRecipe::cosine();
        return absolute ? result.withAbsoluteTransform() : result;
    }

    inline GenerationRecipe randomIntegerRecipe(ScalarType                       type,
                                                RandomIntegerRecipeConfiguration configuration = {})
    {
        GenerationRecipe::Component component = [&] {
            if(configuration.small)
                return GenerationRecipe::uniformInteger({.lower = 1, .upper = 10})
                    .withAffineValueMapping({.scale = 0.1});

            switch(type)
            {
            case ScalarType::Float16:
            case ScalarType::BFloat16:
                return GenerationRecipe::uniformInteger({.lower = -2, .upper = 2});
            case ScalarType::Int8:
                return GenerationRecipe::uniformInteger({.lower = 1, .upper = 3});
            case ScalarType::Float4E2M1:
                return GenerationRecipe::uniformInteger({.lower = -4, .upper = 4});
            case ScalarType::Float6E2M3:
                return GenerationRecipe::uniformInteger({.lower = -7, .upper = 7});
            case ScalarType::Float6E3M2:
                return GenerationRecipe::uniformInteger({.lower = -28, .upper = 28});
            case ScalarType::E8M0:
                return GenerationRecipe::randomEncodedExponent(
                    {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
            default:
                return GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
            }
        }();

        if(configuration.alternating)
            component
                = component.withAlternatingSign({.dimensions = {0, 1}, .negativeWhenOdd = false});
        return bindComponentRecipe(
            type, std::move(component), configuration.complexPolicy, configuration.seed);
    }

    inline GenerationRecipe legacyRandomRecipe(ScalarType type)
    {
        if(type == ScalarType::E8M0)
            return GenerationRecipe::realOnly(
                GenerationRecipe::uniformRawInteger({.lower = 1, .upper = 10}));
        return randomIntegerRecipe(type, {.complexPolicy = ComplexGenerationPolicy::RealOnly});
    }

    inline GenerationRecipe trigonometricRecipe(ScalarType              type,
                                                TrigonometricComponent  realComponent,
                                                bool                    positiveOnly = false,
                                                ComplexGenerationPolicy complexPolicy
                                                = ComplexGenerationPolicy::Cartesian)
    {
        const TrigonometricComponent imaginaryComponent
            = realComponent == TrigonometricComponent::Sine ? TrigonometricComponent::Cosine
                                                            : TrigonometricComponent::Sine;
        return bindComponentPairRecipe(type,
                                       trigonometricComponent(realComponent, positiveOnly),
                                       trigonometricComponent(imaginaryComponent, positiveOnly),
                                       complexPolicy,
                                       defaultInitializationSeed);
    }

    inline GenerationRecipe hplRecipe(ScalarType type, HplRecipeConfiguration configuration = {})
    {
        GenerationRecipe::Component component = [&] {
            if(type == ScalarType::E8M0)
                return GenerationRecipe::randomEncodedExponent(
                    {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
            if(type == ScalarType::Int8)
                return GenerationRecipe::uniformInteger(
                    {.lower = configuration.positiveOnly ? 0 : -1, .upper = 1});

            GenerationRecipe::Component uniform
                = GenerationRecipe::uniformReal({.lower = -0.5, .upper = 0.5});
            return configuration.positiveOnly || type == ScalarType::E5M3
                       ? uniform.withAbsoluteTransform()
                       : uniform;
        }();

        if(configuration.alternating)
            component
                = component.withAlternatingSign({.dimensions = {0, 1}, .negativeWhenOdd = false});
        return bindComponentRecipe(
            type, std::move(component), configuration.complexPolicy, configuration.seed);
    }

    inline GenerationRecipe lowPrecisionRecipe(ScalarType              type,
                                               ComplexGenerationPolicy complexPolicy
                                               = ComplexGenerationPolicy::RealOnly,
                                               uint64_t seed = defaultInitializationSeed)
    {
        GenerationRecipe::Component component = [&] {
            if(type == ScalarType::E8M0)
                return GenerationRecipe::randomEncodedExponent(
                    {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
            if(type == ScalarType::Int8)
                return GenerationRecipe::uniformInteger({.lower = -6, .upper = 6});

            GenerationRecipe::Component uniform
                = GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0});
            return type == ScalarType::E5M3 ? uniform.withAbsoluteTransform() : uniform;
        }();
        return bindComponentRecipe(type, std::move(component), complexPolicy, seed);
    }

    inline GenerationRecipe nanRecipe(ScalarType              type,
                                      ComplexGenerationPolicy complexPolicy
                                      = ComplexGenerationPolicy::RealOnly,
                                      uint64_t seed = defaultInitializationSeed)
    {
        if(!scalarTypeInfo(type).supportsNaN)
            return GenerationRecipe::realOnly(GenerationRecipe::randomRawBits(), {.seed = seed});
        return bindComponentRecipe(type, GenerationRecipe::typeNaN(), complexPolicy, seed);
    }

    inline GenerationRecipe normalRecipe(ScalarType              type,
                                         ComplexGenerationPolicy complexPolicy
                                         = ComplexGenerationPolicy::RealOnly,
                                         uint64_t seed = defaultInitializationSeed)
    {
        return bindComponentRecipe(type, GenerationRecipe::normal({}), complexPolicy, seed);
    }

    inline GenerationRecipe uniformZeroOneRecipe(ScalarType              type,
                                                 ComplexGenerationPolicy complexPolicy
                                                 = ComplexGenerationPolicy::RealOnly,
                                                 uint64_t seed = defaultInitializationSeed)
    {
        GenerationRecipe::Component component
            = type == ScalarType::Int8
                  ? GenerationRecipe::uniformInteger({.lower = 0, .upper = 1})
                  : GenerationRecipe::uniformReal({.lower = 0.0, .upper = 1.0});
        return bindComponentRecipe(type, std::move(component), complexPolicy, seed);
    }
} // namespace hipblaslt::host_validation
