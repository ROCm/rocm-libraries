/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP
#define HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include "hiptensor_options.hpp"
#include "hiptensor_length_generation.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

auto inline load_config_params()
{
    hiptensor::ContractionTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
        if(params)
        {
            testParams = params.value();
        }
    }
    else
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile(
            testOptions->inputFilename());
        if(params)
        {
            testParams = params.value();
        }
    }

    // testParams.printParams();
    return testParams;
}

auto inline load_combined_config_params()
{
    auto testParams = load_config_params();

    // Append sizes generated from lower/upper/step parameters to problemLengths
    if(!testParams.problemRanges().empty())
    {
        uint32_t rank = testParams.problemModes()[0][0].size() / 2;

        for (int i = 0; i < testParams.problemRanges().size(); i++)
        {
            auto ranges = testParams.problemRanges()[i];
            std::size_t lower = ranges[0];
            std::size_t upper = ranges[1];
            std::size_t step  = ranges[2];
            std::size_t maxElements = 134217728;

            std::size_t totalSizes = 0;
            if (ranges.size() == 4)
            {
                totalSizes = ranges[3];
            }
            std::vector<std::vector<std::vector<std::size_t>>> generatedLengths;
            hiptensor::generate3DLengths(generatedLengths, lower, upper, step, rank, maxElements, totalSizes);
            testParams.problemLengths().insert(testParams.problemLengths().end(),
                                            generatedLengths.begin(), generatedLengths.end());
        }
    }
    // Append sizes generated randomly from [lower, upper] to problemLengths
    if(!testParams.problemRandRanges().empty())
    {
        uint32_t rank = testParams.problemModes()[0][0].size() / 2;

        for (int i = 0; i < testParams.problemRandRanges().size(); i++)
        {
            auto ranges = testParams.problemRandRanges()[i];
            std::size_t lower = ranges[0];
            std::size_t upper = ranges[1];
            std::size_t totalSizes = ranges[2];
            std::size_t maxElements = 134217728;

            std::vector<std::vector<std::vector<std::size_t>>> generatedRandLengths;
            hiptensor::generate3DLengths(generatedRandLengths, lower, upper, upper, rank, maxElements, totalSizes, true);
            testParams.problemLengths().insert(testParams.problemLengths().end(),
                                            generatedRandLengths.begin(), generatedRandLengths.end());
        }
    }

    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::ValuesIn(testParams.algorithms()),
                              ::testing::ValuesIn(testParams.operators()),
                              ::testing::ValuesIn(testParams.workSizePrefrences()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.problemStrides()),
                              ::testing::ValuesIn(testParams.problemModes()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.betas()));
}

auto inline load_sequence_config_params()
{
    auto testParams = load_config_params();

    auto dataTypes          = testParams.dataTypes();
    auto algorithms         = testParams.algorithms();
    auto operators          = testParams.operators();
    auto workSizePrefrences = testParams.workSizePrefrences();
    auto logLevelMask       = std::vector<std::decay_t<decltype(testParams.logLevelMask())>>(
        1, testParams.logLevelMask());
    auto problemLengths = testParams.problemLengths();
    auto problemStrides = testParams.problemStrides();
    auto problemModes   = testParams.problemModes();
    auto alphas         = testParams.alphas();
    auto betas          = testParams.betas();

    std::vector<size_t> lengths   = {dataTypes.size(),
                                     algorithms.size(),
                                     operators.size(),
                                     workSizePrefrences.size(),
                                     logLevelMask.size(),
                                     problemLengths.size(),
                                     problemStrides.size(),
                                     problemModes.size(),
                                     alphas.size(),
                                     betas.size()};
    auto                maxLength = *std::max_element(lengths.begin(), lengths.end());

    dataTypes.resize(maxLength, dataTypes.back());
    algorithms.resize(maxLength, algorithms.back());
    operators.resize(maxLength, operators.back());
    workSizePrefrences.resize(maxLength, workSizePrefrences.back());
    logLevelMask.resize(maxLength, logLevelMask.back());
    problemLengths.resize(maxLength, problemLengths.back());
    problemStrides.resize(maxLength, problemStrides.back());
    problemModes.resize(maxLength, problemModes.back());
    alphas.resize(maxLength, alphas.back());
    betas.resize(maxLength, betas.back());

    using ParamsTuple = decltype(std::make_tuple(dataTypes.front(),
                                                 algorithms.front(),
                                                 operators.front(),
                                                 workSizePrefrences.front(),
                                                 logLevelMask.front(),
                                                 problemLengths.front(),
                                                 problemStrides.front(),
                                                 problemModes.front(),
                                                 alphas.front(),
                                                 betas.front()));

    std::vector<ParamsTuple> paramsSequence;
    for(int i = 0; i < maxLength; i++)
    {
        paramsSequence.push_back(std::make_tuple(dataTypes[i],
                                                 algorithms[i],
                                                 operators[i],
                                                 workSizePrefrences[i],
                                                 logLevelMask[i],
                                                 problemLengths[i],
                                                 problemStrides[i],
                                                 problemModes[i],
                                                 alphas[i],
                                                 betas[i]));
    }

    return ::testing::ValuesIn(paramsSequence);
}

#endif // HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP
