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

#ifndef ELEMENTWISE_BINARY_OP_TEST_HELPERS_HPP
#define ELEMENTWISE_BINARY_OP_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include "hiptensor_length_generation.hpp"
#include "hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

auto inline load_config_params()
{
    hiptensor::PermutationTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::PermutationTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
        if(params)
        {
            testParams = params.value();
        }
    }
    else
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::PermutationTestParams>::loadFromFile(
            testOptions->inputFilename());
        if(params)
        {
            testParams = params.value();
        }
    }

    // testParams.printParams();
    return testParams;
}

auto inline load_config_helper()
{
    auto testParams = load_config_params();

    // Append sizes generated from lower/upper/step parameters to problemLengths
    if(!testParams.problemRanges().empty())
    {
        uint32_t rank = testParams.permutedDims()[0].size();

        for(int i = 0; i < testParams.problemRanges().size(); i++)
        {
            auto        ranges      = testParams.problemRanges()[i];
            std::size_t lower       = ranges[0];
            std::size_t upper       = ranges[1];
            std::size_t step        = ranges[2];
            std::size_t maxElements = 134217728;

            std::size_t totalSizes = 0;
            if(ranges.size() == 4)
            {
                totalSizes = ranges[3];
            }
            std::vector<std::vector<std::size_t>> generatedLengths;
            hiptensor::generate2DLengths(
                generatedLengths, lower, upper, step, rank, maxElements, totalSizes);
            testParams.problemLengths().insert(testParams.problemLengths().end(),
                                               generatedLengths.begin(),
                                               generatedLengths.end());
        }
    }
    // Append sizes generated randomly from [lower, upper] to problemLengths
    if(!testParams.problemRandRanges().empty())
    {
        uint32_t rank = testParams.permutedDims()[0].size();

        for(int i = 0; i < testParams.problemRandRanges().size(); i++)
        {
            auto        ranges      = testParams.problemRandRanges()[i];
            std::size_t lower       = ranges[0];
            std::size_t upper       = ranges[1];
            std::size_t totalSizes  = ranges[2];
            std::size_t maxElements = 134217728;

            std::vector<std::vector<std::size_t>> generatedRandLengths;
            hiptensor::generate2DLengths(
                generatedRandLengths, lower, upper, upper, rank, maxElements, totalSizes, true);
            testParams.problemLengths().insert(testParams.problemLengths().end(),
                                               generatedRandLengths.begin(),
                                               generatedRandLengths.end());
        }
    }

    // testParams.printParams();

    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.permutedDims()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.gammas()),
                              ::testing::ValuesIn(testParams.operators()));
}

#endif // ELEMENTWISE_BINARY_OP_TEST_HELPERS_HPP
