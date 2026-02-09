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

#include "common.hpp"
#include "rocwmma_options.hpp"
#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    using Options        = rocwmma::RocwmmaOptions;
    auto& loggingOptions = Options::instance();
    loggingOptions->parseOptions(argc, argv);

    if(loggingOptions->emulationOption() == rocwmma::EmulationOption::SMOKE)
    {
        ::testing::GTEST_FLAG(filter) = "*Emulation*Smoke*";
    }
    else if(loggingOptions->emulationOption() == rocwmma::EmulationOption::REGRESSION)
    {
        ::testing::GTEST_FLAG(filter) = "*Emulation*Regression*";
    }
    else if(loggingOptions->emulationOption() == rocwmma::EmulationOption::EXTENDED)
    {
        // Run regression tests when param is 'extended' since there is no specific
        // tests in 'extended' group
        ::testing::GTEST_FLAG(filter) = "*Emulation*Regression*";
    }
    else
    {
        ::testing::GTEST_FLAG(filter) = "-*Emulation*";
    }

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Run the tests
    int status = RUN_ALL_TESTS();

    return status;
}
