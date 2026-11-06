// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../../shared/CLI11.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <random>

// Seed for randomized tests; replay a failing run with --seed <value>.
size_t random_seed = std::random_device{}();

int main(int argc, char** argv)
{
    // strips --gtest_* flags from argv; prints gtest's help for -h/--help
    ::testing::InitGoogleTest(&argc, argv);

    CLI::App app{"rocFFT internal tests"};
    app.add_option(
        "--seed", random_seed, "Seed for randomized tests; replay a failing run with its value");
    CLI11_PARSE(app, argc, argv);

    // keep --gtest_list_tests output parseable for gtest_discover_tests
    if(!::testing::GTEST_FLAG(list_tests))
        std::cout << "Random seed: " << random_seed << std::endl;
    return RUN_ALL_TESTS();
}
