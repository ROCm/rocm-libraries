/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Replaces gtest_main so the suite can install its own console reporter; every GTest
// command-line flag still works, and the reporter can be turned off entirely.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "framework/reporter.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (std::getenv("RPP_TEST_PLAIN_OUTPUT") == nullptr) rpptest::install_concise_reporter();

    const int result = RUN_ALL_TESTS();

    // A --gtest_filter that selects nothing is a success as far as GTest is concerned. CTest
    // registers one test per suite by filter (cmake/grouped_tests.cmake), so a filter gone stale
    // -- a suite renamed against a generated test list that was not regenerated -- would report a
    // green run that executed no cases. Treat selecting nothing as a failure instead. The count
    // is only meaningful after the run: GTest applies the filter inside RUN_ALL_TESTS.
    if (result == 0 && ::testing::UnitTest::GetInstance()->test_to_run_count() == 0) {
        std::fprintf(stderr, "error: --gtest_filter=%s selected no tests\n",
                     GTEST_FLAG_GET(filter).c_str());
        return 1;
    }
    return result;
}
