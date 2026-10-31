/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

#include <cstdlib>
#include <cstring>

class RocalutionEnvironment : public ::testing::Environment
{
public:
    explicit RocalutionEnvironment(int device)
        : device_(device)
    {
    }

    void SetUp() override
    {
        rocalution::set_device_rocalution(device_);
        rocalution::init_rocalution();
    }

    void TearDown() override
    {
        rocalution::stop_rocalution();
    }

private:
    int device_;
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Death tests re-execute the binary instead of forking the initialized process
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    int device = 0;
    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
        {
            device = std::atoi(argv[i + 1]);
        }
    }

    ::testing::AddGlobalTestEnvironment(new RocalutionEnvironment(device));

    return RUN_ALL_TESTS();
}
