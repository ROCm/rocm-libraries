/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>
#include <stdexcept>

#define VAL(str) #str
#define TOSTRING(str) VAL(str)

int device;

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Get device id from command line
    device = 0;

    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--device") == 0 && argc > i + 1)
        {
            device = atoi(argv[i + 1]);
        }

        if(strcmp(argv[i], "--version") == 0)
        {
            // Print version and exit, if requested
            std::cout << "rocALUTION version: " << __ROCALUTION_VER_MAJOR << "."
                      << __ROCALUTION_VER_MINOR << "." << __ROCALUTION_VER_PATCH << "-"
                      << TOSTRING(__ROCALUTION_VER_TWEAK) << std::endl;

            return 0;
        }
    }

    rocalution::set_device_rocalution(device);
    rocalution::init_rocalution();
    rocalution::info_rocalution();
    rocalution::stop_rocalution();

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    return RUN_ALL_TESTS();
}
