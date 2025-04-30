/* **************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <cstdio>
#include <string>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <rocsolver/rocsolver.h>

#include "common/misc/clientcommon.hpp"

static std::string rocblas_version()
{
    size_t size;
    rocblas_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocblas_get_version_string(str.data(), size);
    return str;
}

static std::string rocsolver_version()
{
    size_t size;
    rocsolver_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocsolver_get_version_string(str.data(), size);
    return str;
}

static void print_version_info()
{
    fmt::print("rocSOLVER version {} (with rocBLAS {})\n", rocsolver_version(), rocblas_version());
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    print_version_info();

    // print device info
    int device_count = query_device_property();
    if(device_count <= 0)
    {
        fmt::print(stderr, "Error: No devices found\n");
        return -1;
    }
    set_device(0); // use first device

    // Initialize gtest and rocBLAS
    ::testing::InitGoogleTest(&argc, argv);
    rocblas_initialize();

    int status = RUN_ALL_TESTS();
    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
