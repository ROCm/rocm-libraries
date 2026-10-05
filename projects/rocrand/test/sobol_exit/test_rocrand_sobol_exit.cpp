// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// The Sobol direction-vector tables live in a function-local static and are
// released from its destructor at process exit. hipDeviceReset() frees that
// memory first, so the destructor sees hipErrorInvalidValue. This program is
// its own process, with no test framework, so returning from main runs that
// destructor. A GoogleTest death test would _exit() the child and skip it.

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <iostream>
#include <string>

namespace
{

int fail(const char* what, int code)
{
    std::cerr << what << " failed: " << code << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " <sobol32|scrambled_sobol32> <reset|noreset>\n";
        return 2;
    }

    const std::string generator_name(argv[1]);
    const std::string mode(argv[2]);

    rocrand_rng_type generator_type;
    if(generator_name == "sobol32")
    {
        generator_type = ROCRAND_RNG_QUASI_SOBOL32;
    }
    else if(generator_name == "scrambled_sobol32")
    {
        generator_type = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    }
    else
    {
        std::cerr << "unknown generator: " << generator_name << '\n';
        return 2;
    }

    const bool reset_device = mode == "reset";
    if(!reset_device && mode != "noreset")
    {
        std::cerr << "unknown mode: " << mode << '\n';
        return 2;
    }

    rocrand_generator generator{};
    const rocrand_status create_status = rocrand_create_generator(&generator, generator_type);
    if(create_status != ROCRAND_STATUS_SUCCESS)
    {
        return fail("rocrand_create_generator", static_cast<int>(create_status));
    }

    unsigned int* data = nullptr;
    const hipError_t malloc_status = hipMalloc(&data, 256 * sizeof(unsigned int));
    if(malloc_status != hipSuccess)
    {
        return fail("hipMalloc", static_cast<int>(malloc_status));
    }

    const rocrand_status generate_status = rocrand_generate(generator, data, 256);
    if(generate_status != ROCRAND_STATUS_SUCCESS)
    {
        return fail("rocrand_generate", static_cast<int>(generate_status));
    }

    const rocrand_status destroy_status = rocrand_destroy_generator(generator);
    if(destroy_status != ROCRAND_STATUS_SUCCESS)
    {
        return fail("rocrand_destroy_generator", static_cast<int>(destroy_status));
    }

    const hipError_t free_status = hipFree(data);
    if(free_status != hipSuccess)
    {
        return fail("hipFree", static_cast<int>(free_status));
    }

    if(reset_device)
    {
        const hipError_t reset_status = hipDeviceReset();
        if(reset_status != hipSuccess)
        {
            return fail("hipDeviceReset", static_cast<int>(reset_status));
        }
    }

    return 0;
}
