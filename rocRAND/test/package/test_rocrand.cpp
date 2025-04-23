// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << error << std::endl; \
        exit(error); \
    } \
  }

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status status = condition;           \
    if(status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << status << std::endl; \
        exit(status); \
    } \
  }

int main(int argc, char *argv[])
{
    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT));

    const size_t size = 128;
    unsigned int * data = NULL;
    HIP_CHECK(hipMalloc(&data, size * sizeof(unsigned int)));
    ROCRAND_CHECK(rocrand_generate(generator, (unsigned int *) data, size));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipFree(data));

    if(ROCRAND_VERSION < 0) exit(1);

    return 0;
}
