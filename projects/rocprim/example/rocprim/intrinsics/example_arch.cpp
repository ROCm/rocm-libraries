// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../example_utils.hpp"

// Query wavefront properties on device and write them to global memory
__global__
void query_wavefront_kernel(unsigned int* d_runtime_size,
                            unsigned int* d_min_size,
                            unsigned int* d_max_size,
                            unsigned int* d_target_size)
{
    if(threadIdx.x == 0)
    {
        // Runtime wavefront size (not constexpr)
        d_runtime_size[0] = rocprim::arch::wavefront::size();

        // Compile-time min/max wavefront size
        d_min_size[0] = rocprim::arch::wavefront::min_size();
        d_max_size[0] = rocprim::arch::wavefront::max_size();

        // Map constexpr target to a numeric value
        constexpr auto tgt = rocprim::arch::wavefront::get_target();
        if constexpr(tgt == rocprim::arch::wavefront::target::size32)
        {
            d_target_size[0] = 32u;
        }
        else if constexpr(tgt == rocprim::arch::wavefront::target::size64)
        {
            d_target_size[0] = 64u;
        }
        else
        {
            // dynamic / unknown (e.g. SPIR-V)
            d_target_size[0] = 0u;
        }
    }
}

int main()
{
    // Host input
    std::vector<unsigned int> h_zero(1, 0u);

    common::device_ptr<unsigned int> d_runtime(h_zero);
    common::device_ptr<unsigned int> d_min(h_zero);
    common::device_ptr<unsigned int> d_max(h_zero);
    common::device_ptr<unsigned int> d_target(h_zero);

    // Launch one block, only use thread 0
    hipLaunchKernelGGL(query_wavefront_kernel,
                       dim3(1),
                       dim3(64),
                       0,
                       0,
                       d_runtime.get(),
                       d_min.get(),
                       d_max.get(),
                       d_target.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Load results back to host
    auto h_runtime = d_runtime.load(); // size 1
    auto h_min     = d_min.load();
    auto h_max     = d_max.load();
    auto h_target  = d_target.load();

    // On AMD HIP device builds expecting: runtime == min == max (32 or 64)
    bool passed = true;
    passed      = passed && (h_runtime[0] == h_min[0]);
    passed      = passed && (h_runtime[0] == h_max[0]);

    // If device reported a concrete target (32 or 64), check against it
    if(h_target[0] != 0u)
    {
        passed = passed && (h_runtime[0] == h_target[0]);
    }

    ASSERT_TRUE(passed);

    return 0;
}
