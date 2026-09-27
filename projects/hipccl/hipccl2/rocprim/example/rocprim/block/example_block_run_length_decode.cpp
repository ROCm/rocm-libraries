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

// Single-pass run-length decode to a global buffer (window size == total size)
__global__
void rld_kernel(int* d_out, unsigned int* d_total_size)
{
    constexpr unsigned int BLOCK_DIM_X              = 4;
    constexpr int          RUNS_PER_THREAD          = 2;
    constexpr int          DECODED_ITEMS_PER_THREAD = 2;

    using decoder_t = rocprim::
        block_run_length_decode<int, BLOCK_DIM_X, RUNS_PER_THREAD, DECODED_ITEMS_PER_THREAD>;

    __shared__ typename decoder_t::storage_type storage;

    const unsigned int tid = threadIdx.x;

    // Each thread contributes two runs (length 1 each) -> total_decoded_size = 8
    //  thread 0 -> [10, 11], thread 1 -> [20, 21], thread 2 -> [30, 31], thread 3 -> [40, 41]
    int          run_values[RUNS_PER_THREAD];
    unsigned int run_lengths[RUNS_PER_THREAD];

    run_values[0]  = 10 * (int(tid) + 1);
    run_values[1]  = 10 * (int(tid) + 1) + 1;
    run_lengths[0] = 1;
    run_lengths[1] = 1;

    // Construct decoder using run lengths; returns total decoded size across the block
    unsigned int total_decoded_size = 0;
    decoder_t    rld(storage, run_values, run_lengths, total_decoded_size);

    // Decode a single window starting at offset 0
    int          decoded[DECODED_ITEMS_PER_THREAD];
    unsigned int rel_offsets[DECODED_ITEMS_PER_THREAD];
    rld.run_length_decode(decoded, rel_offsets, /*from_decoded_offset=*/0);

    // Compute valid count for this window (here it equals window size)
    const unsigned int window_size = BLOCK_DIM_X * DECODED_ITEMS_PER_THREAD; // 8
    const unsigned int num_valid
        = total_decoded_size < window_size ? total_decoded_size : window_size;

    // Store in "blocked arrangement": flatten tid-local items to linear memory
    const unsigned int base = tid * DECODED_ITEMS_PER_THREAD;
    for(int i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
    {
        const unsigned int g = base + i;
        if(g < num_valid)
            d_out[g] = decoded[i];
    }

    if(tid == 0)
    {
        *d_total_size = total_decoded_size;
    }
}

int main()
{
    // Window = 4 threads * 2 items/thread = 8
    constexpr int BLOCK_DIM_X              = 4;
    constexpr int DECODED_ITEMS_PER_THREAD = 2;
    constexpr int WINDOW_SIZE              = BLOCK_DIM_X * DECODED_ITEMS_PER_THREAD;

    // Device buffers
    common::device_ptr<int>          d_output(WINDOW_SIZE);
    common::device_ptr<unsigned int> d_total(1u);

    hipLaunchKernelGGL(rld_kernel, dim3(1), dim3(BLOCK_DIM_X), 0, 0, d_output.get(), d_total.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto out   = d_output.load();
    const auto total = d_total.load();

    // Expected decoded stream (concatenation of all runs across threads)
    // [10, 11, 20, 21, 30, 31, 40, 41]
    std::vector<int> expected = {10, 11, 20, 21, 30, 31, 40, 41};

    bool ok_total = (total[0] == expected.size());
    bool ok_vals  = true;
    for(int i = 0; i < (int)expected.size(); i++)
    {
        ok_vals = ok_vals && (out[i] == expected[i]);
    }
    ASSERT_TRUE(ok_total && ok_vals);
    return 0;
}
