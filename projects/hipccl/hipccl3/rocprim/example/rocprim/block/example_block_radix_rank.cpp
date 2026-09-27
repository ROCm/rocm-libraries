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

// Block config
constexpr unsigned int BlockSize      = 128;
constexpr unsigned int ItemsPerThread = 3;
constexpr unsigned int total_items    = BlockSize * ItemsPerThread;

// Bits config
constexpr unsigned int begin_bit = 10;
constexpr unsigned int pass_bits = 4; // examine 4 bits starting at bit 10

// Kernel: rank per-block using block_radix_rank (ascending)
__global__
void rank_kernel(const float* d_in, unsigned int* d_ranks)
{
    // Specialize: BlockSize=128, RadixBits=4
    using block_rank_t = rocprim::block_radix_rank<BlockSize, pass_bits>;
    __shared__ typename block_rank_t::storage_type storage;

    const unsigned int tid  = threadIdx.x;
    const unsigned int base = tid * ItemsPerThread;

    // Load 3 items/thread in blocked layout
    float        keys[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
#pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        keys[i] = d_in[base + i];
    }

    // Rank (ascending) on bits [begin_bit, begin_bit + pass_bits)
    block_rank_t().rank_keys(keys, ranks, storage, begin_bit, pass_bits);

// Write ranks back in blocked layout
#pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        d_ranks[base + i] = ranks[i];
    }
}

int main()
{
    // Host input: easy to generate, 0..total_items-1 as float ----
    std::vector<float> h_in(total_items);
    for(unsigned i = 0; i < total_items; ++i)
    {
        h_in[i] = static_cast<float>(i);
    }

    common::device_ptr<float>        d_in(h_in);
    common::device_ptr<unsigned int> d_ranks(total_items);

    // Launch one block of 128 threads
    hipLaunchKernelGGL(rank_kernel, dim3(1), dim3(BlockSize), 0, 0, d_in.get(), d_ranks.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out_ranks = d_ranks.load();

    // Build expected ranks on host (stable counting by extracted digit)
    // extract digit from float's bit pattern: ((bits >> begin_bit) & ((1<<pass_bits)-1))
    std::vector<unsigned> digits(total_items);
    std::vector<unsigned> counts(1u << pass_bits, 0u);

    for(unsigned i = 0; i < total_items; ++i)
    {
        // Reinterpret float bits
        uint32_t bits;
        std::memcpy(&bits, &h_in[i], sizeof(bits));
        unsigned d = (bits >> begin_bit) & ((1u << pass_bits) - 1u);
        digits[i]  = d;
        counts[d]++;
    }
    // Prefix sum to get starting offsets (ascending)
    std::vector<unsigned> offsets(counts.size(), 0u);
    unsigned              acc = 0;
    for(size_t d = 0; d < counts.size(); ++d)
    {
        offsets[d] = acc;
        acc += counts[d];
    }

    // Stable rank: rank[i] = offsets[digits[i]]++
    std::vector<unsigned> expected(total_items);
    for(unsigned i = 0; i < total_items; ++i)
    {
        expected[i] = offsets[digits[i]]++;
    }

    bool passed = true;
    for(unsigned i = 0; i < total_items; ++i)
    {
        passed = passed && (h_out_ranks[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
