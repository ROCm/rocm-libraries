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
#ifndef ROCPRIM_WITH_TOPK
    #define ROCPRIM_WITH_TOPK
#endif
#include "../../example_utils.hpp"

void example_topk_keys()
{
    // Number of elements to be selected from the input.
    size_t k = 4;

    // Prepare input and output
    common::device_ptr<int> input(std::vector<int>{5, -3, 1, 7, 8, 2, 4, 6});
    common::device_ptr<int> output(k);

    // Get required size of the temporary storage
    size_t temp_storage_size;
    HIP_CHECK(
        rocprim::topk(nullptr, temp_storage_size, input.get(), output.get(), input.size(), k));

    // Allocate temporary storage
    common::device_ptr<void> temp_storage(temp_storage_size);

    // Perform topk
    HIP_CHECK(rocprim::topk(temp_storage.get(),
                            temp_storage_size,
                            input.get(),
                            output.get(),
                            input.size(),
                            k));

    // Check for any errors
    HIP_CHECK(hipGetLastError());

    // Wait for the algorithm to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    auto                   result   = output.load();
    const std::vector<int> expected = {-3, 1, 2, 4};

    // The output can be in any order in this case, so we sort it
    std::sort(result.begin(), result.end(), std::less<int>{});

    // Check that the result is correct
    ASSERT_TRUE(result == expected);
}

void example_topk_pairs()
{
    // Number of elements to be selected from the input.
    size_t k = 4;

    // Prepare input and output
    common::device_ptr<int> input_keys(std::vector<int>{5, -3, 1, 7, 8, 2, 4, 6});
    common::device_ptr<int> output_keys(k);
    common::device_ptr<int> input_vals(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    common::device_ptr<int> output_vals(k);

    // Get required size of the temporary storage
    size_t temp_storage_size;
    HIP_CHECK(rocprim::topk_pairs(nullptr,
                                  temp_storage_size,
                                  input_keys.get(),
                                  output_keys.get(),
                                  input_vals.get(),
                                  output_vals.get(),
                                  input_keys.size(),
                                  k));

    // Allocate temporary storage
    common::device_ptr<void> temp_storage(temp_storage_size);

    // Perform topk
    HIP_CHECK(rocprim::topk_pairs(temp_storage.get(),
                                  temp_storage_size,
                                  input_keys.get(),
                                  output_keys.get(),
                                  input_vals.get(),
                                  output_vals.get(),
                                  input_keys.size(),
                                  k));

    // Check for any errors
    HIP_CHECK(hipGetLastError());

    // Wait for the algorithm to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    const auto                             h_output_keys = output_keys.load();
    const auto                             h_output_vals = output_vals.load();
    const std::vector<std::pair<int, int>> expected      = {
        {-3, 1},
        { 1, 2},
        { 2, 5},
        { 4, 6}
    };
    std::vector<std::pair<int, int>> result = {};
    for(unsigned int i = 0; i < k; ++i)
    {
        result.emplace(result.end(), h_output_keys[i], h_output_vals[i]);
    }

    struct comp
    {
        bool operator()(std::pair<int, int> a, std::pair<int, int> b) const
        {
            // In this example, we assume that, keys and values are unique.
            return std::less{}(a.first, b.first);
        }
    };
    std::sort(result.begin(), result.end(), comp{});

    // Check that the result is correct
    ASSERT_TRUE(result == expected);
}

int main()
{
    example_topk_keys();
    example_topk_pairs();
    return 0;
}
