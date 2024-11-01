// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// required test headers
#include "../common_test_header.hpp"

// required rocprim headers
#include "../rocprim/test_seed.hpp"
#include "../rocprim/test_utils.hpp"

#include <rocprim/device/device_binary_search.hpp>
#include <rocprim/device/device_merge_sort.hpp>

// required STL headers
#include <algorithm>

template<typename KeyType>
void generate_needles(const std::vector<KeyType>& input, std::vector<KeyType>& output, const size_t search_needle_size, const std::pair<KeyType, KeyType>& bounds, const seed_type& seed_value)
{
    // Pick 50% of the needles from the input vector
    std::vector<KeyType> indices = test_utils::get_random_data<KeyType>(search_needle_size / 2, 0, input.size() - 1, seed_value);

    // Do selection on the in-bounds indices and write the results to the output vector
    std::transform(indices.begin(), indices.end(), output.begin(), [&input](const KeyType& index)
    {
        return input[index];
    });

    // Generate the other 50% from outside the input vector
    const KeyType max_val = std::get<1>(bounds);
    std::vector<KeyType> out_of_bounds_vals = test_utils::get_random_data<KeyType>(search_needle_size - search_needle_size / 2, max_val, max_val * 2, seed_value);

    // Append the out-of-bounds values
    for (size_t i = 0; i < out_of_bounds_vals.size(); i++)
        output[indices.size() + i] = out_of_bounds_vals[i];

    // Mix up the in-bounds and out-of-bounds values to make the test a bit more robust
    std::random_device         rd;
    std::default_random_engine gen(rd());
    std::shuffle(output.begin(), output.end(), gen);
}

template<typename KeyType, typename BinaryFunction>
void computeExpectedSortAndSearchResult(std::vector<KeyType>& sort_input, const std::vector<KeyType>& search_needles, std::vector<KeyType>& expected_search_output, BinaryFunction compare_op)
{
    // Sort
    std::stable_sort(sort_input.begin(), sort_input.end(), compare_op);
    
    // Search
    for (size_t i = 0; i < search_needles.size(); i++)
        expected_search_output[i] = std::binary_search(sort_input.begin(), sort_input.end(), search_needles[i], compare_op);
}

// This test creates a graph that performs a device-wide merge_sort followed by a device-wide binary_search.
// After the graph is created, it is launched multiple times, each time using different data.
TEST(TestHipGraphAlgs, SortAndSearch)
{
    // Test case params
    using key_type = int;
    using compare_fcn_type = typename ::rocprim::less<key_type>;
    compare_fcn_type compare_op;
    const size_t sort_data_size = 4096;
    const size_t search_needle_size = 100;
    const size_t num_trials = 5;
    std::pair<key_type, key_type> bounds = std::make_pair(-10000, 10000); // generated data will fall in this range
    const bool debug_synchronous = false;

    // Set the device
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Generate data on the host
    const seed_type seed_value = seeds[random_seeds_count - 1];
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
    SCOPED_TRACE(testing::Message() << "with sort_data_size = " << sort_data_size);
    SCOPED_TRACE(testing::Message() << "with search_needle_size = " << search_needle_size);

    // Allocate device buffers and copy data into them
    key_type* d_sort_input = nullptr;
    key_type* d_sort_output = nullptr; // also used as search_input
    key_type* d_search_output = nullptr;
    key_type* d_search_needles = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_sort_input, sort_data_size * sizeof(key_type)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_sort_output, sort_data_size * sizeof(key_type)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_search_output, search_needle_size * sizeof(key_type)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_search_needles, search_needle_size * sizeof(key_type)));

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Get temporary storage size required for merge_sort.
    // Note: doing this inside a graph doesn't gain us any benefit,
    // since these calls run entirely on the host - however, it is
    // important to validate that they work inside a graph capture block.
    size_t sort_temp_storage_bytes = 0;
    HIP_CHECK(rocprim::merge_sort(nullptr,
                                  sort_temp_storage_bytes,
                                  d_sort_input,
                                  d_sort_output,
                                  sort_data_size,
                                  compare_op,
                                  stream,
                                  debug_synchronous
                                  ));

    // Get size of temporary storage required for binary_search
    size_t search_temp_storage_bytes = 0;
    HIP_CHECK(rocprim::binary_search(nullptr,
                                     search_temp_storage_bytes,
                                     d_sort_output,
                                     d_search_needles,
                                     d_search_output,
                                     sort_data_size,
                                     search_needle_size,
                                     compare_op,
                                     stream,
                                     debug_synchronous
                                     ));

    // Allocate the temp storage
    // Note: a single store will be used for both the sort and search algorithms
    size_t temp_storage_bytes = std::max(sort_temp_storage_bytes, search_temp_storage_bytes);
    // temp_storage_size_bytes must be > 0
    ASSERT_GT(temp_storage_bytes, 0);

    void* d_temp_storage = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Begin graph capture
    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(stream);

    // Launch merge_sort
    HIP_CHECK(
              rocprim::merge_sort(d_temp_storage,
                                  sort_temp_storage_bytes,
                                  d_sort_input,
                                  d_sort_output,
                                  sort_data_size,
                                  compare_op,
                                  stream,
                                  false
                                  )
              );

    // Launch binary_search
    HIP_CHECK(
              rocprim::binary_search<rocprim::default_config>(d_temp_storage,
                                                              search_temp_storage_bytes,
                                                              d_sort_output,
                                                              d_search_needles,
                                                              d_search_output,
                                                              sort_data_size,
                                                              search_needle_size,
                                                              compare_op,
                                                              stream,
                                                              false
                                                              );
              );

    // End graph capture, but do not execute the graph yet.
    gHelper.endStreamCapture(stream);
    gHelper.createGraph();

    std::vector<key_type> sort_input;
    std::vector<key_type> search_needles(search_needle_size);
    std::vector<key_type> expected_search_output(search_needle_size);
    std::vector<key_type> device_output(search_needle_size);

    // We'll launch the graph multiple times with different data.
    for (size_t i = 0; i < num_trials; i++)
    {
        // Generate the test data
        sort_input = test_utils::get_random_data<key_type>(sort_data_size, std::get<0>(bounds), std::get<1>(bounds), seed_value);
        generate_needles(sort_input, search_needles, search_needle_size, bounds, seed_value);

        // Compute the expected result on the host
        computeExpectedSortAndSearchResult<key_type, compare_fcn_type>(sort_input, search_needles, expected_search_output, compare_op);
        
        // Copy input data to the device
        HIP_CHECK(hipMemcpy(d_sort_input, sort_input.data(), sort_data_size * sizeof(key_type), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_search_needles, search_needles.data(), search_needle_size * sizeof(key_type), hipMemcpyHostToDevice));
        
        // Launch the graph
        gHelper.launchGraph(stream, true); 

        // Copy output back to host
        HIP_CHECK(hipMemcpy(device_output.data(), d_search_output, search_needle_size * sizeof(key_type), hipMemcpyDeviceToHost));
        
        // Validate the results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(device_output, expected_search_output));
    }

    // Clean up
    HIP_CHECK(hipFree(d_sort_input));
    HIP_CHECK(hipFree(d_sort_output));
    HIP_CHECK(hipFree(d_search_output));
    HIP_CHECK(hipFree(d_search_needles));
    HIP_CHECK(hipFree(d_temp_storage));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}
                                  
                                  