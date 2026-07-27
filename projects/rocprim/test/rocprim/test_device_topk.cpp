// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_sort_comparator.hpp"

// required rocprim headers
#include <rocprim/device/device_topk.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

// std library
#include <unordered_set>
#include <vector>

// Params for tests
template<class KeyType,
         class ValueType          = size_t,
         bool Descending          = false,
         class Decomposer         = rocprim::identity_decomposer,
         class Config             = rocprim::default_config,
         bool Ordered             = false,
         bool Deterministic       = false,
         bool Stable              = false,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceTopkParams
{
    using key_type                              = KeyType;
    using value_type                            = ValueType;
    static constexpr bool descending            = Descending;
    using decomposer_t                          = Decomposer;
    using config                                = Config;
    static constexpr bool ordered               = Ordered;
    static constexpr bool deterministic         = Deterministic;
    static constexpr bool stable                = Stable;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<bool Descending, class InputVector, class OutputVector, class SizeOut>
void inline compare_k(InputVector input, OutputVector output, SizeOut k)
{
    using key_type          = typename InputVector::value_type;
    constexpr int start_bit = 0;
    constexpr int end_bit   = sizeof(key_type) * 8;
    using comparator        = test_utils::key_comparator<key_type, Descending, start_bit, end_bit>;

    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::stable_sort(sorted_input.begin(), sorted_input.end(), comparator());

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);
    std::stable_sort(sorted_output.begin(), sorted_output.end(), comparator());

    // Only check that first k coincide
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input, k));
}

// Get the free memory size with a coefficient.
size_t get_current_free_mem_size(float coefficient)
{
    size_t free_mem;
    [[maybe_unused]]
    size_t total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    return static_cast<size_t>(static_cast<float>(free_mem) * coefficient);
}

namespace utils
{
    template<class T>
ROCPRIM_FORCE_INLINE bool isnan(T h)
    {
        return std::isnan(h);
    }
    template<>
ROCPRIM_FORCE_INLINE bool isnan<rocprim::half>(rocprim::half h)
    {
        const auto& bits = reinterpret_cast<const uint16_t&>(h);
        return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
    }
    template<>
ROCPRIM_FORCE_INLINE bool isnan<rocprim::bfloat16>(rocprim::bfloat16 h)
    {
        const auto& bits = reinterpret_cast<const uint16_t&>(h);
        return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
    }
    template<class T>
ROCPRIM_FORCE_INLINE bool isinf(T h)
    {
        return std::isinf(h);
    }
    template<>
ROCPRIM_FORCE_INLINE bool isinf<rocprim::half>(rocprim::half h)
    {
        const auto& bits = reinterpret_cast<const uint16_t&>(h);
        return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) == 0;
    }
    template<>
ROCPRIM_FORCE_INLINE bool isinf<rocprim::bfloat16>(rocprim::bfloat16 h)
    {
        const auto& bits = reinterpret_cast<const uint16_t&>(h);
        return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) == 0;
    }

    template<class T>
    bool is_negative(T h)
    {
        if constexpr(std::is_same_v<T, rocprim::bfloat16> || std::is_same_v<T, rocprim::half>)
        {
            const auto& bits = reinterpret_cast<const uint16_t&>(h);
            return (bits & 0x8000) != 0;
        }
        else
        {
            return std::signbit(h);
        }
    }

    template<class T>
    auto less(T a, T b)
    {
        if constexpr(rocprim::is_floating_point<T>::value)
        {
            if(utils::isnan(a))
            {
                return is_negative(a);
            }
            else if(utils::isnan(b))
            {
                return !is_negative(b);
            }
            if(utils::isinf(a))
            {
                return is_negative(a);
            }
            else if(utils::isinf(b))
            {
                return !is_negative(b);
            }
        }
        return rocprim::less<T>{}(a, b);
    }

    template<class T>
    auto greater(T a, T b)
    {
        if constexpr(rocprim::is_floating_point<T>::value)
        {
            if(utils::isnan(a))
            {
                return !is_negative(a);
            }
            else if(utils::isnan(b))
            {
                return is_negative(b);
            }
            if(utils::isinf(a))
            {
                return !is_negative(a);
            }
            else if(utils::isinf(b))
            {
                return is_negative(b);
            }
        }
        return rocprim::greater<T>{}(a, b);
    }

    } // namespace utils

    template<class OutIterable, class InIterable, class Predicate, class KType>
    void host_nth_element(OutIterable&      h_output,
                          InIterable const& h_input,
                          KType             K,
                          Predicate         predicate)
    {
        h_output.clear();
        const auto N   = h_input.size();
        using common_t = typename std::common_type<decltype(K), decltype(N)>::type;
        ASSERT_GE(N, 0);
        ASSERT_GE(K, 0);
        ASSERT_GE(static_cast<common_t>(N), static_cast<common_t>(K));
        InIterable sorted_input(h_input.cbegin(), h_input.cend());
        // Sort the whole input
        std::sort(sorted_input.begin(), sorted_input.end(), predicate);
        h_output.insert(h_output.end(), sorted_input.begin(), sorted_input.begin() + K);
    }

template<bool Descending,
         class InputKeyVector,
         class InputValueVector,
         class OutputKeyVector,
         class OutputValueVector,
         class SizeOut>
ROCPRIM_FORCE_INLINE void compare_pairs_k(InputKeyVector    keys_input,
                            InputValueVector  values_input,
                            OutputKeyVector   keys_output,
                            OutputValueVector values_output,
                            SizeOut           k)
{
    using key_type          = typename InputKeyVector::value_type;
    using value_type        = typename InputValueVector::value_type;
    using key_value         = std::pair<key_type, value_type>;
    constexpr int start_bit = 0;
    constexpr int end_bit   = sizeof(key_type) * 8;
    using comparator
        = test_utils::key_value_comparator<key_type, value_type, Descending, start_bit, end_bit>;

    const std::size_t size = values_input.size();

    // Calculate sorted input results on host
    std::vector<key_value> sorted_input(size);
    for(std::size_t i = 0; i < size; i++)
    {
        sorted_input[i] = key_value(keys_input[i], values_input[i]);
    }
    std::stable_sort(sorted_input.begin(), sorted_input.end(), comparator());

    // Calculate sorted output results on host
    std::vector<key_value> sorted_output(k);
    for(std::size_t i = 0; i < k; i++)
    {
        sorted_output[i] = key_value(keys_output[i], values_output[i]);
    }
    std::stable_sort(sorted_output.begin(), sorted_output.end(), comparator());

    // Only check that first k coincide
    if constexpr(rocprim::is_integral<key_type>::value)
    {
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input, k));
    }
    else
    {
        for(decltype(k) i = 0; i < k; ++i)
        {
            const auto& out = sorted_output[i];
            const auto& in  = sorted_input[i];
            if(utils::isnan(out.first) && utils::isnan(in.first))
            {
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(out.second, in.second));
            }
            else
            {
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(out, in));
            }
        }
    }
}

// ---------------------------------------------------------
// Test for ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceTopkTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using value_type                            = typename Params::value_type;
    static constexpr bool descending            = Params::descending;
    using decomposer_t                          = typename Params::decomposer_t;
    using config                                = typename Params::config;
    static constexpr bool ordered               = Params::ordered;
    static constexpr bool deterministic         = Params::deterministic;
    static constexpr bool stable                = Params::stable;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

template<class T>
using grouped_params = ::testing::Types<
    // Non-ordered, non-deterministic, non-stable
    // Ascending
    DeviceTopkParams<T, size_t, false>,
    // Descending
    DeviceTopkParams<T, size_t, true>,

    // Non-ordered, non-deterministic, stable
    // Ascending
    DeviceTopkParams<T,
                     size_t,
                     false,
                     rocprim::identity_decomposer,
                     rocprim::default_config,
                     false,
                     false,
                     true>,
    // Descending
    DeviceTopkParams<T,
                     size_t,
                     true,
                     rocprim::identity_decomposer,
                     rocprim::default_config,
                     false,
                     false,
                     true>>;

using RocprimDeviceTopkTestsParams =
    typename test_utils::merge_sequence<grouped_params<int8_t>,
                                        grouped_params<short>,
                                        grouped_params<int>,
                                        grouped_params<rocprim::uint128_t>,
                                        grouped_params<long long>,
                                        grouped_params<rocprim::half>,
                                        grouped_params<float>,
                                        grouped_params<double>>::type;

TYPED_TEST_SUITE(RocprimDeviceTopkTests, RocprimDeviceTopkTestsParams);

// Basic TopK tests

TYPED_TEST(RocprimDeviceTopkTests, TopkKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                       = typename TestFixture::key_type;
    constexpr bool descending            = TestFixture::descending;
    using decomposer_t                   = typename TestFixture::decomposer_t;
    using config                         = typename TestFixture::config;
    using size_in_type                   = unsigned int;
    using size_out_type                  = unsigned int;
    constexpr bool ordered               = TestFixture::ordered;
    constexpr bool deterministic         = TestFixture::deterministic;
    constexpr bool stable                = TestFixture::stable;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_graphs            = TestFixture::use_graphs;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(unsigned int seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_in_type size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if constexpr(use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            size_out_type k = 0;
            if(size > 0)
            {
                k = test_utils::get_random_value<size_out_type>(0, size - 1, seed_value);
            }

            SCOPED_TRACE(testing::Message() << "with k = " << k);

            // Generate data
            engine_type           rng_engine(seed_value);
            std::vector<key_type> input(size);
            test_utils::generate_key_input(input.begin(), size, rng_engine);

            common::device_ptr<key_type> d_input;
            common::device_ptr<key_type> d_output;

            if(!d_input.resize_with_memory_check(size) || !d_output.resize_with_memory_check(k))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }
            d_input.store(input);

            const auto input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

            // decomposer
            decomposer_t decomposer;

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK((rocprim::topk<config, descending, ordered, deterministic, stable>(
                nullptr,
                temp_storage_size_bytes,
                input_it,
                d_output.get(),
                size,
                k,
                decomposer,
                stream,
                debug_synchronous)));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;

            if constexpr(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK((rocprim::topk<config, descending, ordered, deterministic, stable>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                input_it,
                d_output.get(),
                size,
                k,
                decomposer,
                stream,
                debug_synchronous)));

            if constexpr(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();
            if(size > 0)
            {
                compare_k<descending>(input, output, k);
            }

            if constexpr(use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceTopkTests, TopkPairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                       = typename TestFixture::key_type;
    using value_type                     = typename TestFixture::value_type;
    constexpr bool descending            = TestFixture::descending;
    using decomposer_t                   = typename TestFixture::decomposer_t;
    using config                         = typename TestFixture::config;
    using size_in_type                   = unsigned int;
    using size_out_type                  = unsigned int;
    constexpr bool ordered               = TestFixture::ordered;
    constexpr bool deterministic         = TestFixture::deterministic;
    constexpr bool stable                = TestFixture::stable;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_graphs            = TestFixture::use_graphs;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(unsigned int seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_in_type size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if constexpr(use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            size_out_type k = 0;

            if(size > 0)
            {
                k = test_utils::get_random_value<size_out_type>(0, size - 1, seed_value);
            }

            SCOPED_TRACE(testing::Message() << "with k = " << k);

            // Generate data
            engine_type           rng_engine(seed_value);
            std::vector<key_type> keys_input(size);
            test_utils::generate_key_input(keys_input.begin(), size, rng_engine);
            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type>   d_keys_input;
            common::device_ptr<value_type> d_values_input;
            common::device_ptr<key_type>   d_keys_output;
            common::device_ptr<value_type> d_values_output;

            if(!d_keys_input.resize_with_memory_check(size)
               || !d_values_input.resize_with_memory_check(size)
               || !d_keys_output.resize_with_memory_check(k)
               || !d_values_output.resize_with_memory_check(k))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }
            d_keys_input.store(keys_input);
            d_values_input.store(values_input);

            const auto keys_input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_keys_input.get());
            const auto values_input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(
                    d_values_input.get());

            // decomposer
            decomposer_t decomposer;

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK((rocprim::topk_pairs<config, descending, ordered, deterministic, stable>(
                nullptr,
                temp_storage_size_bytes,
                keys_input_it,
                d_keys_output.get(),
                values_input_it,
                d_values_output.get(),
                size,
                k,
                decomposer,
                stream,
                debug_synchronous)));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;

            if constexpr(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK((rocprim::topk_pairs<config, descending, ordered, deterministic, stable>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                keys_input_it,
                d_keys_output.get(),
                values_input_it,
                d_values_output.get(),
                size,
                k,
                decomposer,
                stream,
                debug_synchronous)));

            if constexpr(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            if constexpr(use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }

            if(size <= 0)
            {
                continue;
            }

            if constexpr(stable)
            { // For stable implementation, directly use compare_pairs_k
                // Copy output to host
                const auto keys_output   = d_keys_output.load();
                const auto values_output = d_values_output.load();
                compare_pairs_k<descending>(keys_input,
                                            values_input,
                                            keys_output,
                                            values_output,
                                            k);
            }
            else
            {
                // For unstable implementation, using this device algorithm radix_sort_pairs makes
                // it a faster.

                // Verify ouput by using radix_sort_pairs
                if constexpr(use_graphs)
                {
                    HIP_CHECK(hipStreamCreate(&stream));
                }
                // Get size of temporary_storage
                temp_storage_size_bytes = 0;

                auto ret = rocprim::radix_sort_pairs(nullptr,
                                                     temp_storage_size_bytes,
                                                     d_keys_output.get(),
                                                     d_keys_output.get(),
                                                     d_values_output.get(),
                                                     d_values_output.get(),
                                                     k,
                                                     0,
                                                     8 * sizeof(key_type),
                                                     stream);
                HIP_CHECK(ret);
                ASSERT_GT(temp_storage_size_bytes, 0);
                if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                ret = rocprim::radix_sort_pairs(d_temp_storage.get(),
                                                temp_storage_size_bytes,
                                                d_keys_output.get(),
                                                d_keys_output.get(),
                                                d_values_output.get(),
                                                d_values_output.get(),
                                                k,
                                                0,
                                                8 * sizeof(key_type),
                                                stream);
                HIP_CHECK(ret);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                if(use_graphs)
                {
                    HIP_CHECK(hipStreamDestroy(stream));
                }

                // Get output from device
                auto h_output_keys   = d_keys_output.load();
                auto h_output_values = d_values_output.load();
                if(descending)
                {
                    std::reverse(h_output_keys.begin(), h_output_keys.end());
                    std::reverse(h_output_values.begin(), h_output_values.end());
                }
                // Calculate expected output keys
                std::vector<key_type> h_expected_keys{};
                auto                  pred = [](auto _1, auto _2)
                {
                    if constexpr(!descending)
                    {
                        return utils::less<decltype(_1)>(_1, _2);
                    }
                    else
                    {
                        return utils::greater<decltype(_1)>(_1, _2);
                    }
                };
                host_nth_element(h_expected_keys, keys_input, k, pred);
                const auto output_size = k;
                ASSERT_EQ(d_keys_output.size(), output_size);
                ASSERT_EQ(h_output_values.size(), output_size);
                ASSERT_EQ(h_expected_keys.size(), output_size);

                if constexpr(rocprim::is_integral<key_type>::value)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_eq(h_output_keys, h_expected_keys, output_size));
                }
                else
                {
                    for(size_out_type i = 0; i < output_size; ++i)
                    {
                        const auto& out      = h_output_keys[i];
                        const auto& expected = h_expected_keys[i];
                        if(utils::isnan(out) && utils::isnan(expected))
                        {
                            continue;
                        }
                        else
                        {
                            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(out, expected));
                        }
                    }
                }

                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(h_output_keys, h_expected_keys, output_size));
                // Check values
                if constexpr(rocprim::is_integral<value_type>::value)
                {
                    // Integral types can be directly used as hash maps
                    for(unsigned int i = 0; i < output_size; ++i)
                    {
                        const auto& expected = keys_input[h_output_values[i]];
                        const auto& out      = h_output_keys[i];
                        if constexpr(rocprim::is_floating_point<key_type>::value)
                        {
                            if(utils::isnan(expected) && utils::isnan(out))
                            {
                                continue;
                            }
                        }
                        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected, out));
                    }
                }
                else
                {
                    // Note: this code would not be compiled, because, we only test for integral value types
                    // Using a hash map might be faster, but the creation of hash
                    // map also takes time. So here for now, we directly iterate
                    // over all items to see if the values can be found in the result.
                    for(unsigned int i = 0; i < output_size; ++i)
                    {
                        const auto& key   = h_output_keys[i];
                        const auto& val   = h_output_values[i];
                        bool        found = false;
                        for(unsigned int j = 0; j < size; ++j)
                        {
                            const auto& in_key = keys_input[j];
                            const auto& in_val = values_input[j];
                            if(in_val == val && in_key == key)
                            {
                                found = true;
                                break;
                            }
                            if(val == in_val && utils::isnan(in_key) && utils::isnan(key))
                            {
                                found = true;
                                break;
                            }
                        }
                        ASSERT_TRUE(found);
                    }
                }
            }
        }
    }
}

// Large sizes TopK tests

template<class config,
         bool descending,
         bool stable,
         class decomposer_t,
         bool use_graphs,
         bool use_indirect_iterator,
         class size_in_type  = unsigned int,
         class size_out_type = unsigned int>
void topk_large_sizes_test(bool debug_synchronous)
{
    using key_type = size_in_type;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    unsigned int seed_value = seeds[1];
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    std::vector<size_in_type> sizes = {(size_in_type{1} << 30) - 1, (size_in_type{1} << 32) + 15};

    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default
        if constexpr(use_graphs)
        {
            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }

        // This test needs to allocate 2 buffers of size: sizeof(key_type) * size, plus temporary storage.
        // Check if there could be no enough memory space
        if((2 * sizeof(key_type) * size) > get_current_free_mem_size(0.7))
        {
            std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
            return;
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_out_type k = 1;

        SCOPED_TRACE(testing::Message() << "with k = " << k);

        // Generate data
        std::vector<key_type> input(size);
        test_utils::iota(input.begin(), input.end(), 0);

        common::device_ptr<key_type> d_input;
        common::device_ptr<key_type> d_output;

        if(!d_input.resize_with_memory_check(size) || !d_output.resize_with_memory_check(k))
        {
            std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
            return;
        }
        d_input.store(input);

        const auto input_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

        // decomposer
        decomposer_t decomposer;

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK((rocprim::topk<config, descending, false, false, stable>(nullptr,
                                                                           temp_storage_size_bytes,
                                                                           input_it,
                                                                           d_output.get(),
                                                                           size,
                                                                           k,
                                                                           decomposer,
                                                                           stream,
                                                                           debug_synchronous)));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage;

        if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
        {
            std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
            return;
        }

        test_utils::GraphHelper gHelper;

        if constexpr(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK((rocprim::topk<config, descending, false, false, stable>(d_temp_storage.get(),
                                                                           temp_storage_size_bytes,
                                                                           input_it,
                                                                           d_output.get(),
                                                                           size,
                                                                           k,
                                                                           decomposer,
                                                                           stream,
                                                                           debug_synchronous)));

        if constexpr(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Validate output
        const auto output = d_output.load()[0];
        if constexpr(descending)
        {
            test_utils::assert_eq(output, size_in_type{size - 1});
        }
        else
        {
            test_utils::assert_eq(output, size_in_type{0});
        }

        if constexpr(use_graphs)
        {
            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

TEST(RocprimDeviceTopkTests, TopkLargeSizesStable)
{
    constexpr bool descending            = false;
    constexpr bool stable                = true;
    using decomposer_t                   = rocprim::identity_decomposer;
    using config                         = rocprim::default_config;
    using size_in_type                   = std::size_t;
    using size_out_type                  = std::size_t;
    const bool     debug_synchronous     = false;
    constexpr bool use_graphs            = false;
    constexpr bool use_indirect_iterator = false;

    // Radix TopK
    // Same data type for 'size' and 'k'
    topk_large_sizes_test<config,
                          descending,
                          stable,
                          decomposer_t,
                          use_graphs,
                          use_indirect_iterator,
                          size_in_type,
                          size_out_type>(debug_synchronous);
}

TEST(RocprimDeviceTopkTests, TopkLargeSizesUnstable)
{
    constexpr bool descending            = false;
    constexpr bool stable                = false;
    using decomposer_t                   = rocprim::identity_decomposer;
    using config                         = rocprim::default_config;
    using size_in_type                   = std::size_t;
    using size_out_type                  = std::size_t;
    const bool     debug_synchronous     = false;
    constexpr bool use_graphs            = false;
    constexpr bool use_indirect_iterator = false;

    // Radix TopK
    // Same data type for 'size' and 'k'
    topk_large_sizes_test<config,
                          descending,
                          stable,
                          decomposer_t,
                          use_graphs,
                          use_indirect_iterator,
                          size_in_type,
                          size_out_type>(debug_synchronous);
}
