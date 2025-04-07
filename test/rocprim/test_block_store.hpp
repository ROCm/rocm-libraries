#include "../common_test_header.hpp"
#include "test_utils.hpp"

#include "../../common/utils.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/block/block_sort.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <random>

block_store_test_suite_type_def(suite_name, name_suffix);
typed_test_suite_def(suite_name, name_suffix, block_params);

// using header guards for these test functions because this file is included multiple times:
// once for the integrals test suite and once for the floating point test suite.
#ifndef TEST_ROCPRIM_TEST_BLOCK_STORE_HPP_
    #define TEST_ROCPRIM_TEST_BLOCK_STORE_HPP_

template<
    bool use_size,
    unsigned int block_size,
    unsigned int items_per_thread,
    typename DataType,
    rocprim::block_store_method algorithm
>
void TestStore(){

    constexpr size_t items_per_block = block_size * items_per_thread;
    constexpr size_t grid_size = 120;
    constexpr size_t size = items_per_block * grid_size;
    

    std::vector<DataType> host_input(size);
    common::device_ptr<DataType> device_output(host_input);
    
    for(size_t i = 0; i < size; i++) host_input[i] = static_cast<DataType>(i);
    common::device_ptr<DataType> device_input(host_input);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_kernel<
                                        use_size,
                                        block_size,
                                        items_per_thread,
                                        DataType,
                                        algorithm
                                    >),
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        device_input.get(),
        device_output.get()
    );

    HIP_CHECK(hipGetLastError());
    std::vector<DataType> host_output = device_output.load();
    test_utils::assert_eq(host_input, host_output);
}

template<
    bool use_size,
    unsigned int block_size,
    unsigned int items_per_thread,
    typename DataType,
    rocprim::block_store_method algorithm
>
void TestStoreWithStorage(){

    constexpr size_t items_per_block = block_size * items_per_thread;
    constexpr size_t grid_size = 120;
    constexpr size_t size = items_per_block * grid_size;
    

    std::vector<DataType> host_input(size);
    common::device_ptr<DataType> device_output(host_input);
    
    for(size_t i = 0; i < size; i++) host_input[i] = static_cast<DataType>(i);
    common::device_ptr<DataType> device_input(host_input);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_kernel_with_storage<
                                        use_size,
                                        block_size,
                                        items_per_thread,
                                        DataType,
                                        algorithm
                                    >),
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        device_input.get(),
        device_output.get()
    );

    HIP_CHECK(hipGetLastError());
    std::vector<DataType> host_output = device_output.load();
    test_utils::assert_eq(host_input, host_output);
}

#endif

typed_test_def(suite_name, name_suffix, Store)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 1;
    static constexpr const bool                        use_size = false;
    TestStore<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreMultipleItemsPerThread)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 4;
    static constexpr const bool                        use_size = false;
    TestStore<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreWithSize)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 1;
    static constexpr const bool                        use_size = true;
    TestStore<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreWithSizeMultipleItemsPerThread)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 4;
    static constexpr const bool                        use_size = true;
    TestStore<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreWithStorage)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 1;
    static constexpr const bool                        use_size = false;
    TestStoreWithStorage<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreMultipleItemsPerThreadWithStorage)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 4;
    static constexpr const bool                        use_size = false;
    TestStoreWithStorage<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreWithSizeWithStorage)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 1;
    static constexpr const bool                        use_size = true;
    TestStoreWithStorage<use_size, block_size, items_per_thread, DataType, algo>();
}

typed_test_def(suite_name, name_suffix, StoreWithSizeMultipleItemsPerThreadWithStorage)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 4;
    static constexpr const bool                        use_size = true;
    TestStoreWithStorage<use_size, block_size, items_per_thread, DataType, algo>();
}
