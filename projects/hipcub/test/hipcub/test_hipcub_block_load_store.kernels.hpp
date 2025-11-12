// MIT License
//
// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_HIPCUB_BLOCK_LOAD_STORE_KERNELS_HPP
#define HIPCUB_TEST_HIPCUB_BLOCK_LOAD_STORE_KERNELS_HPP

#include "test_utils.hpp"

// hipcub API
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>

template<class Type,
         hipcub::BlockLoadAlgorithm  Load,
         hipcub::BlockStoreAlgorithm Store,
         unsigned int                BlockSize,
         unsigned int                ItemsPerThread>
struct class_params
{
    using type                                                    = Type;
    static constexpr hipcub::BlockLoadAlgorithm  load_method      = Load;
    static constexpr hipcub::BlockStoreAlgorithm store_method     = Store;
    static constexpr unsigned int                block_size       = BlockSize;
    static constexpr unsigned int                items_per_thread = ItemsPerThread;
};

#define class_param_items(load_algo, store_algo, type, block_size) \
    class_params<type, load_algo, store_algo, block_size, 1>,      \
        class_params<type, load_algo, store_algo, block_size, 4>

#define class_param_block_size(load_algo, store_algo, type) \
    class_param_items(load_algo, store_algo, type, 64U),    \
        class_param_items(load_algo, store_algo, type, 256U)

#define class_param_block_size_512(load_algo, store_algo, type) \
    class_param_block_size(load_algo, store_algo, type),        \
        class_param_items(load_algo, store_algo, type, 512U)

#define class_param_type(load_algo, store_algo)                                           \
    class_param_block_size_512(load_algo, store_algo, int),                               \
        class_param_block_size_512(load_algo, store_algo, double),                        \
        class_param_block_size_512(load_algo, store_algo, test_utils::half),              \
        class_param_block_size_512(load_algo, store_algo, test_utils::bfloat16),          \
        class_param_block_size(load_algo, store_algo, test_utils::custom_test_type<int>), \
        class_param_block_size(load_algo, store_algo, test_utils::custom_test_type<double>)

using LoadStoreParamsDirect
    = ::testing::Types<class_param_type(hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                                        hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT)>;

using LoadStoreParamsStriped
    = ::testing::Types<class_param_type(hipcub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED,
                                        hipcub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED)>;

using LoadStoreParamsVectorize
    = ::testing::Types<class_param_type(hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                                        hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE)>;

using LoadStoreParamsTranspose
    = ::testing::Types<class_param_type(hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                                        hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE)>;

template<class Type,
         hipcub::BlockLoadAlgorithm  LoadMethod,
         hipcub::BlockStoreAlgorithm StoreMethod,
         unsigned int                BlockSize,
         unsigned int                ItemsPerThread>
__global__ __launch_bounds__(BlockSize) void load_store_kernel(Type * device_input,
                                                               Type * device_output)
{
    Type         items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod>   load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items);
    store.Store(device_output + offset, items);
}

template<class Type,
         hipcub::BlockLoadAlgorithm  LoadMethod,
         hipcub::BlockStoreAlgorithm StoreMethod,
         unsigned int                BlockSize,
         unsigned int                ItemsPerThread>
__global__ __launch_bounds__(BlockSize) void load_store_valid_kernel(Type * device_input,
                                                                     Type * device_output,
                                                                     size_t valid)
{
    Type         items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod>   load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items, valid);
    store.Store(device_output + offset, items, valid);
}

template<class Type,
         hipcub::BlockLoadAlgorithm  LoadMethod,
         hipcub::BlockStoreAlgorithm StoreMethod,
         unsigned int                BlockSize,
         unsigned int                ItemsPerThread>
__global__ __launch_bounds__(BlockSize) void load_store_valid_default_kernel(Type*  device_input,
                                                                             Type*  device_output,
                                                                             size_t valid,
                                                                             Type   _default)
{
    Type         items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod>   load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items, valid, _default);
    store.Store(device_output + offset, items);
}

template<typename InputIteratorT,
         typename OutputIteratorT,
         hipcub::BlockLoadAlgorithm  LoadMethod,
         hipcub::BlockStoreAlgorithm StoreMethod,
         unsigned int                BlockSize,
         unsigned int                ItemsPerThread>
__launch_bounds__(BlockSize) __global__
    void load_store_guarded_kernel(InputIteratorT  d_in,
                                   OutputIteratorT d_out_unguarded,
                                   OutputIteratorT d_out_guarded,
                                   int             num_items)
{
    enum
    {
        TileSize = BlockSize * ItemsPerThread
    };

    // The input value type
    using InputT = typename std::iterator_traits<InputIteratorT>::value_type;

    // The output value type
    using OutputT = typename std::conditional<
        (std::is_same<typename std::iterator_traits<OutputIteratorT>::value_type,
                      void>::value), // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type, // ... then the input iterator's
        // value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::
        type; // ... else the output iterator's value type

    // Threadblock load/store abstraction types
    using BlockLoad  = hipcub::BlockLoad<InputT, BlockSize, ItemsPerThread, LoadMethod>;
    using BlockStore = hipcub::BlockStore<OutputT, BlockSize, ItemsPerThread, StoreMethod>;

    // Shared memory type for this thread block
    union TempStorage
    {
        typename BlockLoad::TempStorage  load;
        typename BlockStore::TempStorage store;
    };

    // Allocate temp storage in shared memory
    __shared__ TempStorage temp_storage;

    // Threadblock work bounds
    int block_offset     = blockIdx.x * TileSize;
    int guarded_elements = max(num_items - block_offset, 0);

    // Tile of items
    OutputT data[ItemsPerThread];

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_unguarded + block_offset, data);

    __syncthreads();

    // reset data
#pragma unroll
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
        data[item] = OutputT();

    __syncthreads();

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data, guarded_elements);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_guarded + block_offset, data, guarded_elements);
}

#endif  // HIPCUB_TEST_HIPCUB_BLOCK_LOAD_STORE_KERNELS_HPP
