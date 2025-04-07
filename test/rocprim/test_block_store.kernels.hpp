#ifndef TEST_BLOCK_STORE_KERNELS_HPP_
#define TEST_BLOCK_STORE_KERNELS_HPP_



constexpr bool is_buildable(unsigned int BlockSize,
                            unsigned int ItemsPerThread,
                            rocprim::block_store_method algorithm
                        )
{
    switch(algorithm)
    {
        case rocprim::block_store_method::block_store_direct: 
        case rocprim::block_store_method::block_store_striped: 
        case rocprim::block_store_method::block_store_transpose:
            return true;
        case rocprim::block_store_method::block_store_vectorize:
            return (ItemsPerThread % 2 == 0) && ((BlockSize * ItemsPerThread) % 4 == 0);
        case rocprim::block_store_method::block_store_warp_transpose:
            return BlockSize % rocprim::device_warp_size() == 0;
    }
    return false;
}

template<
    bool useSize, 
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    rocprim::block_store_method algorithm,
    std::enable_if_t<(is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
__global__ __launch_bounds__(BlockSize) void store_kernel(DataType * input, DataType * output){
    using bstore_type = rocprim::block_store<DataType, BlockSize, ItemsPerThread, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + (threadIdx.x * ItemsPerThread);

    DataType temp[ItemsPerThread]; 
    __shared__ DataType storage[ItemsPerBlock]; 

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        switch(algorithm){
            case rocprim::block_store_method::block_store_direct:
            case rocprim::block_store_method::block_store_transpose:
            case rocprim::block_store_method::block_store_vectorize:
                temp[i] = input[index + i];
            break;
            
            case rocprim::block_store_method::block_store_striped:
                temp[i] = input[block_offset + (threadIdx.x + i * BlockSize)];
                break;
        }
    }

    if(useSize)
        bstore_type().store(storage, temp, ItemsPerBlock);
    else
        bstore_type().store(storage, temp);

    __syncthreads();

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        output[index + i] = storage[threadIdx.x * ItemsPerThread + i];
    }
}

template<
    bool useSize, 
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    rocprim::block_store_method algorithm,
    std::enable_if_t<!(is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
__global__ __launch_bounds__(BlockSize) void store_kernel(DataType * input, DataType * output){
    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + (threadIdx.x * ItemsPerThread);

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        output[index + i] = input[index + i];
    }
}

template<
    bool useSize, 
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    rocprim::block_store_method algorithm,
    std::enable_if_t<(is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
__global__ __launch_bounds__(BlockSize) void store_kernel_with_storage(DataType * input, DataType * output){
    using bstore_type = rocprim::block_store<DataType, BlockSize, ItemsPerThread, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + (threadIdx.x * ItemsPerThread);

    DataType temp[ItemsPerThread]; 
    __shared__ DataType temp_out[ItemsPerBlock]; 

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        switch(algorithm){
            case rocprim::block_store_method::block_store_direct:
            case rocprim::block_store_method::block_store_transpose:
            case rocprim::block_store_method::block_store_vectorize:
                temp[i] = input[index + i];
            break;
            
            case rocprim::block_store_method::block_store_striped:
                temp[i] = input[block_offset + (threadIdx.x + i * BlockSize)];
                break;
        }
    }
    ROCPRIM_SHARED_MEMORY typename bstore_type::storage_type storage;

    if(useSize)
        bstore_type().store(temp_out, temp, ItemsPerBlock, storage);
    else
        bstore_type().store(temp_out, temp, storage);

    __syncthreads();

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        output[index + i] = temp_out[threadIdx.x * ItemsPerThread + i];
    }
}

template<
    bool useSize, 
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    rocprim::block_store_method algorithm,
    std::enable_if_t<!(is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
__global__ __launch_bounds__(BlockSize) void store_kernel_with_storage(DataType * input, DataType * output){
    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + (threadIdx.x * ItemsPerThread);

    for(unsigned int i = 0; i < ItemsPerThread; i++){
        output[index + i] = input[index + i];
    }
}

#endif