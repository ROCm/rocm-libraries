// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include "../../config.hpp"

#include "../util_type.hpp"

#include "block_scan.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename KeyT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    typename ValueT = NullType,
    int RADIX_BITS = 4, /* ignored */
    bool MEMOIZE_OUTER_SCAN = true, /* ignored */
    BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS, /* ignored */
    hipSharedMemConfig SMEM_CONFIG = hipSharedMemBankSizeFourByte, /* ignored */
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int PTX_ARCH = HIPCUB_ARCH /* ignored */
>
class BlockRadixSort
    : private ::rocprim::block_radix_sort<
        KeyT,
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
        ITEMS_PER_THREAD,
        ValueT
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_radix_sort<
            KeyT,
            BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
            ITEMS_PER_THREAD,
            ValueT
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockRadixSort() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockRadixSort(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
              int begin_bit = 0,
              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
              ValueT (&values)[ITEMS_PER_THREAD],
              int begin_bit = 0,
              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                        ValueT (&values)[ITEMS_PER_THREAD],
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                              int begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                              ValueT (&values)[ITEMS_PER_THREAD],
                              int begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                        int begin_bit = 0,
                                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                        ValueT (&values)[ITEMS_PER_THREAD],
                                        int begin_bit = 0,
                                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
