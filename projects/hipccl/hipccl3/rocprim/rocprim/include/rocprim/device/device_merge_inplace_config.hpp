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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_CONFIG_HPP_

#include "../config.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \tparam GlobalMergeBlockSize Number of threads per block for global merging.
/// \tparam BlockMergeBlockSize Number of threads per block for block-level merging.
/// \tparam BlockMergeItemsPerThread number of items per thread for block-level merging.
template<unsigned int GlobalMergeBlockSize     = 256,
         unsigned int BlockMergeBlockSize      = 1024,
         unsigned int BlockMergeItemsPerThread = 1024>
struct merge_inplace_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    static constexpr unsigned int global_merge_block_size      = GlobalMergeBlockSize;
    static constexpr unsigned int block_merge_block_size       = BlockMergeBlockSize;
    static constexpr unsigned int block_merge_items_per_thread = BlockMergeItemsPerThread;
#endif
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif
