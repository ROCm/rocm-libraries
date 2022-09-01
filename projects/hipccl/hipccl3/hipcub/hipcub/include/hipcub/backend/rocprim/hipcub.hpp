/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2022, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_HIPCUB_HPP_
#define HIPCUB_ROCPRIM_HIPCUB_HPP_

#include "../../config.hpp"

// Block
#include "block/block_adjacent_difference.hpp"
#include "block/block_discontinuity.hpp"
#include "block/block_exchange.hpp"
#include "block/block_histogram.hpp"
#include "block/block_load.hpp"
#include "block/block_merge_sort.hpp"
#include "block/block_radix_rank.hpp"
#include "block/block_radix_sort.hpp"
#include "block/block_raking_layout.hpp"
#include "block/block_reduce.hpp"
#include "block/block_run_length_decode.hpp"
#include "block/block_scan.hpp"
#include "block/block_shuffle.hpp"
#include "block/block_store.hpp"
#include "block/radix_rank_sort_operations.hpp"

// Device
#include "device/device_adjacent_difference.hpp"
#include "device/device_histogram.hpp"
#include "device/device_merge_sort.hpp"
#include "device/device_partition.hpp"
#include "device/device_radix_sort.hpp"
#include "device/device_reduce.hpp"
#include "device/device_run_length_encode.hpp"
#include "device/device_scan.hpp"
#include "device/device_segmented_radix_sort.hpp"
#include "device/device_segmented_reduce.hpp"
#include "device/device_segmented_sort.hpp"
#include "device/device_select.hpp"
#include "device/device_spmv.hpp"

// Grid
#include "grid/grid_barrier.hpp"
#include "grid/grid_even_share.hpp"
#include "grid/grid_mapping.hpp"
#include "grid/grid_queue.hpp"

// Iterator
#include "iterator/arg_index_input_iterator.hpp"
#include "iterator/cache_modified_input_iterator.hpp"
#include "iterator/cache_modified_output_iterator.hpp"
#include "iterator/constant_input_iterator.hpp"
#include "iterator/counting_input_iterator.hpp"
#include "iterator/discard_output_iterator.hpp"
#include "iterator/tex_obj_input_iterator.hpp"
#include "iterator/tex_ref_input_iterator.hpp"
#include "iterator/transform_input_iterator.hpp"

// Thread
#include "thread/thread_load.hpp"
#include "thread/thread_operators.hpp"
#include "thread/thread_reduce.hpp"
#include "thread/thread_scan.hpp"
#include "thread/thread_search.hpp"
#include "thread/thread_sort.hpp"
#include "thread/thread_store.hpp"

// Warp
#include "warp/warp_exchange.hpp"
#include "warp/warp_load.hpp"
#include "warp/warp_merge_sort.hpp"
#include "warp/warp_reduce.hpp"
#include "warp/warp_scan.hpp"
#include "warp/warp_store.hpp"

// Util
#include "util_allocator.hpp"
#include "util_ptx.hpp"
#include "util_type.hpp"

#endif // HIPCUB_ROCPRIM_HIPCUB_HPP_
