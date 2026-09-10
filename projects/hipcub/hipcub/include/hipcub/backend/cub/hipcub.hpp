/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CUB_HIPCUB_HPP_
#define HIPCUB_CUB_HIPCUB_HPP_

#include "../../config.hpp"

// Block
#include <cub/block/block_adjacent_difference.cuh> // IWYU pragma: export
#include <cub/block/block_discontinuity.cuh> // IWYU pragma: export
#include <cub/block/block_exchange.cuh> // IWYU pragma: export
#include <cub/block/block_histogram.cuh> // IWYU pragma: export
#include <cub/block/block_load.cuh> // IWYU pragma: export
#include <cub/block/block_merge_sort.cuh> // IWYU pragma: export
#include <cub/block/block_radix_rank.cuh> // IWYU pragma: export
#include <cub/block/block_radix_sort.cuh> // IWYU pragma: export
#include <cub/block/block_raking_layout.cuh> // IWYU pragma: export
#include <cub/block/block_reduce.cuh> // IWYU pragma: export
#include <cub/block/block_run_length_decode.cuh> // IWYU pragma: export
#include <cub/block/block_scan.cuh> // IWYU pragma: export
#include <cub/block/block_shuffle.cuh> // IWYU pragma: export
#include <cub/block/block_store.cuh> // IWYU pragma: export
#include <cub/block/radix_rank_sort_operations.cuh> // IWYU pragma: export

// Device functions must be wrapped so they return
// hipError_t instead of cudaError_t
#include "device/device_adjacent_difference.hpp"
#include "device/device_copy.hpp"
#include "device/device_for.hpp"
#include "device/device_histogram.hpp"
#include "device/device_memcpy.hpp"
#include "device/device_merge.hpp"
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
#include "device/device_transform.hpp"

// Grid
#include <cub/grid/grid_even_share.cuh> // IWYU pragma: export
#include <cub/grid/grid_mapping.cuh> // IWYU pragma: export
#include <cub/grid/grid_queue.cuh> // IWYU pragma: export
// These functions must be wrapped so they return
// hipError_t instead of cudaError_t
#include "grid/grid_barrier.hpp"

// Iterator
#include <cub/iterator/arg_index_input_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/cache_modified_input_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/cache_modified_output_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/constant_input_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/counting_input_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/discard_output_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/tex_obj_input_iterator.cuh> // IWYU pragma: export
#include <cub/iterator/transform_input_iterator.cuh> // IWYU pragma: export

// Thread
#include <cub/thread/thread_load.cuh> // IWYU pragma: export
#include <cub/thread/thread_operators.cuh> // IWYU pragma: export
#include <cub/thread/thread_reduce.cuh> // IWYU pragma: export
#include <cub/thread/thread_scan.cuh> // IWYU pragma: export
#include <cub/thread/thread_search.cuh> // IWYU pragma: export
#include <cub/thread/thread_sort.cuh> // IWYU pragma: export
#include <cub/thread/thread_store.cuh> // IWYU pragma: export

// Warp
#include <cub/warp/warp_exchange.cuh> // IWYU pragma: export
#include <cub/warp/warp_load.cuh> // IWYU pragma: export
#include <cub/warp/warp_merge_sort.cuh> // IWYU pragma: export
#include <cub/warp/warp_reduce.cuh> // IWYU pragma: export
#include <cub/warp/warp_scan.cuh> // IWYU pragma: export
#include <cub/warp/warp_store.cuh> // IWYU pragma: export

// Util
#include "tuple.hpp"
#include <cub/util_debug.cuh> // IWYU pragma: export
#include <cub/util_device.cuh> // IWYU pragma: export
#include <cub/util_macro.cuh> // IWYU pragma: export
#include <cub/util_ptx.cuh> // IWYU pragma: export
#include <cub/util_type.cuh> // IWYU pragma: export
// These functions must be wrapped so they return
// hipError_t instead of cudaError_t
#include "util_allocator.hpp"
#include "util_device.hpp"
#include "util_temporary_storage.hpp"

#endif // HIPCUB_CUB_HIPCUB_HPP_
