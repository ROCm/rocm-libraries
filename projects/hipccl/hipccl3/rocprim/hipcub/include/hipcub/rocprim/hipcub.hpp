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

#ifndef HIPCUB_ROCPRIM_HIPCUB_HPP_
#define HIPCUB_ROCPRIM_HIPCUB_HPP_

#include "../config.hpp"

#include "util_type.hpp"
#include "util_ptx.hpp"
#include "thread/thread_operators.hpp"

// Iterator
#include "iterator/arg_index_input_iterator.hpp"
#include "iterator/counting_input_iterator.hpp"
#include "iterator/tex_obj_input_iterator.hpp"
#include "iterator/transform_input_iterator.hpp"

// Warp
#include "warp/warp_reduce.hpp"
#include "warp/warp_scan.hpp"

// Block
#include "block/block_discontinuity.hpp"
#include "block/block_exchange.hpp"
#include "block/block_histogram.hpp"
#include "block/block_load.hpp"
#include "block/block_radix_sort.hpp"
#include "block/block_reduce.hpp"
#include "block/block_scan.hpp"
#include "block/block_store.hpp"

// Device
#include "device/device_histogram.hpp"
#include "device/device_radix_sort.hpp"
#include "device/device_reduce.hpp"
#include "device/device_run_length_encode.hpp"
#include "device/device_scan.hpp"
#include "device/device_segmented_radix_sort.hpp"
#include "device/device_segmented_reduce.hpp"
#include "device/device_select.hpp"

#endif // HIPCUB_ROCPRIM_HIPCUB_HPP_
