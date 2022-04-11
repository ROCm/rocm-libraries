// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level adjacent_difference primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread
/// \tparam LoadMethod - method for loading input values
/// \tparam StoreMethod - method for storing values
/// \tparam SizeLimit - limit on the number of items for a single adjacent_difference kernel launch.
/// Larger input sizes will be broken up to multiple kernel launches.
template <unsigned int       BlockSize,
          unsigned int       ItemsPerThread,
          block_load_method  LoadMethod  = block_load_method::block_load_transpose,
          block_store_method StoreMethod = block_store_method::block_store_transpose,
          unsigned int       SizeLimit   = ROCPRIM_GRID_SIZE_LIMIT>
struct adjacent_difference_config : kernel_config<BlockSize, ItemsPerThread, SizeLimit>
{
    static constexpr block_load_method  load_method  = LoadMethod;
    static constexpr block_store_method store_method = StoreMethod;
};

namespace detail
{

template <class Value>
struct adjacent_difference_config_fallback
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = adjacent_difference_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template <unsigned int TargetArch, class Value>
struct default_adjacent_difference_config
    : select_arch<TargetArch, adjacent_difference_config_fallback<Value>>
{
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_
