// Copyright (c) 2018-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"
#include "../block/block_scan.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level scan primitives.
///
/// \tparam BlockSize - number of threads in a block.
/// \tparam ItemsPerThread - number of items processed by each thread.
/// \tparam UseLookback - whether to use lookback scan or reduce-then-scan algorithm.
/// \tparam BlockLoadMethod - method for loading input values.
/// \tparam StoreLoadMethod - method for storing values.
/// \tparam BlockScanMethod - algorithm for block scan.
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool UseLookback,
    ::rocprim::block_load_method BlockLoadMethod,
    ::rocprim::block_store_method BlockStoreMethod,
    ::rocprim::block_scan_algorithm BlockScanMethod
>
struct scan_config
{
    /// \brief Number of threads in a block.
    static constexpr unsigned int block_size = BlockSize;
    /// \brief Number of items processed by each thread.
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    /// \brief Whether to use lookback scan or reduce-then-scan algorithm.
    static constexpr bool use_lookback = UseLookback;
    /// \brief Method for loading input values.
    static constexpr block_load_method block_load_method = BlockLoadMethod;
    /// \brief Method for storing values.
    static constexpr block_store_method block_store_method = BlockStoreMethod;
    /// \brief Algorithm for block scan.
    static constexpr block_scan_algorithm block_scan_method = BlockScanMethod;
};

namespace detail
{

template<class Value>
struct scan_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value)>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct scan_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value)>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<unsigned int TargetArch, class Value>
struct default_scan_config
    : select_arch<
        TargetArch,
        select_arch_case<803, scan_config_803<Value>>,
        select_arch_case<900, scan_config_900<Value>>,
        scan_config_900<Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_CONFIG_HPP_
