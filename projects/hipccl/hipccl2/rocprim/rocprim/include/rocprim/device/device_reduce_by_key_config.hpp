// Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include "config_types.hpp"

#include "../block/block_load.hpp"
#include "../block/block_scan.hpp"
#include "../block/block_store.hpp"

#include "../config.hpp"

#include <algorithm>

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/**
 * \brief Configuration of device-level reduce-by-key operation.
 * 
 * \tparam BlockSize number of threads in a block.
 * \tparam ItemsPerThread number of items processed by each thread per tile. 
 * \tparam LoadKeysMethod method of loading keys
 * \tparam LoadValuesMethod method of loading values
 * \tparam ScanAlgorithm block level scan algorithm to use
 * \tparam TilesPerBlock number of tiles (`BlockSize` * `ItemsPerThread` items) to process per block
 * \tparam SizeLimit limit on the number of items for a single reduce_by_key kernel launch.
 */
template<unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_load_method    LoadKeysMethod   = block_load_method::block_load_transpose,
         block_load_method    LoadValuesMethod = block_load_method::block_load_transpose,
         block_scan_algorithm ScanAlgorithm    = block_scan_algorithm::using_warp_scan,
         unsigned int         TilesPerBlock    = 1,
         unsigned int         SizeLimit        = ROCPRIM_GRID_SIZE_LIMIT>
struct reduce_by_key_config_v2
{
    static constexpr unsigned int         block_size         = BlockSize;
    static constexpr unsigned int         tiles_per_block    = TilesPerBlock;
    static constexpr unsigned int         items_per_thread   = ItemsPerThread;
    static constexpr block_load_method    load_keys_method   = LoadKeysMethod;
    static constexpr block_load_method    load_values_method = LoadValuesMethod;
    static constexpr block_scan_algorithm scan_algorithm     = ScanAlgorithm;
    static constexpr unsigned int         size_limit         = SizeLimit;
};

/// \brief Legacy configuration of device-level reduce-by-key operation.
///
/// \deprecated Due to a new implementation the configuration options no longer match the algorithm
/// parameters. Use `reduce_by_key_config_v2` for the new parameters of the algorithm. Only a best
/// effort mapping is provided for these options, parameters not applicable to the new algorithm
/// are ignored.
///
/// \tparam ScanConfig - configuration of carry-outs scan kernel. Must be \p kernel_config.
/// \tparam ReduceConfig - configuration of the main reduce-by-key kernel. Must be \p kernel_config.
template<class ScanConfig, class ReduceConfig>
struct [[deprecated("use reduce_by_key_config_v2")]] reduce_by_key_config
    : reduce_by_key_config_v2<ReduceConfig::BlockSize, ReduceConfig::ItemsPerThread>
{
    /// \brief Configuration of carry-outs scan kernel.
    using scan = ScanConfig;
    /// \brief Configuration of the main reduce-by-key kernel.
    using reduce = ReduceConfig;
};

namespace detail
{

namespace reduce_by_key
{

template<typename Key, typename Value>
struct fallback_config
{
    static constexpr unsigned int size_memory_per_item = std::max(sizeof(Key), sizeof(Value));

    static constexpr unsigned int item_scale
        = static_cast<unsigned int>(ceiling_div(size_memory_per_item, 2 * sizeof(int)));

    static constexpr unsigned int items_per_thread = std::max(1u, 15u / item_scale);

    using type
        = reduce_by_key_config_v2<detail::limit_block_size<256U,
                                                           items_per_thread * size_memory_per_item,
                                                           ROCPRIM_WARP_SIZE_64>::value,
                                  items_per_thread,
                                  block_load_method::block_load_transpose,
                                  block_load_method::block_load_transpose,
                                  block_scan_algorithm::using_warp_scan,
                                  2>;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_config
    : std::conditional_t<std::max(sizeof(Key), sizeof(Value)) <= 16,
                         rocprim::reduce_by_key_config_v2<256,
                                                          15,
                                                          block_load_method::block_load_transpose,
                                                          block_load_method::block_load_transpose,
                                                          block_scan_algorithm::using_warp_scan,
                                                          sizeof(Value) < 16 ? 1 : 2>,
                         typename reduce_by_key::fallback_config<Key, Value>::type>
{};

} // namespace reduce_by_key

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
