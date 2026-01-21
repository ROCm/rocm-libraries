// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/config/device_merge_sort_block_merge.hpp"
#include "detail/config/device_merge_sort_block_sort.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// \brief Kernel parameters for device merge sort.
struct merge_sort_config_params
{
    merge_sort_block_sort_config_params  block_sort_config;
    merge_sort_block_merge_config_params block_merge_config;
};

} // namespace detail

/// \brief Configuration of device-level merge primitives.
///
/// \tparam SortBlockSize block size in the block-sort step
/// \tparam SortItemsPerThread ItemsPerThread in the block-sort step
/// \tparam MergeOddevenBlockSize block size in the block merge step using oddeven impl
///         (used when input_size < MinInputSizeMergepath)
/// \tparam MergeMergepathPartitionBlockSize block size of the partition kernel in the block merge
///         step using mergepath impl
/// \tparam MergeMergepathBlockSize block size in the block merge step using mergepath impl
/// \tparam MergeMergepathItemsPerThread ItemsPerThread in the block merge step using
///         mergepath impl
/// \tparam MinInputSizeMergepath breakpoint of input-size to use mergepath impl for
///         block merge step
template<unsigned int MergeOddevenBlockSize            = 512,
         unsigned int SortBlockSize                    = MergeOddevenBlockSize,
         unsigned int SortItemsPerThread               = 1,
         unsigned int MergeMergepathPartitionBlockSize = 128,
         unsigned int MergeMergepathBlockSize          = 128,
         unsigned int MergeMergepathItemsPerThread     = 4,
         unsigned int MinInputSizeMergepath            = (1 << 17) + 70000>
struct merge_sort_config : detail::merge_sort_config_params
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \remark Here we map the public parameters to our internal structure.
    using block_sort_config
        = detail::merge_sort_block_sort_config<SortBlockSize,
                                               SortItemsPerThread,
                                               block_sort_algorithm::stable_merge_sort>;
    using block_merge_config = detail::merge_sort_block_merge_config<MergeOddevenBlockSize,
                                                                     1,
                                                                     MinInputSizeMergepath,
                                                                     MergeMergepathBlockSize,
                                                                     MergeMergepathBlockSize,
                                                                     MergeMergepathItemsPerThread>;
    constexpr merge_sort_config()
        : detail::merge_sort_config_params{block_sort_config(), block_merge_config()} {};
#endif
};

namespace detail
{

// Sub algorithm block_merge:
template<class Config, class Selector, class Target>
struct merge_oddeven_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.merge_oddeven_config.block_size;
};

template<class Config, class Selector, class Target>
struct merge_mergepath_partition_config_static_selector
{
    static constexpr auto block_size = target_config<Config, Selector, Target>::params
                                           .merge_mergepath_partition_config.block_size;
};

template<class Config, class Selector, class Target>
struct merge_mergepath_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.merge_mergepath_config.block_size;
};

template<class Key, class Value>
struct merge_sort_block_merge_config_selector
{
    using targets    = merge_sort_block_merge_targets;
    using param_type = merge_sort_block_merge_config_params;

    param_type params;

    template<class Target>
    constexpr merge_sort_block_merge_config_selector(Target)
        : params(merge_sort_block_merge_config_picker<Target, Key, Value>())
    {}
};

// Sub-algorithm block_sort:
template<class Key, class Value>
struct merge_sort_block_sort_config_selector
{
    using targets    = merge_sort_block_sort_targets;
    using param_type = merge_sort_block_sort_config_params;

    param_type params;

    template<class Target>
    constexpr merge_sort_block_sort_config_selector(Target)
        : params(merge_sort_block_sort_config_picker<Target, Key, Value>())
    {}
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
