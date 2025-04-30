// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_CONFIG_HPP_

#include "config_types.hpp"

#include "device_nth_element_config.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level partial sort.
///
/// \tparam NthElementConfig configuration of device-level nth element operation.
/// Must be \p nth_element_config or \p default_config.
/// \tparam MergeSortConfig configuration of device-level merge sort operation.
/// Must be \p merge_sort_config or \p default_config.
/// \tparam RadixSortConfig configuration of device-level radix sort operation.
/// Must be \p radix_sort_config or \p default_config.
template<class NthElementConfig,
         class MergeSortConfig = default_config,
         class RadixSortConfig = default_config>
struct partial_sort_config
{
    /// \brief Configuration of device-level nth element operation.
    using nth_element = NthElementConfig;
    /// \brief Configuration of device-level merge sort operation.
    using merge_sort = MergeSortConfig;
    /// \brief Configuration of device-level radix sort operation.
    using radix_sort = RadixSortConfig;
};

namespace detail
{

template<typename Type>
using default_partial_sort_config
    = partial_sort_config<default_config, default_config, default_config>;

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_PARTIAL_SORT_CONFIG_HPP_
