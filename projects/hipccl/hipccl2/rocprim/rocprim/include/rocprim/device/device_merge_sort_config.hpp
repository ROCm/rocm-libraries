// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level merge primitives.
///
/// \tparam BlockSize - block size used in merge sort.
template<unsigned int BlockSize>
using merge_sort_config = kernel_config<BlockSize, 1>;

namespace detail
{

template<class Key, class Value>
struct merge_sort_config_803
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(Key) + sizeof(Value)>::value>;
};

template<class Key>
struct merge_sort_config_803<Key, empty_type>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(Key)>::value>;
};

template<class Key, class Value>
struct merge_sort_config_900
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(Key) + sizeof(Value)>::value>;
};

template<class Key>
struct merge_sort_config_900<Key, empty_type>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(Key)>::value>;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_merge_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, merge_sort_config_803<Key, Value>>,
        select_arch_case<900, merge_sort_config_900<Key, Value>>,
        merge_sort_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
