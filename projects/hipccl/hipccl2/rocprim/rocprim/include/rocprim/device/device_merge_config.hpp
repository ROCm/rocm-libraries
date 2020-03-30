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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level merge primitives.
template<unsigned int BlockSize, unsigned int ItemsPerThread>
using merge_config = kernel_config<BlockSize, ItemsPerThread>;

namespace detail
{

template<class Key, class Value>
struct merge_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    // TODO Tune when merge-by-key is ready
    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_803<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};

template<class Key, class Value>
struct merge_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    // TODO Tune when merge-by-key is ready
    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_900<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_merge_config
    : select_arch<
        TargetArch,
        select_arch_case<803, merge_config_803<Key, Value>>,
        select_arch_case<900, merge_config_900<Key, Value>>,
        merge_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_
