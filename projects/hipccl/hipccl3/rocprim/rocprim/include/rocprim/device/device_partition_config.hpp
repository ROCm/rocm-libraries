// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_CONFIG_HPP_

#include "detail/config/device_partition_flag.hpp"
#include "detail/config/device_partition_predicate.hpp"
#include "detail/config/device_partition_three_way.hpp"
#include "detail/config/device_partition_two_way_flag.hpp"
#include "detail/config/device_partition_two_way_predicate.hpp"
#include "detail/config/device_select_flag.hpp"
#include "detail/config/device_select_predicate.hpp"
#include "detail/config/device_select_predicated_flag.hpp"
#include "detail/config/device_select_unique.hpp"
#include "detail/config/device_select_unique_by_key.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<partition_subalgo Algo>
constexpr auto algo_target_type()
{
    if constexpr(Algo == partition_subalgo::partition_two_way_predicate)
    {
        return type_identity<partition_two_way_predicate_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::partition_two_way_flag)
    {
        return type_identity<partition_two_way_flag_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::partition_flag)
    {
        return type_identity<partition_flag_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::partition_predicate)
    {
        return type_identity<partition_predicate_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::partition_three_way)
    {
        return type_identity<partition_three_way_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::select_flag)
    {
        return type_identity<select_flag_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::select_predicate)
    {
        return type_identity<select_predicate_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::select_predicated_flag)
    {
        return type_identity<select_predicated_flag_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::select_unique)
    {
        return type_identity<select_unique_targets>{};
    }
    else if constexpr(Algo == partition_subalgo::select_unique_by_key)
    {
        return type_identity<select_unique_by_key_targets>{};
    }
}

template<partition_subalgo SubAlgo, class Key, class Value, class Flag>
struct partition_config_selector
{
    using targets    = typename decltype(algo_target_type<SubAlgo>())::type;
    using param_type = partition_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        if constexpr(SubAlgo == partition_subalgo::partition_two_way_predicate)
        {
            return partition_two_way_predicate_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::partition_two_way_flag)
        {
            return partition_two_way_flag_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::partition_flag)
        {
            return partition_flag_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::partition_predicate)
        {
            return partition_predicate_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::partition_three_way)
        {
            return partition_three_way_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::select_flag)
        {
            return select_flag_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::select_predicate)
        {
            return select_predicate_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::select_predicated_flag)
        {
            return select_predicated_flag_config_picker<Target, Key, Flag>();
        }
        else if constexpr(SubAlgo == partition_subalgo::select_unique)
        {
            return select_unique_config_picker<Target, Key>();
        }
        else if constexpr(SubAlgo == partition_subalgo::select_unique_by_key)
        {
            return select_unique_by_key_config_picker<Target, Key, Value>();
        }
    }

    template<class Target>
    constexpr partition_config_selector(Target) : params(picker_helper<Target>())
    {}
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_CONFIG_HPP_
