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

template<typename PartitionConfig, partition_subalgo, typename, typename>
struct wrapped_partition_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params = PartitionConfig{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::partition_two_way_predicate,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_partition_two_way_predicate_config<static_cast<unsigned int>(Arch),
                                                         KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::partition_two_way_flag,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_partition_two_way_flag_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::partition_flag,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_partition_flag_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::partition_predicate,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_partition_predicate_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::partition_three_way,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_partition_three_way_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config, partition_subalgo::select_flag, KeyType, empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_select_flag_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::select_predicate,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_select_predicate_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType, typename ValueType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::select_predicated_flag,
                                KeyType,
                                ValueType>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_select_predicated_flag_config<static_cast<unsigned int>(Arch),
                                                    KeyType,
                                                    ValueType>{};
    };
};

template<typename KeyType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::select_unique,
                                KeyType,
                                empty_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_select_unique_config<static_cast<unsigned int>(Arch), KeyType>{};
    };
};

template<typename KeyType, typename ValueType>
struct wrapped_partition_config<default_config,
                                partition_subalgo::select_unique_by_key,
                                KeyType,
                                ValueType>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr partition_config_params params
            = default_select_unique_by_key_config<static_cast<unsigned int>(Arch),
                                                  KeyType,
                                                  ValueType>{};
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<typename PartitionConfig, partition_subalgo SubAlgo, typename KeyType, typename ValueType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<PartitionConfig, SubAlgo, KeyType, ValueType>::architecture_config<
        Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::partition_two_way_predicate,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::partition_two_way_flag,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::partition_flag,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::partition_predicate,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::partition_three_way,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config, partition_subalgo::select_flag, KeyType, empty_type>::
        architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::select_predicate,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType, typename ValueType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::select_predicated_flag,
                             KeyType,
                             ValueType>::architecture_config<Arch>::params;

template<typename KeyType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::select_unique,
                             KeyType,
                             empty_type>::architecture_config<Arch>::params;

template<typename KeyType, typename ValueType>
template<target_arch Arch>
constexpr partition_config_params
    wrapped_partition_config<default_config,
                             partition_subalgo::select_unique_by_key,
                             KeyType,
                             ValueType>::architecture_config<Arch>::params;

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_CONFIG_HPP_
