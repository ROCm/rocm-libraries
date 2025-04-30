// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../functional.hpp"
#include "detail/config/device_merge.hpp"
#include "detail/device_config_helper.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// generic struct that instantiates custom configurations
template<typename Config, typename, typename>
struct wrapped_merge_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr merge_config_params params = Config();
    };
};

// specialized for rocprim::default_config, which instantiates the default_ALGO_config
template<typename KeyType, typename ValueType>
struct wrapped_merge_config<default_config, KeyType, ValueType>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr merge_config_params params
            = default_merge_config<static_cast<unsigned int>(Arch), KeyType, ValueType>{};
    };
};

#ifndef DOXYGEN_DOCUMENTATION_BUILD
template<typename Config, typename Key, typename Value>
template<target_arch Arch>
constexpr merge_config_params
    wrapped_merge_config<Config, Key, Value>::architecture_config<Arch>::params;

template<class Key, class Value>
template<target_arch Arch>
constexpr merge_config_params
    wrapped_merge_config<rocprim::default_config, Key, Value>::architecture_config<Arch>::params;
#endif // DOXYGEN_DOCUMENTATION_BUILD

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_
