// Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// device transform does not have config tuning
template<unsigned int arch, class value_type>
struct default_transform_config : default_transform_config_base<value_type>
{};

template<typename TransformConfig>
constexpr transform_config_params wrap_transform_config()
{
    return transform_config_params{
        {
         TransformConfig::block_size,
         TransformConfig::items_per_thread,
         TransformConfig::size_limit,
         }
    };
}

template<typename TransformConfig, typename>
struct wrapped_transform_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr transform_config_params params = wrap_transform_config<TransformConfig>();
    };
};

template<typename Value>
struct wrapped_transform_config<default_config, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr transform_config_params params = wrap_transform_config<
            default_transform_config<static_cast<unsigned int>(Arch), Value>>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename TransformConfig, typename Value>
template<target_arch Arch>
constexpr transform_config_params
    wrapped_transform_config<TransformConfig, Value>::architecture_config<Arch>::params;

template<typename Value>
template<target_arch Arch>
constexpr transform_config_params
    wrapped_transform_config<default_config, Value>::architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_TRANSFORM_CONFIG_HPP_
