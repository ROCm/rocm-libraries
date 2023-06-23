// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_CONFIG_HPP_

#include "../config.hpp"

#include "config_types.hpp"
#include "detail/config/device_binary_search.hpp"
#include "detail/config/device_lower_bound.hpp"
#include "detail/config/device_upper_bound.hpp"
#include "detail/device_config_helper.hpp"
#include "device_transform_config.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Value, class Output>
struct default_config_for_binary_search
{};

template<class Value, class Output>
struct default_config_for_upper_bound
{};

template<class Value, class Output>
struct default_config_for_lower_bound
{};

template<class Unused, class Value, class Output>
struct wrapped_transform_config<default_config_for_binary_search<Value, Output>, Unused>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr transform_config_params params = wrap_transform_config<
            default_binary_search_config<static_cast<unsigned int>(Arch), Value, Output>>();
    };
};

template<class Unused, class Value, class Output>
struct wrapped_transform_config<default_config_for_upper_bound<Value, Output>, Unused>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr transform_config_params params = wrap_transform_config<
            default_upper_bound_config<static_cast<unsigned int>(Arch), Value, Output>>();
    };
};

template<class Unused, class Value, class Output>
struct wrapped_transform_config<default_config_for_lower_bound<Value, Output>, Unused>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr transform_config_params params = wrap_transform_config<
            default_lower_bound_config<static_cast<unsigned int>(Arch), Value, Output>>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class Unused, class Value, class Output>
template<target_arch Arch>
constexpr transform_config_params
    wrapped_transform_config<default_config_for_binary_search<Value, Output>,
                             Unused>::architecture_config<Arch>::params;
template<class Unused, class Value, class Output>
template<target_arch Arch>
constexpr transform_config_params
    wrapped_transform_config<default_config_for_upper_bound<Value, Output>,
                             Unused>::architecture_config<Arch>::params;
template<class Unused, class Value, class Output>
template<target_arch Arch>
constexpr transform_config_params
    wrapped_transform_config<default_config_for_lower_bound<Value, Output>,
                             Unused>::architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_CONFIG_HPP_
