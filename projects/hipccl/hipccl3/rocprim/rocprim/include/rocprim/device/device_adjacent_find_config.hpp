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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_CONFIG_HPP_

#include "config_types.hpp"

#include "detail/config/device_adjacent_find.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Type>
struct adjacent_find_config_selector
{
    using targets    = adjacent_find_targets;
    using param_type = adjacent_find_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        // Different configs if it is inplace.
        if constexpr(rocprim::is_arithmetic<Type>::value)
        {
            return adjacent_find_config_picker<Target, Type>();
        }
        else
        {
            return adjacent_find_config_params_base<Type>();
        }
    }

    template<class Target>
    constexpr adjacent_find_config_selector(Target) : params(picker_helper<Target>())
    {}
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_CONFIG_HPP_
