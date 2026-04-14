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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include "detail/config/device_reduce_by_key.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key, class Value, class BinaryFunction>
struct reduce_by_key_config_selector
{
    using targets    = reduce_by_key_targets;
    using param_type = reduce_by_key_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        // Specialization for default config if types are not custom: instantiate the tuned config.
        if constexpr(rocprim::is_arithmetic<Key>::value && rocprim::is_arithmetic<Value>::value
                     && rocprim::detail::is_binary_functional<BinaryFunction>::value)
        {
            return reduce_by_key_config_picker<Target, Key, Value>();
        }
        else
        {
            return reduce_by_key_config_params_base<Key, Value>();
        }
    }

    template<class Target>
    constexpr reduce_by_key_config_selector(Target) : params(picker_helper<Target>())
    {}
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
