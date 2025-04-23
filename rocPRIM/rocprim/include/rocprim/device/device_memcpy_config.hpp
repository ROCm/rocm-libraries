// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_MEMCPY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MEMCPY_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"
#include "detail/config/device_batch_copy.hpp"
#include "detail/config/device_batch_memcpy.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Specialization for user provided configuration
template<typename BatchMemcpyConfig, typename, bool>
struct wrapped_batch_memcpy_config
{
    static_assert(std::is_same<typename BatchMemcpyConfig::tag, batch_memcpy_config_tag>::value,
                  "Config must be a specialization of struct template batch_memcpy_config");

    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr batch_memcpy_config_params params = BatchMemcpyConfig{};
    };
};

// Specialization for selecting the default configuration for out of place
template<typename Value, bool IsMemCpy>
struct wrapped_batch_memcpy_config<default_config, Value, IsMemCpy>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr batch_memcpy_config_params params
            = IsMemCpy ? (batch_memcpy_config_params)
                      default_batch_memcpy_config<static_cast<unsigned int>(Arch), Value>{}
                       : (batch_memcpy_config_params)
                           default_batch_copy_config<static_cast<unsigned int>(Arch), Value>{};
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class BatchMemcpyConfig, class Value, bool IsMemCpy>
template<target_arch Arch>
constexpr batch_memcpy_config_params
    wrapped_batch_memcpy_config<BatchMemcpyConfig, Value, IsMemCpy>::architecture_config<
        Arch>::params;
template<class Value, bool IsMemCpy>
template<target_arch Arch>
constexpr batch_memcpy_config_params
    wrapped_batch_memcpy_config<rocprim::default_config, Value, IsMemCpy>::architecture_config<
        Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif
