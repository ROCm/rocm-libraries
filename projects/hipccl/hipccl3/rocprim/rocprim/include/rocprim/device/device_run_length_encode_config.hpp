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

#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level run-length encoding operation.
///
/// \tparam ReduceByKeyConfig - configuration of device-level reduce-by-key operation.
/// Must be \p reduce_by_key_config or \p default_config.
/// \tparam SelectConfig - configuration of device-level select operation.
/// Must be \p select_config or \p default_config.
template<
    class ReduceByKeyConfig,
    class SelectConfig = default_config
>
struct run_length_encode_config
{
    /// \brief Configuration of device-level reduce-by-key operation.
    using reduce_by_key = ReduceByKeyConfig;
    /// \brief Configuration of device-level select operation.
    using select = SelectConfig;
};

namespace detail
{

using default_run_length_encode_config = run_length_encode_config<default_config, default_config>;

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_
