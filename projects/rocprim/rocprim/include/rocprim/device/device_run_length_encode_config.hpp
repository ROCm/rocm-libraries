// Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "config_types.hpp"
#include "device_reduce_by_key_config.hpp"

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../type_traits.hpp"
#include "detail/config/device_run_length_encode.hpp"
#include "detail/config/device_run_length_encode_non_trivial.hpp"

#include <type_traits>

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level run-length encoding operation.
///
/// \tparam ReduceByKeyConfig configuration of device-level reduce-by-key operation.
/// Must be \p reduce_by_key_config or \p default_config.
/// \tparam SelectConfig configuration of device-level select operation.
/// Must be \p select_config or \p default_config.
template<typename ReduceByKeyConfig, typename SelectConfig = default_config>
struct run_length_encode_config
{
    /// \brief Configuration of device-level reduce-by-key operation.
    using reduce_by_key = ReduceByKeyConfig;
    /// \brief Configuration of device-level select operation.
    using select = SelectConfig;
};

namespace detail
{

template<class Config>
struct select_reduce_by_key_config
{
    using type = typename Config::reduce_by_key;
};

template<>
struct select_reduce_by_key_config<rocprim::default_config>
{
    using type = rocprim::default_config;
};

template<class Key, class Value, class BinaryFunction>
struct run_length_encode_config_selector
{
    using targets    = run_length_encode_targets;
    using param_type = reduce_by_key_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        // Specialization for default config if types are not custom: instantiate the tuned config.
        if constexpr(rocprim::is_arithmetic<Key>::value && rocprim::is_arithmetic<Value>::value
                     && rocprim::detail::is_binary_functional<BinaryFunction>::value)
        {
            return run_length_encode_config_picker<Target, Key, Value>();
        }
        else
        {
            return reduce_by_key_config_params_base<Key, Value>();
        }
    }

    template<class Target>
    constexpr run_length_encode_config_selector(Target) : params(picker_helper<Target>())
    {}
};

template<class Config>
struct convert_to_non_trivial_config
{
    using type = Config; // identity by default
};

// Only convert run_length_encode_config
template<typename ReduceByKeyConfig, typename SelectConfig>
struct convert_to_non_trivial_config<run_length_encode_config<ReduceByKeyConfig, SelectConfig>>
{
    static constexpr unsigned int block_size       = ReduceByKeyConfig::block_size;
    static constexpr unsigned int items_per_thread = ReduceByKeyConfig::items_per_thread;

    static constexpr block_load_method    load_input_method = ReduceByKeyConfig::load_keys_method;
    static constexpr block_scan_algorithm scan_algorithm    = ReduceByKeyConfig::scan_algorithm;

    using type
        = non_trivial_runs_config<block_size, items_per_thread, load_input_method, scan_algorithm>;
};

template<>
struct convert_to_non_trivial_config<rocprim::default_config>
{
    using type = rocprim::default_config;
};

template<class Key>
struct run_length_encode_non_trivial_config_selector
{
    using targets    = run_length_encode_non_trivial_targets;
    using param_type = non_trivial_runs_config_params;

    param_type params;

    template<class Target>
    constexpr param_type picker_helper()
    {
        // Specialization for default config if types are not custom: instantiate the tuned config.
        if constexpr(rocprim::is_arithmetic<Key>::value)
        {
            return run_length_encode_non_trivial_config_picker<Target, Key>();
        }
        else
        {
            return non_trivial_runs_config_params_base<Key>();
        }
    }

    template<class Target>
    constexpr run_length_encode_non_trivial_config_selector(Target)
        : params(picker_helper<Target>())
    {}
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_
