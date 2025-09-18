// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template<typename ReduceByKeyConfig,
         typename KeyType,
         typename AccumulatorType,
         typename BinaryFunction>
struct wrapped_trivial_runs_config
    : wrapped_reduce_by_key_config<ReduceByKeyConfig, KeyType, AccumulatorType, BinaryFunction>
{};

template<typename ReduceByKeyConfig,
         typename SelectConfig,
         typename KeyType,
         typename AccumulatorType,
         typename BinaryFunction>
struct wrapped_trivial_runs_config<
    rocprim::run_length_encode_config<ReduceByKeyConfig, SelectConfig>,
    KeyType,
    AccumulatorType,
    BinaryFunction>
    : wrapped_reduce_by_key_config<ReduceByKeyConfig, KeyType, AccumulatorType, BinaryFunction>
{};

template<typename KeyType,
         typename AccumulatorType,
         typename BinaryFunction,
         typename Enable = void>
struct wrapped_trivial_runs_impl
    : wrapped_reduce_by_key_impl<KeyType, AccumulatorType, BinaryFunction, Enable>
{};

template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
struct wrapped_trivial_runs_impl<
    KeyType,
    AccumulatorType,
    BinaryFunction,
    std::enable_if_t<is_arithmetic<KeyType>::value && is_arithmetic<AccumulatorType>::value
                     && is_binary_functional<BinaryFunction>::value>>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_by_key_config_params params
            = default_trivial_runs_config<static_cast<unsigned int>(Arch),
                                          KeyType,
                                          AccumulatorType>{};
    };
};

template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
struct wrapped_trivial_runs_config<default_config, KeyType, AccumulatorType, BinaryFunction>
    : wrapped_trivial_runs_impl<KeyType, AccumulatorType, BinaryFunction>
{};

// Wrap around run_length_encode_config and the newly added non_trivial_runs_config for the
// run_length_encode_non_trivial_runs algorithm. Three cases are considered for selecting
// the appropriate config:
//
//   - When a run_length_encode_config struct is passed as argument, an specialization of
//     this struct takes care of mapping the parameters of that config to the newly added
//     non_trivial_runs_config.
//
//   - When a default config is passed, another specialization takes care of using the
//     default set up of non_trivial_runs_config.
//
//   - When a non_trivial_runs_config is passed, the params are set from this config.
//
template<typename RLENonTrivialRunsConfig, typename>
struct wrapped_non_trivial_runs_config
{
    static_assert(std::is_same<typename RLENonTrivialRunsConfig::tag,
                               detail::non_trivial_runs_config_tag>::value,
                  "Config must be a specialization of struct template non_trivial_runs_config");

    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr non_trivial_runs_config_params params = RLENonTrivialRunsConfig{};
    };
};

template<typename ReduceByKeyConfig, typename SelectConfig, typename InputType>
struct wrapped_non_trivial_runs_config<
    rocprim::run_length_encode_config<ReduceByKeyConfig, SelectConfig>,
    InputType>
{
    template<target_arch Arch>
    struct architecture_config
    {
        // Mapping <reduce_by_key_config, select_config> to non_trivial_runs_config.
        // Beware that this mapping may impact performance of executions of
        // run_length_encode_non_trivial_runs with the former run_length_encode_config,
        // as it may not be the best for all cases.
        static constexpr unsigned int block_size       = ReduceByKeyConfig::block_size;
        static constexpr unsigned int items_per_thread = ReduceByKeyConfig::items_per_thread;

        static constexpr block_load_method load_input_method = ReduceByKeyConfig::load_keys_method;
        static constexpr block_scan_algorithm scan_algorithm = ReduceByKeyConfig::scan_algorithm;

        static constexpr non_trivial_runs_config_params params
            = non_trivial_runs_config<block_size,
                                      items_per_thread,
                                      load_input_method,
                                      scan_algorithm>{};
    };
};

// Generic for default config: instantiate base config.
template<typename InputType, typename Enable = void>
struct wrapped_non_trivial_runs_impl
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr non_trivial_runs_config_params params =
            typename default_non_trivial_runs_config_base<InputType>::type{};
    };
};

// Specialization for default config if types are arithmetic or half/bfloat16-precision
// floating point types: instantiate the tuned config.
template<typename InputType>
struct wrapped_non_trivial_runs_impl<InputType,
                                     std::enable_if_t<rocprim::is_arithmetic<InputType>::value>>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr non_trivial_runs_config_params params
            = default_non_trivial_runs_config<static_cast<unsigned int>(Arch), InputType>{};
    };
};

// Specialization for default config.
template<typename InputType>
struct wrapped_non_trivial_runs_config<default_config, InputType>
    : wrapped_non_trivial_runs_impl<InputType>
{};

#ifndef DOXYGEN_DOCUMENTATION_BUILD

template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
template<target_arch Arch>
constexpr reduce_by_key_config_params wrapped_trivial_runs_impl<
    KeyType,
    AccumulatorType,
    BinaryFunction,
    std::enable_if_t<is_arithmetic<KeyType>::value && is_arithmetic<AccumulatorType>::value
                     && is_binary_functional<BinaryFunction>::value>>::architecture_config<Arch>::
    params;

template<typename RLENonTrivialRunsConfig, typename InputType>
template<target_arch Arch>
constexpr non_trivial_runs_config_params
    wrapped_non_trivial_runs_config<RLENonTrivialRunsConfig,
                                    InputType>::architecture_config<Arch>::params;

template<typename ReduceByKeyConfig, typename SelectConfig, typename InputType>
template<target_arch Arch>
constexpr non_trivial_runs_config_params wrapped_non_trivial_runs_config<
    rocprim::run_length_encode_config<ReduceByKeyConfig, SelectConfig>,
    InputType>::architecture_config<Arch>::params;

template<typename InputType, typename Enable>
template<target_arch Arch>
constexpr non_trivial_runs_config_params
    wrapped_non_trivial_runs_impl<InputType, Enable>::architecture_config<Arch>::params;

template<typename InputType>
template<target_arch Arch>
constexpr non_trivial_runs_config_params wrapped_non_trivial_runs_impl<
    InputType,
    std::enable_if_t<is_arithmetic<InputType>::value>>::architecture_config<Arch>::params;

#endif // DOXYGEN_DOCUMENTATION_BUILD

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_
