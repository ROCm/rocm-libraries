// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"
#include "detail/config/device_adjacent_difference.hpp"
#include "detail/config/device_adjacent_difference_inplace.hpp"
#include "detail/device_config_helper.hpp"

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Specialization for user provided configuration
template<typename AdjacentDifferenceConfig, bool InPlace, typename>
struct wrapped_adjacent_difference_config
{
    static_assert(
        std::is_same<typename AdjacentDifferenceConfig::tag, adjacent_difference_config_tag>::value,
        "Config must be a specialization of struct template adjacent_difference_config");

    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr adjacent_difference_config_params params = AdjacentDifferenceConfig{};
    };
};

// Specialization for selecting the default configuration for in place
template<typename Value>
struct wrapped_adjacent_difference_config<default_config, true, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr adjacent_difference_config_params params
            = default_adjacent_difference_inplace_config<static_cast<unsigned int>(Arch), Value>{};
    };
};

// Specialization for selecting the default configuration for out of place
template<typename Value>
struct wrapped_adjacent_difference_config<default_config, false, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr adjacent_difference_config_params params
            = default_adjacent_difference_config<static_cast<unsigned int>(Arch), Value>{};
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class AdjacentDifferenceConfig, bool InPlace, class Value>
template<target_arch Arch>
constexpr adjacent_difference_config_params
    wrapped_adjacent_difference_config<AdjacentDifferenceConfig, InPlace, Value>::
        architecture_config<Arch>::params;
template<class Value>
template<target_arch Arch>
constexpr adjacent_difference_config_params
    wrapped_adjacent_difference_config<rocprim::default_config, true, Value>::architecture_config<
        Arch>::params;
template<class Value>
template<target_arch Arch>
constexpr adjacent_difference_config_params
    wrapped_adjacent_difference_config<rocprim::default_config, false, Value>::architecture_config<
        Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_
