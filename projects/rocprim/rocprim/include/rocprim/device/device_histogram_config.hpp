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

#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/config/device_histogram.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config, class Selector, class Target>
struct histogram_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.histogram_config.block_size;
};

template<class Config, class Selector, class Target>
struct histogram_global_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.histogram_global_config.block_size;
};

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_selector
{
    using targets    = histogram_targets;
    using param_type = histogram_config_params;

    param_type params;

    template<class Target>
    constexpr histogram_config_selector(Target)
        : params(histogram_config_picker<Target, Sample, Channels, ActiveChannels>())
    {}
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
