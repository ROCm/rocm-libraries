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

template<typename HistogramConfig>
constexpr histogram_config_params wrap_histogram_config()
{
    return histogram_config_params{
        {HistogramConfig::histogram::block_size,
         HistogramConfig::histogram::items_per_thread,
         HistogramConfig::histogram::size_limit},
        HistogramConfig::max_grid_size,
        HistogramConfig::shared_impl_max_bins,
        HistogramConfig::shared_impl_histograms
    };
}

template<typename HistogramConfig, typename, unsigned int, unsigned int>
struct wrapped_histogram_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr histogram_config_params params = wrap_histogram_config<HistogramConfig>();
    };
};

template<typename Sample, unsigned int Channels, unsigned int ActiveChannels>
struct wrapped_histogram_config<default_config, Sample, Channels, ActiveChannels>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr histogram_config_params params
            = wrap_histogram_config<default_histogram_config<static_cast<unsigned int>(Arch),
                                                             Sample,
                                                             Channels,
                                                             ActiveChannels>>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename HistogramConfig,
         typename Sample,
         unsigned int Channels,
         unsigned int ActiveChannels>
template<target_arch Arch>
constexpr histogram_config_params
    wrapped_histogram_config<HistogramConfig, Sample, Channels, ActiveChannels>::
        architecture_config<Arch>::params;

template<typename Sample, unsigned int Channels, unsigned int ActiveChannels>
template<target_arch Arch>
constexpr histogram_config_params
    wrapped_histogram_config<default_config, Sample, Channels, ActiveChannels>::architecture_config<
        Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
