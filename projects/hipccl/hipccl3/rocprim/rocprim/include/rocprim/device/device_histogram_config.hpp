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

#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level histogram operation.
///
/// \tparam HistogramConfig - configuration of histogram kernel. Must be \p kernel_config.
/// \tparam MaxGridSize - maximim number of blocks to launch.
/// \tparam SharedImplMaxBins - maximum total number of bins for all active channels
/// for the shared memory histogram implementation (samples -> shared memory bins -> global memory bins),
/// when exceeded the global memory implementation is used (samples -> global memory bins).
template<
    class HistogramConfig,
    unsigned int MaxGridSize = 1024,
    unsigned int SharedImplMaxBins = 2048
>
struct histogram_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using histogram = HistogramConfig;

    static constexpr unsigned int max_grid_size = MaxGridSize;
    static constexpr unsigned int shared_impl_max_bins = SharedImplMaxBins;
#endif
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<
    class HistogramConfig,
    unsigned int MaxGridSize,
    unsigned int SharedImplMaxBins
> constexpr unsigned int
histogram_config<HistogramConfig, MaxGridSize, SharedImplMaxBins>::max_grid_size;
template<
    class HistogramConfig,
    unsigned int MaxGridSize,
    unsigned int SharedImplMaxBins
> constexpr unsigned int
histogram_config<HistogramConfig, MaxGridSize, SharedImplMaxBins>::shared_impl_max_bins;
#endif

namespace detail
{

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_803
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(10u / Channels / item_scale, 1u)>>;
};

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_900
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(8u / Channels / item_scale, 1u)>>;
};

template<unsigned int TargetArch, class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct default_histogram_config
    : select_arch<
        TargetArch,
        select_arch_case<803, histogram_config_803<Sample, Channels, ActiveChannels> >,
        select_arch_case<900, histogram_config_900<Sample, Channels, ActiveChannels> >,
        histogram_config_900<Sample, Channels, ActiveChannels>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
