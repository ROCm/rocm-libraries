// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/grouped_convolution_forward_depthwise_pipeline.hpp"

namespace ck_tile {

// Stub epilogue for depthwise path (no GEMM epilogue needed)
template <typename OutDataType_>
struct DepthwiseNullEpilogue
{
    using DsLayout      = ck_tile::tuple<>;
    using DsDataType    = ck_tile::tuple<>;
    using ODataType     = OutDataType_;
    using CDElementwise = element_wise::PassThrough;
};

struct GroupedConvFwdDepthwiseResult
{
    float best_time = std::numeric_limits<float>::max();
    std::string best_config;
    index_t best_instance_idx = -1;
    index_t valid_count       = 0;
    index_t total_count       = 0;
    float best_tflops         = 0.0f;
    float best_gb_per_sec     = 0.0f;
};

namespace depthwise_dispatch_detail {

struct InstanceRunResult
{
    float time_ms     = std::numeric_limits<float>::max();
    bool is_supported = false;
    std::string config_name;
    float tflops     = 0.0f;
    float gb_per_sec = 0.0f;
};

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType,
          index_t TileH,
          index_t TileW,
          index_t FilterSize,
          index_t DilationH,
          index_t DilationW,
          index_t StrideH,
          index_t StrideW,
          index_t PadH,
          index_t PadW,
          index_t NBatch,
          index_t SubTileH,
          index_t SubTileW,
          index_t InVecSize,
          index_t OutVecSize>
InstanceRunResult try_instance(const GroupedConvFwdHostArgs<>& host_args,
                               const stream_config& s,
                               const std::size_t flop,
                               const std::size_t num_byte)
{
    InstanceRunResult result;

    using DwTraits = DepthwiseConvFwdTraits<InDataType,
                                            WeiDataType,
                                            AccDataType,
                                            OutDataType,
                                            64,
                                            TileH,
                                            TileW,
                                            FilterSize,
                                            FilterSize,
                                            StrideH,
                                            StrideW,
                                            DilationH,
                                            DilationW,
                                            PadH,
                                            PadW,
                                            NBatch,
                                            SubTileH,
                                            SubTileW,
                                            InVecSize,
                                            OutVecSize>;

    using DwPipeline = DepthwiseConvFwdPipeline<DwTraits>;

    using ConvTraitsType = GroupedConvTraits<2,
                                             ConvolutionSpecialization::Default,
                                             void,
                                             void,
                                             tuple<>,
                                             void,
                                             1,
                                             1,
                                             1,
                                             1,
                                             false,
                                             false,
                                             DwTraits>;

    using Kernel = GroupedConvolutionForwardKernel<ConvTraitsType,
                                                   void,
                                                   DwPipeline,
                                                   DepthwiseNullEpilogue<OutDataType>>;

    auto kargs = Kernel::MakeKernelArgs(host_args);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        return result;
    }

    result.config_name = concat('_',
                                "depthwise_conv_fwd",
                                DwTraits::BlockSize,
                                concat('x', DwTraits::FilterH, DwTraits::FilterW),
                                concat('x', DwTraits::StrideH, DwTraits::StrideW),
                                concat('x', DwTraits::DilationH, DwTraits::DilationW),
                                concat('x', DwTraits::PadH, DwTraits::PadW),
                                DwTraits::NBatch,
                                concat('x', DwTraits::SubTileH, DwTraits::SubTileW),
                                concat('x', DwTraits::InVectorSize, DwTraits::OutVectorSize),
                                concat('x', DwTraits::TileOutH, DwTraits::TileOutW));

    const auto grids  = Kernel::GridSize(kargs);
    const auto blocks = Kernel::BlockSize();

    const float time_ms = launch_kernel(s, make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    result.is_supported = true;
    result.time_ms      = time_ms;

    if(flop > 0 && time_ms > 0)
    {
        result.tflops     = static_cast<float>(flop) / 1.E9 / time_ms;
        result.gb_per_sec = static_cast<float>(num_byte) / 1.E6 / time_ms;
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Perf: " << std::setw(10) << std::fixed << std::setprecision(6) << time_ms
                  << " ms, " << std::setprecision(4) << result.tflops << " TFlops, "
                  << std::setprecision(3) << result.gb_per_sec << " GB/s, " << result.config_name
                  << std::endl;
    }

    return result;
}

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
GroupedConvFwdDepthwiseResult run_all_instances(const GroupedConvFwdHostArgs<>& host_args,
                                                const stream_config& s,
                                                const std::size_t flop,
                                                const std::size_t num_byte)
{
    GroupedConvFwdDepthwiseResult best;

    if(s.log_level_ > 0)
    {
        std::cout << "\n=== Depthwise conv fwd: testing all instances ===" << std::endl;
    }

    auto process_result = [&](const InstanceRunResult& result) {
        if(result.is_supported)
        {
            best.valid_count++;
            if(result.time_ms < best.best_time)
            {
                best.best_time         = result.time_ms;
                best.best_config       = result.config_name;
                best.best_instance_idx = best.total_count;
                best.best_tflops       = result.tflops;
                best.best_gb_per_sec   = result.gb_per_sec;
            }
        }
        best.total_count++;
    };

// Parameters: TileH, TileW, Filter (square), StrH, StrW, PadH, PadW,
//             NBatch, SubTileH, SubTileW, InVecSize, OutVecSize
// Dilation is hardcoded to 1x1; expand when non-unit dilation is supported
#define CK_TILE_DEPTHWISE_TRY_INSTANCE(                                              \
    TileH, TileW, Filter, StrH, StrW, PadH, PadW, NBatch, SubH, SubW, InVec, OutVec) \
    process_result(try_instance<InDataType,                                          \
                                WeiDataType,                                         \
                                AccDataType,                                         \
                                OutDataType,                                         \
                                TileH,                                               \
                                TileW,                                               \
                                Filter,                                              \
                                1,                                                   \
                                1,                                                   \
                                StrH,                                                \
                                StrW,                                                \
                                PadH,                                                \
                                PadW,                                                \
                                NBatch,                                              \
                                SubH,                                                \
                                SubW,                                                \
                                InVec,                                               \
                                OutVec>(host_args, s, flop, num_byte))

    // Instance table generated from:
    // experimental/grouped_convolution_tile_instances/configs/forward/profiler/ngchw_depthwise.conf
    // To regenerate: python3 generate_depthwise_instances.py
#include "../../../experimental/grouped_convolution_tile_instances/instances/forward/depthwise_fwd_instances.inc"

#undef CK_TILE_DEPTHWISE_TRY_INSTANCE

    return best;
}

} // namespace depthwise_dispatch_detail

/// @brief Dispatch depthwise forward instances. Returns empty result on ineligible problems.
template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
GroupedConvFwdDepthwiseResult
grouped_conv_fwd_depthwise_dispatch(const GroupedConvFwdHostArgs<>& host_args,
                                    const stream_config& stream_cfg,
                                    std::size_t flop,
                                    std::size_t num_byte)
{
    constexpr bool is_supported_dtype =
        std::is_same_v<InDataType, half_t> || std::is_same_v<InDataType, float>;

    if constexpr(!is_supported_dtype)
    {
        return {};
    }
    else
    {
        if(host_args.num_dim_spatial_ != 2 || host_args.C_ != 1 || host_args.K_ != 1)
        {
            return {};
        }

        return depthwise_dispatch_detail::
            run_all_instances<InDataType, WeiDataType, AccDataType, OutDataType>(
                host_args, stream_cfg, flop, num_byte);
    }
}

} // namespace ck_tile
