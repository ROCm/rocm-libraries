// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Generated Convolution Kernel Backend
//
// Wraps CK Tile grouped convolution launchers for use through the
// GroupedConvDispatcher.  Each generated kernel launcher is wrapped in
// a ConvKernelRunFn that builds the correct host-args type (forward,
// bwd-data, or bwd-weight) and calls Launcher::launch().

#pragma once

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_invocation.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include <hip/hip_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace backends {

inline std::string normalize_conv_arch_name(std::string arch)
{
    const auto separator = arch.find(':');
    if(separator != std::string::npos) arch.resize(separator);
    return arch;
}

inline std::string query_conv_device_arch(int ordinal)
{
    hipDeviceProp_t properties{};
    const auto status = hipGetDeviceProperties(&properties, ordinal);
    if(status != hipSuccess) throw std::runtime_error("Failed to query grouped convolution device architecture for device ordinal " + std::to_string(ordinal));
    return properties.gcnArchName;
}

// CK Tile grouped-conv launchers only accept symmetric padding.
inline bool has_symmetric_padding(const GroupedConvProblem& problem)
{
    return problem.padding_left == problem.padding_right;
}

inline bool has_grouped_channel_counts(const GroupedConvProblem& problem)
{
    return problem.G > 0 && problem.C % problem.G == 0 && problem.K % problem.G == 0;
}

enum class ConvKBatchRule
{
    MustBeOne,
    AllowSplitK
};

inline void validate_conv_invocation(const GroupedConvProblem& problem,
                                     const ConvInvocationContext& context,
                                     ConvKBatchRule k_batch_rule)
{
    if(context.device_ordinal < 0)
        throw std::invalid_argument(
            "Grouped convolution invocation requires a device ordinal");
    if(k_batch_rule == ConvKBatchRule::MustBeOne && context.k_batch != 1)
        throw std::invalid_argument(
            "Grouped convolution invocation requires k_batch=1");
    const auto actual = normalize_conv_arch_name(
        context.arch_provider ? context.arch_provider(context.device_ordinal)
                              : query_conv_device_arch(context.device_ordinal));
    if(actual != problem.arch)
        throw std::runtime_error(
            "Grouped convolution device architecture mismatch: problem=" + problem.arch +
            ", device=" + actual);
}

// Helper: build ck_tile::conv::ConvParam from GroupedConvProblem
inline ck_tile::conv::ConvParam make_conv_param_2d(const GroupedConvProblem& p)
{
    return ck_tile::conv::ConvParam{
        2,
        static_cast<ck_tile::index_t>(p.G),
        static_cast<ck_tile::index_t>(p.N),
        static_cast<ck_tile::index_t>(p.K / p.G),
        static_cast<ck_tile::index_t>(p.C / p.G),
        {static_cast<ck_tile::index_t>(p.filter_spatial[1]),
         static_cast<ck_tile::index_t>(p.filter_spatial[2])},
        {static_cast<ck_tile::index_t>(p.input_spatial[1]),
         static_cast<ck_tile::index_t>(p.input_spatial[2])},
        {static_cast<ck_tile::index_t>(p.stride[1]), static_cast<ck_tile::index_t>(p.stride[2])},
        {static_cast<ck_tile::index_t>(p.dilation[1]),
         static_cast<ck_tile::index_t>(p.dilation[2])},
        {static_cast<ck_tile::index_t>(p.padding_left[1]),
         static_cast<ck_tile::index_t>(p.padding_left[2])},
        {static_cast<ck_tile::index_t>(p.padding_right[1]),
         static_cast<ck_tile::index_t>(p.padding_right[2])}};
}

inline ck_tile::conv::ConvParam make_conv_param_3d(const GroupedConvProblem& p)
{
    return ck_tile::conv::ConvParam{3,
                                    static_cast<ck_tile::index_t>(p.G),
                                    static_cast<ck_tile::index_t>(p.N),
                                    static_cast<ck_tile::index_t>(p.K / p.G),
                                    static_cast<ck_tile::index_t>(p.C / p.G),
                                    {static_cast<ck_tile::index_t>(p.filter_spatial[0]),
                                     static_cast<ck_tile::index_t>(p.filter_spatial[1]),
                                     static_cast<ck_tile::index_t>(p.filter_spatial[2])},
                                    {static_cast<ck_tile::index_t>(p.input_spatial[0]),
                                     static_cast<ck_tile::index_t>(p.input_spatial[1]),
                                     static_cast<ck_tile::index_t>(p.input_spatial[2])},
                                    {static_cast<ck_tile::index_t>(p.stride[0]),
                                     static_cast<ck_tile::index_t>(p.stride[1]),
                                     static_cast<ck_tile::index_t>(p.stride[2])},
                                    {static_cast<ck_tile::index_t>(p.dilation[0]),
                                     static_cast<ck_tile::index_t>(p.dilation[1]),
                                     static_cast<ck_tile::index_t>(p.dilation[2])},
                                    {static_cast<ck_tile::index_t>(p.padding_left[0]),
                                     static_cast<ck_tile::index_t>(p.padding_left[1]),
                                     static_cast<ck_tile::index_t>(p.padding_left[2])},
                                    {static_cast<ck_tile::index_t>(p.padding_right[0]),
                                     static_cast<ck_tile::index_t>(p.padding_right[1]),
                                     static_cast<ck_tile::index_t>(p.padding_right[2])}};
}

// Create a RunFn for a forward convolution launcher (2D or 3D)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_fwd_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        const auto& ctx = conv_invocation_context();
        validate_conv_invocation(problem, ctx, ConvKBatchRule::MustBeOne);
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        ck_tile::GroupedConvFwdHostArgs<> args(
            param, ctx.input_ptr, ctx.weight_ptr, {}, ctx.output_ptr, ctx.k_batch);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

// Create a RunFn for a backward-data convolution launcher.
// Dispatcher convention: run(dY, W, dX, problem) where dX is computed.
// BwdDataHostArgs(param, in_ptr=dX, wei_ptr=W, {}, out_ptr=dY, k_batch)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_bwd_data_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        const auto& ctx = conv_invocation_context();
        validate_conv_invocation(problem, ctx, ConvKBatchRule::MustBeOne);
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        ck_tile::GroupedConvBwdDataHostArgs args(
            param,
            ctx.output_ptr, // in_ptr = dX (being computed)
            ctx.weight_ptr, // wei_ptr = W
            {},
            ctx.input_ptr, // out_ptr = dY (gradient from next layer)
            1);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

// Create a RunFn for a backward-weight convolution launcher.
// Dispatcher convention: run(X, dY, dW, problem) where dW is computed.
// BwdWeightHostArgs(param, in_ptr=X, wei_ptr=dW, {}, out_ptr=dY, k_batch)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_bwd_weight_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        const auto& ctx   = conv_invocation_context();
        validate_conv_invocation(problem, ctx, ConvKBatchRule::AllowSplitK);
        auto param        = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        const int k_batch = (ctx.k_batch > 1) ? ctx.k_batch : 1;
        ck_tile::GroupedConvBwdWeightHostArgs args(param,
                                                   ctx.input_ptr,  // in_ptr = X
                                                   ctx.output_ptr, // wei_ptr = dW (being computed)
                                                   {},
                                                   ctx.weight_ptr, // out_ptr = dY
                                                   k_batch);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

// -------------------------------------------------------------------------
// IsSupportedFn factories -- check kernel applicability without launching
// -------------------------------------------------------------------------

template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::IsSupportedFn make_conv_bwd_weight_is_supported_fn()
{
    return [](const GroupedConvProblem& problem) -> bool {
        if(!has_grouped_channel_counts(problem) || !has_symmetric_padding(problem))
            return false;
        const auto& ctx   = conv_invocation_context();
        auto param        = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        const int k_batch = ctx.k_batch;
        return LauncherType::is_supported(param, k_batch);
    };
}

template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::IsSupportedFn make_conv_fwd_is_supported_fn()
{
    return [](const GroupedConvProblem& problem) -> bool {
        if(!has_grouped_channel_counts(problem) || !has_symmetric_padding(problem))
            return false;
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        return LauncherType::is_supported(param, 1);
    };
}

template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::IsSupportedFn make_conv_bwd_data_is_supported_fn()
{
    return [](const GroupedConvProblem& problem) -> bool {
        if(!has_grouped_channel_counts(problem) || !has_symmetric_padding(problem))
            return false;
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        return LauncherType::is_supported(param, 1);
    };
}

// -------------------------------------------------------------------------
// Instance string extraction -- get CK Tile GetInstanceString() representation
// -------------------------------------------------------------------------

#ifdef CK_EXPERIMENTAL_BUILDER
template <typename LauncherType>
inline std::string get_instance_string()
{
    return LauncherType::get_instance_string();
}
#endif

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
