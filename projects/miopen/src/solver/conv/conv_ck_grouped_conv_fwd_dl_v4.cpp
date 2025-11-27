// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#endif

#define DISABLE_OUTPUT_LDS 1
#include "miopen/conv/device_grouped_conv_fwd_dl_v4.hpp"
#include <array>

template <ck::index_t... Is>
using S           = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// kernel data types
using InKernelDataType  = ck::half_t;
using WeiKernelDataType = ck::half_t;
using AccDataType       = float;
using CShuffleDataType  = ck::half_t;
using OutKernelDataType = ck::half_t;

// tensor data types
using InDataType  = InKernelDataType;
using WeiDataType = WeiKernelDataType;
using OutDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

using InType  = InKernelDataType;
using WeiType = WeiKernelDataType;
using AccType = AccDataType;
using OutType = OutKernelDataType;

constexpr ck::index_t NDimSpatial = 2;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD)

namespace miopen {
namespace solver {
namespace conv {
using DeviceConvFwdFactory = std::tuple<

    //                                                NDimSpatial BlockSize In      Wei        Acc
    //                                                Out      BlockTileSize FilterSize FilterParam
    //                                                (dilation, stride, padding) NBatch  SubTileH W
    //                                                ScalarPerVector(in out)    RequirePadding>
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<7, 7>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           1,
                                                           1,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<14, 14>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           2,
                                                           2,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<28, 28>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           4,
                                                           4,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<14, 14>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           2,
                                                           1,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<28, 28>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           4,
                                                           2,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<56, 56>,
                                                           5,
                                                           ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           8,
                                                           4,
                                                           4,
                                                           8,
                                                           4,
                                                           false>

    ,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<7, 7>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           1,
                                                           1,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<14, 14>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           2,
                                                           2,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<56, 56>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           8,
                                                           7,
                                                           8,
                                                           8,
                                                           8,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<112, 112>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           2,
                                                           14,
                                                           16,
                                                           8,
                                                           8,
                                                           false>

    ,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<28, 28>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           32,
                                                           4,
                                                           4,
                                                           4,
                                                           2,
                                                           false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV4<2,
                                                           64,
                                                           InType,
                                                           WeiType,
                                                           AccType,
                                                           OutType,
                                                           S<112, 112>,
                                                           3,
                                                           ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>,
                                                           InElementOp,
                                                           WeiElementOp,
                                                           OutElementOp,
                                                           8,
                                                           7,
                                                           8,
                                                           8,
                                                           8,
                                                           false>>;

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);

        input_lengths = {G, N, C, Hi, Wi}; // input
        out_lens      = {G, N, K, Ho, Wo}; // output
        wei_lens      = {G, K, C, Y, X};   // filter = wei
        bias_lens     = {G, 1, K, 1, 1};
        bias_strides  = {K, 0, 1, 0, 0};

        const std::string layout = problem.GetInLayout();

        in_strides  = {Hi * Wi * C, G * Hi * Wi * C, 1, Wi * C, C};
        out_strides = {Ho * Wo * K, G * Ho * Wo * K, 1, Wo * K, K};
        wei_strides = {Y * X * C, G * Y * X * C, 1, X * C, C};

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                         ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgument(const ConvPtr& conv_ptr, ConstData_t in, ConstData_t w, Data_t out) const
    {
        return conv_ptr.MakeArgument(in,
                                     w,
                                     std::array<const void*, 0>{},
                                     out,
                                     input_lengths,
                                     in_strides,
                                     wei_lens,
                                     wei_strides,
                                     std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                     std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                     out_lens,
                                     out_strides,
                                     filter_stride,
                                     filter_dilation,
                                     lPadding,
                                     rPadding,
                                     InElementOp{},
                                     WeiElementOp{},
                                     OutElementOp{});
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;
    std::array<ck::index_t, 5> input_lengths;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> out_lens;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> wei_lens;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 5> bias_lens;
    std::array<ck::index_t, 5> bias_strides;
    std::array<ck::index_t, 2> filter_stride;
    std::array<ck::index_t, 2> filter_dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
} // namespace

bool PerformanceConfigConvDepthwiseFwd::operator==(
    const PerformanceConfigConvDepthwiseFwd& other) const
{
    return kernel_id == other.kernel_id;
}

template <typename DataType>
void PerformanceConfigConvDepthwiseFwd::Init(const ProblemDescription& problem)
{
    const auto& ck_args            = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        const auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
        auto argument       = ck_args.MakeArgument(conv_ptr, nullptr, nullptr, nullptr);
        if(conv_ptr.IsSupportedArgument(argument))
        {
            valid_kernels.push_back(std::move(conv_ptr.GetTypeString()));
        }
    });
    MIOPEN_LOG_I2("valid kernels count: " << valid_kernels.size()
                                          << ", total kernel count:" << kernelCount);

    index     = 0;
    kernel_id = valid_kernels[index];
}

void PerformanceConfigConvDepthwiseFwd::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenInt8: break;
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat:
    case miopenBFloat16:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64:
    case miopenInt32:
    case miopenDouble: break;
    }
#endif
    // Todo: Add AI heuristic support
}

bool PerformanceConfigConvDepthwiseFwd::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit({}, problem);
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        index++;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvDepthwiseFwd::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseFwd::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
    return IsValidValue();
}

bool PerformanceConfigConvDepthwiseFwd::IsModelApplicable(const ExecutionContext& ctx,
                                                          const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942")
        return false;
    if(problem.GetInDataType() != miopenHalf)
        return false;
    return true;
}

ConvDepthwiseFwd::ConvDepthwiseFwd() {}

bool ConvDepthwiseFwd::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // TODO: support NHWC layout
    if(!problem.IsLayoutDefault())
        return false;

    // TODO: support more data type
    if(!problem.IsFp16())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    // Only depthwise convolution is supported
    if((problem.GetGroupCount() != problem.GetOutChannels()) ||
       (problem.GetGroupCount() != problem.GetInChannels()))
        return false;

    if(GetSupportedSolutionCount(ctx, problem) == 0)
    {
        return false;
    }

    return true;
}

uint32_t
ConvDepthwiseFwd::GetSupportedSolutionCount([[maybe_unused]] const ExecutionContext& ctx,
                                            const miopen::conv::ProblemDescription& problem) const
{
    uint32_t solutionCount = 0;
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto& ck_args = CKArgs{problem};

    ck::static_for<0, std::tuple_size_v<DeviceConvFwdFactory>, 1>{}([&](auto i) -> void {
        const auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});

        auto argument = ck_args.MakeArgument(conv_ptr, nullptr, nullptr, nullptr);

        if(conv_ptr.IsSupportedArgument(argument))
        {
            solutionCount++;
        }
    });
#endif
    return solutionCount;
}

PerformanceConfigConvDepthwiseFwd
ConvDepthwiseFwd::GetDefaultPerformanceConfig([[maybe_unused]] const ExecutionContext& ctx,
                                              const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseFwd pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseFwd::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseFwd& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseFwd
ConvDepthwiseFwd::Search(const ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvDepthwiseFwd::GetSolution(const ExecutionContext& ctx,
                                           const miopen::conv::ProblemDescription& problem,
                                           const PerformanceConfigConvDepthwiseFwd& config) const
{
    (void)ctx; // Unused, as we use the config directly
    ConvSolution sol;
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto& ck_args = CKArgs{problem};

    // auto factory_list = DeviceConvFwdFactory{};
    ck::static_for<0, std::tuple_size_v<DeviceConvFwdFactory>, 1>{}([&](auto i) -> void {
        const auto device_conv_fwd_instance = std::get<i>(DeviceConvFwdFactory{});
        using DeviceConvFwdInstance = ck::remove_cvref_t<decltype(device_conv_fwd_instance)>;
        auto conv_ptr               = std::make_shared<DeviceConvFwdInstance>();

        if(conv_ptr->GetTypeString() == config.kernel_id)
        {
            MIOPEN_LOG_I("Run conv : " << conv_ptr->GetTypeString());
            sol.invoker_factory = [conv_ptr = std::move(conv_ptr),
                                   problem](const std::vector<Kernel>& kernels) {
                (void)kernels; // Unused, as we use the conv_ptr directly
                return [conv_ptr = std::move(conv_ptr),
                        problem](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    const auto& fwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                    const auto& ck_args = CKArgs{problem};
                    auto invoker        = conv_ptr->MakeInvoker();
                    auto argument       = ck_args.MakeArgument(*conv_ptr.get(),
                                                         fwd_ctx.tensors.in,
                                                         fwd_ctx.tensors.w,
                                                         fwd_ctx.tensors.out);

                    {
                        WorkAroundHipEventProfiler prf(handle);
                        float avg_time = invoker.Run(argument, StreamConfig{nullptr, false});

                        if(handle.IsProfilingEnabled())
                        {
                            avg_time = handle.GetKernelTime();
                            handle.ResetKernelTime();
                            handle.AccumKernelTime(avg_time);
                        }
                    }
                };
            };
        }
    });
#endif

    return sol;
}
} // namespace conv
} // namespace solver
} // namespace miopen
