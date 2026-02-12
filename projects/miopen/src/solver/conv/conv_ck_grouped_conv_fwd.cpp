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
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include "ck/ck.hpp"
#include "miopen/conv/device_grouped_conv_fwd.hpp"
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <ck::index_t... Is>
using S                           = ck::Sequence<Is...>;
using InElementOp                 = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp                = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp                = ck::tensor_operation::element_wise::PassThrough;
using InType                      = ck::half_t;
using WeiType                     = ck::half_t;
using AccType                     = float;
using OutType                     = ck::half_t;
constexpr ck::index_t NDimSpatial = 2;
constexpr ck::index_t BlockSize   = 64;
constexpr bool RequirePadding     = false;

// Tuple of potential device CK kernels. Shapes taken to target fp16 Pytorch EfficientNet B0 model:
// https://docs.pytorch.org/vision/main/models/efficientnet.html
using DeviceConvFwdFactory =
    std::tuple<ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<7, 7>,                              // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   4,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<56, 56>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   4, // SubTileH
                   4, // SubTileW
                   8, // InScalarPerVector
                   4, // OutScalarPerVector
                   RequirePadding>

               ,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<7, 7>,                              // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<14, 14>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<56, 56>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   7, // SubTileH
                   8, // SubTileW
                   8, // InScalarPerVector
                   8, // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   2,  // NBatch
                   14, // SubTileH
                   16, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>

               ,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<28, 28>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   BlockSize,
                   InType,
                   WeiType,
                   AccType,
                   OutType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8, // NBatch
                   7, // SubTileH
                   8, // SubTileW
                   8, // InScalarPerVector
                   8, // OutScalarPerVector
                   RequirePadding>>;

namespace {
struct CKArgs
{
    explicit CKArgs(const ProblemDescription& problem)
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

        input_lengths = {G, N, C, Hi, Wi}; // input
        out_lens      = {G, N, K, Ho, Wo}; // output
        wei_lens      = {G, K, C, Y, X};   // filter = wei
        bias_lens     = {G, 1, K, 1, 1};
        bias_strides  = {K, 0, 1, 0, 0};

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

    CKArgs(const CKArgs&)            = default;
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
    int Ho;
    int Wo;
    int Y;
    int X;
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

template <typename DataType>
void PerformanceConfigConvDepthwiseFwd2D::Init(const ProblemDescription& problem)
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

    if(valid_kernels.empty())
        MIOPEN_THROW("No ConvDepthwiseFwd2D kernels found");

    index     = 0;
    kernel_id = valid_kernels[index];
}
#endif

void PerformanceConfigConvDepthwiseFwd2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    default: break;
    }
#endif
}

bool PerformanceConfigConvDepthwiseFwd2D::SetNextValue(const ProblemDescription& problem)
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

bool PerformanceConfigConvDepthwiseFwd2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
    return IsValidValue();
}

bool PerformanceConfigConvDepthwiseFwd2D::operator==(
    const PerformanceConfigConvDepthwiseFwd2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // Kernel requires a wavefront size of 64
    if(64 != ctx.GetStream().GetWavefrontWidth())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

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
#else
    return false;
#endif
}

uint32_t
ConvDepthwiseFwd2D::GetSupportedSolutionCount([[maybe_unused]] const ExecutionContext& ctx,
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

PerformanceConfigConvDepthwiseFwd2D ConvDepthwiseFwd2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseFwd2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseFwd2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseFwd2D
ConvDepthwiseFwd2D::Search(const ExecutionContext& ctx,
                           const miopen::conv::ProblemDescription& problem,
                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseFwd2D::GetSolution(const ExecutionContext&,
                                const miopen::conv::ProblemDescription& problem,
                                const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    ConvSolution sol;
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    ck::static_for<0, std::tuple_size_v<DeviceConvFwdFactory>, 1>{}([&](auto i) -> void {
        const auto device_conv_fwd_instance = std::get<i>(DeviceConvFwdFactory{});
        using DeviceConvFwdInstance = ck::remove_cvref_t<decltype(device_conv_fwd_instance)>;
        auto conv_ptr               = std::make_shared<DeviceConvFwdInstance>();

        if(conv_ptr->GetTypeString() == config.kernel_id)
        {
            MIOPEN_LOG_I("Run conv : " << conv_ptr->GetTypeString());

            sol.invoker_factory = [conv_ptr = std::move(conv_ptr),
                                   ck_args  = CKArgs{problem}](const std::vector<Kernel>&) {
                return [conv_ptr = std::move(conv_ptr),
                        ck_args](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    const auto& fwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                    auto invoker        = conv_ptr->MakeInvoker();
                    auto argument       = ck_args.MakeArgument(*conv_ptr.get(),
                                                         fwd_ctx.tensors.in,
                                                         fwd_ctx.tensors.w,
                                                         fwd_ctx.tensors.out);

                    {
                        {
                            WorkAroundHipEventProfiler prf(handle);
                            invoker.Run(argument, StreamConfig{nullptr, false});
                        }
                        if(handle.IsProfilingEnabled())
                        {
                            float avg_time = handle.GetKernelTime();
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
