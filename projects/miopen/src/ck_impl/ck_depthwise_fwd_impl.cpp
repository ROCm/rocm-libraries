// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>

#include "ck_grouped_conv_common.hpp"
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "miopen/conv/device_grouped_conv_fwd.hpp"

// ---------------------------------------------------------------------------
// CK type aliases and kernel factory for depthwise conv forward (FP16)
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

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

// ---------------------------------------------------------------------------
// CKArgs -- extracts convolution dimensions from ProblemDescription for the
// custom depthwise kernel API.
// ---------------------------------------------------------------------------

struct CKArgs
{
    explicit CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input channels per group
        K  = K1 / G; // Number of output channels per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input_lengths = {G, N, C, Hi, Wi}; // input
        out_lens      = {G, N, K, Ho, Wo}; // output
        wei_lens      = {G, K, C, Y, X};   // filter = wei
        in_strides    = {Hi * Wi * C, G * Hi * Wi * C, 1, Wi * C, C};
        out_strides   = {Ho * Wo * K, G * Ho * Wo * K, 1, Wo * K, K};
        wei_strides   = {Y * X * C, G * Y * X * C, 1, X * C, C};

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
    std::array<ck::index_t, 2> filter_stride;
    std::array<ck::index_t, 2> filter_dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

// ---------------------------------------------------------------------------
// Helpers: enumerate valid kernels, check applicability, check arg support
// ---------------------------------------------------------------------------

std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;
    std::vector<std::string> valid_kernels;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
        auto argument_ptr =
            conv_ptr.MakeArgumentPointer(nullptr,
                                         nullptr,
                                         std::array<const void*, 0>{},
                                         nullptr,
                                         ck_args.input_lengths,
                                         ck_args.in_strides,
                                         ck_args.wei_lens,
                                         ck_args.wei_strides,
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         ck_args.out_lens,
                                         ck_args.out_strides,
                                         ck_args.filter_stride,
                                         ck_args.filter_dilation,
                                         ck_args.lPadding,
                                         ck_args.rPadding,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(conv_ptr.GetTypeString());
        }
    });
    return valid_kernels;
}

bool CheckCKApplicability(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;
    bool found                     = false;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        if(found)
            return;
        auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
        auto argument_ptr =
            conv_ptr.MakeArgumentPointer(nullptr,
                                         nullptr,
                                         std::array<const void*, 0>{},
                                         nullptr,
                                         ck_args.input_lengths,
                                         ck_args.in_strides,
                                         ck_args.wei_lens,
                                         ck_args.wei_strides,
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         ck_args.out_lens,
                                         ck_args.out_strides,
                                         ck_args.filter_stride,
                                         ck_args.filter_dilation,
                                         ck_args.lPadding,
                                         ck_args.rPadding,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            found = true;
        }
    });
    return found;
}

bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;
    bool supported                 = false;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr = std::get<i>(DeviceConvFwdFactory{});
        if(conv_ptr.GetTypeString() == kernel_id)
        {
            auto argument_ptr = conv_ptr.MakeArgumentPointer(
                nullptr,
                nullptr,
                std::array<const void*, 0>{},
                nullptr,
                ck_args.input_lengths,
                ck_args.in_strides,
                ck_args.wei_lens,
                ck_args.wei_strides,
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                ck_args.out_lens,
                ck_args.out_strides,
                ck_args.filter_stride,
                ck_args.filter_dilation,
                ck_args.lPadding,
                ck_args.rPadding,
                InElementOp{},
                WeiElementOp{},
                OutElementOp{});
            supported = conv_ptr.IsSupportedArgument(argument_ptr.get());
        }
    });
    return supported;
}

} // anonymous namespace

// ===========================================================================
// Depthwise Conv FWD extern "C" functions
// ===========================================================================

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool /*use_tf32*/,
                                         CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result = std::make_unique<CKKernelListHandle>();
        if(data_type == miopenHalf)
            result->kernels = FillValidKernels(*problem);
        *out_handle = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                                    miopenDataType_t data_type,
                                    bool /*use_tf32*/,
                                    bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        if(data_type != miopenHalf)
        {
            *out_result = false;
            return;
        }
        *out_result = CheckCKApplicability(*problem);
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                        const char* kernel_id,
                                        miopenDataType_t data_type,
                                        bool /*use_tf32*/,
                                        bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");
        if(data_type != miopenHalf)
        {
            *out_result = false;
            return;
        }
        std::string kid(kernel_id);
        *out_result = CheckIsArgSupported(*problem, kid);
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool /*use_tf32*/,
                                         size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        if(data_type != miopenHalf)
        {
            *out_size = 0;
            return;
        }
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_get_solution(const miopen::ExecutionContext* ctx,
                                   const miopen::conv::ProblemDescription* problem,
                                   const char* kernel_id,
                                   bool /*use_tf32*/,
                                   miopen::solver::ConvSolution** out_solution)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_solution, CK_IMPL_STATUS_BAD_PARAM, "Null out_solution");
        CK_IMPL_THROW_IF_NULL(ctx, CK_IMPL_STATUS_BAD_PARAM, "Null ctx");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");

        std::string kid(kernel_id);
        constexpr uint32_t kernelCount = std::tuple_size_v<DeviceConvFwdFactory>;

        miopen::solver::ConvSolution solution;
        bool found = false;

        ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
            if(found)
                return;
            const auto device_conv_fwd_instance = std::get<i>(DeviceConvFwdFactory{});
            using DeviceConvFwdInstance = ck::remove_cvref_t<decltype(device_conv_fwd_instance)>;

            if(device_conv_fwd_instance.GetTypeString() != kid)
                return;

            found              = true;
            auto conv_instance = std::make_shared<DeviceConvFwdInstance>();

            solution.invoker_factory = [conv_ptr_ = std::move(conv_instance),
                                        ck_args   = CKArgs{*problem}](
                                           const std::vector<miopen::Kernel>&) mutable {
                return [conv_ptr = std::move(conv_ptr_),
                        ck_args](const miopen::Handle& handle,
                                 const miopen::AnyInvokeParams& primitive_params) {
                    const auto& fwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                    auto argument_ptr   = conv_ptr->MakeArgumentPointer(
                        fwd_ctx.tensors.in,
                        fwd_ctx.tensors.w,
                        std::array<const void*, 0>{},
                        fwd_ctx.tensors.out,
                        ck_args.input_lengths,
                        ck_args.in_strides,
                        ck_args.wei_lens,
                        ck_args.wei_strides,
                        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                        ck_args.out_lens,
                        ck_args.out_strides,
                        ck_args.filter_stride,
                        ck_args.filter_dilation,
                        ck_args.lPadding,
                        ck_args.rPadding,
                        InElementOp{},
                        WeiElementOp{},
                        OutElementOp{});
                    auto invoker_ptr = conv_ptr->MakeInvokerPointer();
                    {
                        miopen::HipEventProfiler prf(handle);
                        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
                    }
                    if(handle.IsProfilingEnabled())
                    {
                        float avg_time = handle.GetKernelTime();
                        handle.ResetKernelTime();
                        handle.AccumKernelTime(avg_time);
                    }
                };
            };
        });

        CK_IMPL_THROW_IF_FALSE(
            found, CK_IMPL_STATUS_INVALID_VALUE, "No matching kernel found for kernel_id");

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
