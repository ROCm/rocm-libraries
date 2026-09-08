// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Depthwise convolution backward-data (dgrad) via CK forward depthwise kernel reuse (fp16/bf16).
//
// For stride-1, symmetric odd filters (e.g. convnext 7x7 / pad 3), the input
// gradient is a forward depthwise convolution of the output gradient with the
// 180-degree-rotated weights:
//     dx = conv_fwd(dy, rot180(w))   (stride 1, same padding)
// So we reuse the forward CK depthwise kernel (DeviceGroupedConvFwd) with
// FlipFilter=true instances and swap the in/out tensor pointers. Applicability
// is naturally limited to stride-1 problems because only stride-1 instances are
// present in the factory (CK's IsSupportedArgument rejects stride mismatches).

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
constexpr bool FlipFilter         = true; // rot180 weights => dgrad via fwd kernel

// 7x7 stride-1 pad-3 depthwise instances (FlipFilter=true) covering the convnext
// depthwise spatial stages (56/28/14/7). Same tuning as the forward 7x7 set.
// dgrad via forward kernel is valid for stride-1 (dx = conv_fwd(dy, rot180(w))).
// Generalized over filter size and padding; tuning mirrors the proven fwd stride-1
// instances. Covers convnext (7x7) and efficientnet/mobilenet-style (3x3, 5x5) depthwise.
template <typename DType,
          ck::index_t Filter,
          ck::index_t Pad,
          ck::index_t TileHW,
          ck::index_t NBatch,
          ck::index_t SubH,
          ck::index_t SubW,
          ck::index_t VecIn,
          ck::index_t VecOut>
using DwBwd = ck::tensor_operation::device::DeviceGroupedConvFwd<
    NDimSpatial,
    BlockSize,
    DType,
    DType,
    AccType,
    DType,
    S<TileHW, TileHW>,                        // BlockTileSize
    Filter,                                   // FilterSize
    ck::Tuple<S<1, 1>, S<1, 1>, S<Pad, Pad>>, // FilterParam(dilation, stride=1, padding)
    InElementOp,
    WeiElementOp,
    OutElementOp,
    NBatch,
    SubH,
    SubW,
    VecIn,
    VecOut,
    RequirePadding,
    FlipFilter>;

template <typename DType>
using DeviceConvBwdDataFactoryT = std::tuple<
    // 7x7 stride-1 pad-3 (convnext)
    DwBwd<DType, 7, 3, 7, 32, 4, 4, 1, 1>,
    DwBwd<DType, 7, 3, 14, 32, 4, 4, 2, 2>,
    DwBwd<DType, 7, 3, 28, 32, 4, 4, 4, 4>,
    DwBwd<DType, 7, 3, 56, 8, 7, 8, 8, 8>,
    // 3x3 stride-1 pad-1 (efficientnet/mobilenet)
    DwBwd<DType, 3, 1, 7, 32, 4, 4, 1, 1>,
    DwBwd<DType, 3, 1, 14, 32, 4, 4, 2, 2>,
    DwBwd<DType, 3, 1, 56, 8, 7, 8, 8, 8>,
    // 5x5 stride-1 pad-2 (efficientnet/mobilenet)
    DwBwd<DType, 5, 2, 7, 32, 4, 4, 1, 1>,
    DwBwd<DType, 5, 2, 14, 32, 4, 4, 2, 2>,
    DwBwd<DType, 5, 2, 28, 32, 4, 4, 4, 4>>;
using DeviceConvBwdDataFactory     = DeviceConvBwdDataFactoryT<ck::half_t>;
using DeviceConvBwdDataFactoryBf16 = DeviceConvBwdDataFactoryT<ck::bhalf_t>;

// ---------------------------------------------------------------------------
// CKArgs -- for stride-1 same-size depthwise, dx and dy share spatial dims, so
// the forward-kernel arg layout is symmetric and reused directly.
// ---------------------------------------------------------------------------
struct CKArgs
{
    explicit CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G;
        K  = K1 / G;
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input_lengths = {G, N, C, Hi, Wi};
        out_lens      = {G, N, K, Ho, Wo};
        wei_lens      = {G, K, C, Y, X};
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

    int G, N, K, C, C1, K1, Hi, Wi, Ho, Wo, Y, X;
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

// Each dgrad CK instance is hard-specialized at compile time for a specific filter size, stride,
// padding and dilation. CK's IsSupportedArgument only validates tiling/divisibility -- it does NOT
// reject a shape-mismatched instance -- so without this check a wrong-filter instance (e.g. a 7x7
// instance for a 3x3 problem at a shared tile size) could be enumerated/selected and would read the
// filter out of bounds. Require the instance's compile-time shape to match the problem exactly.
// Mirrors InstanceShapeMatchesProblem in ck_depthwise_fwd_impl.cpp (FlipFilter does not change the
// gridwise shape params, so GridwiseConvFwd exposes the same Filter/Stride/Pad/Dilation constants).
template <typename DeviceOp>
bool InstanceShapeMatchesProblem(const CKArgs& a)
{
    using G = typename DeviceOp::GridwiseConvFwd;
    return G::Filter_Y == a.Y && G::Filter_X == a.X &&                               //
           G::Stride_H == a.filter_stride[0] && G::Stride_W == a.filter_stride[1] && //
           G::Pad_H == a.lPadding[0] && G::Pad_W == a.lPadding[1] &&                 //
           a.lPadding[0] == a.rPadding[0] && a.lPadding[1] == a.rPadding[1] &&       //
           G::Dilation_Y == a.filter_dilation[0] && G::Dilation_X == a.filter_dilation[1];
}

template <typename Factory>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    std::vector<std::string> valid_kernels;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(!InstanceShapeMatchesProblem<DeviceOp>(ck_args))
            return; // instance is specialized for a different filter/stride/pad/dilation
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
            valid_kernels.push_back(conv_ptr.GetTypeString());
    });
    return valid_kernels;
}

template <typename Factory>
bool CheckCKApplicability(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool found                     = false;
    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        if(found)
            return;
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(!InstanceShapeMatchesProblem<DeviceOp>(ck_args))
            return; // instance is specialized for a different filter/stride/pad/dilation
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
            found = true;
    });
    return found;
}

template <typename Factory>
bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool supported                 = false;
    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(conv_ptr.GetTypeString() == kernel_id && InstanceShapeMatchesProblem<DeviceOp>(ck_args))
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

template <typename Factory>
bool BuildDepthwiseBwdSolution(const ProblemDescription& problem,
                               const std::string& kid,
                               miopen::solver::ConvSolution& solution)
{
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool found                     = false;
    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        if(found)
            return;
        const auto device_conv_instance = std::get<i>(Factory{});
        using DeviceConvInstance        = ck::remove_cvref_t<decltype(device_conv_instance)>;
        if(device_conv_instance.GetTypeString() != kid)
            return;
        found              = true;
        auto conv_instance = std::make_shared<DeviceConvInstance>();
        solution.invoker_factory =
            [conv_ptr_ = std::move(conv_instance),
             ck_args   = CKArgs{problem}](const std::vector<miopen::Kernel>&) mutable {
                return [conv_ptr = std::move(conv_ptr_),
                        ck_args](const miopen::Handle& handle,
                                 const miopen::AnyInvokeParams& primitive_params) {
                    const auto& bwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                    // ConvDataTensors normalizes .in=const source, .out=writable dest.
                    // For bwd-data that is .in=dy, .out=dx, so dx = conv_fwd(dy, rot180(w)).
                    auto argument_ptr = conv_ptr->MakeArgumentPointer(
                        bwd_ctx.tensors.in,
                        bwd_ctx.tensors.w,
                        std::array<const void*, 0>{},
                        bwd_ctx.tensors.out,
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
    return found;
}

} // anonymous namespace

extern "C" ck_impl_status_t
ck_impl_depthwise_bwd_data_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                              miopenDataType_t data_type,
                                              bool /*use_tf32*/,
                                              CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result = std::make_unique<CKKernelListHandle>();
        if(data_type == miopenHalf)
            result->kernels = FillValidKernels<DeviceConvBwdDataFactory>(*problem);
        else if(data_type == miopenBFloat16)
            result->kernels = FillValidKernels<DeviceConvBwdDataFactoryBf16>(*problem);
        *out_handle = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_bwd_data_is_applicable(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool /*use_tf32*/,
                                         bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        if(data_type == miopenHalf)
            *out_result = CheckCKApplicability<DeviceConvBwdDataFactory>(*problem);
        else if(data_type == miopenBFloat16)
            *out_result = CheckCKApplicability<DeviceConvBwdDataFactoryBf16>(*problem);
        else
            *out_result = false;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_bwd_data_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                             const char* kernel_id,
                                             miopenDataType_t data_type,
                                             bool /*use_tf32*/,
                                             bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");
        std::string kid(kernel_id);
        if(data_type == miopenHalf)
            *out_result = CheckIsArgSupported<DeviceConvBwdDataFactory>(*problem, kid);
        else if(data_type == miopenBFloat16)
            *out_result = CheckIsArgSupported<DeviceConvBwdDataFactoryBf16>(*problem, kid);
        else
            *out_result = false;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_bwd_data_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                              miopenDataType_t /*data_type*/,
                                              bool /*use_tf32*/,
                                              size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_bwd_data_get_solution(const miopen::ExecutionContext* ctx,
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
        miopen::solver::ConvSolution solution;
        bool found =
            BuildDepthwiseBwdSolution<DeviceConvBwdDataFactory>(*problem, kid, solution) ||
            BuildDepthwiseBwdSolution<DeviceConvBwdDataFactoryBf16>(*problem, kid, solution);

        CK_IMPL_THROW_IF_FALSE(
            found, CK_IMPL_STATUS_INVALID_VALUE, "No matching kernel found for kernel_id");

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
