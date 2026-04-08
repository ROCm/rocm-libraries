// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_scale.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for 3D grouped convolution backward data
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

using InLayout    = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck::tensor_layout::convolution::NDHWGK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;
using Scale       = ck::tensor_operation::element_wise::Scale;

static constexpr ck::index_t NumDimSpatial = 3;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdBilinear =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<InLayout>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<DataType>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdScale =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Scale,
                                                                    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdBilinearPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdBilinear<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdScale<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdDefault<DataType, ComputeType>>;

// ---------------------------------------------------------------------------
// CKArgs -- 3D backward data grouped convolution.
// ---------------------------------------------------------------------------

template <typename DataType, typename ComputeType = DataType>
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
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
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);

        in_lengths  = {G, N, C, Di, Hi, Wi};
        out_lengths = {G, N, K, Do, Ho, Wo};
        wei_lengths = {G, K, C, Z, Y, X};

        if(problem.IsLayoutNHWC())
        {
            auto copy_strides = [](const auto& src, auto& dst) {
                assert(dst.size() == (src.size() + 1));
                std::copy(src.begin(), src.end(), dst.begin() + 1);
            };
            copy_strides(problem.GetIn().GetStrides(), in_strides);
            copy_strides(problem.GetOut().GetStrides(), out_strides);
            copy_strides(problem.GetWeights().GetStrides(), wei_strides);

            // On a backward pass, problem.GetIn() means y(or out),
            // and problem.GetOut means x(or in)
            std::swap(in_strides, out_strides);

            in_strides[0]  = C;
            out_strides[0] = K;
            wei_strides[0] = K * wei_strides[1];
        }
        else
        {
            assert(problem.IsLayoutDefault());
            in_strides  = {C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
            out_strides = {K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
        }

        filter_strides   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilations = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding         = {ProblemInterpreter::GetInputLeftPadD(problem),
                            ProblemInterpreter::GetInputLeftPadH(problem),
                            ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding         = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    Data_t in,
                    ConstData_t w,
                    ConstData_t out,
                    float alpha,
                    float beta) const
    {
        using DeviceP = std::remove_pointer_t<decltype(conv_ptr.get())>;
        if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdBilinear<DataType, ComputeType>>)
        {
            return MakeBilinearArgPtr(conv_ptr, in, w, out, alpha, beta);
        }
        else if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdScale<DataType, ComputeType>>)
        {
            (void)beta;
            return MakeScaleArgPtr(conv_ptr, in, w, out, alpha);
        }
        else
        {
            (void)alpha;
            (void)beta;
            static_assert(std::is_same_v<DeviceP, DeviceOpGBwdDefault<DataType, ComputeType>>,
                          "Default should be bwd pass through");
            return MakeDefaultArgPtr(conv_ptr, in, w, out);
        }
    }

    template <typename ConvPtr>
    auto MakeBilinearArgPtr(const ConvPtr& conv_ptr,
                            Data_t in,
                            ConstData_t w,
                            ConstData_t out,
                            float alpha,
                            float beta) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {in},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {in_lengths},
                                             {in_strides},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             Bilinear{alpha, beta});
    }

    template <typename ConvPtr>
    auto MakeScaleArgPtr(
        const ConvPtr& conv_ptr, Data_t in, ConstData_t w, ConstData_t out, float alpha) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             Scale{alpha});
    }

    template <typename ConvPtr>
    auto MakeDefaultArgPtr(const ConvPtr& conv_ptr, Data_t in, ConstData_t w, ConstData_t out) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             PassThrough{});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const miopen::ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
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
    std::array<ck::index_t, 6> in_lengths;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lengths;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lengths;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> filter_strides;
    std::array<ck::index_t, 3> filter_dilations;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

// ---------------------------------------------------------------------------
// Template helpers — dispatched per alpha/beta case
// ---------------------------------------------------------------------------

template <typename DataType, typename ComputeType>
std::vector<std::string> FillValidKernelsByAlphaBeta(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGBwdBilinearPtrs<DataType, ComputeType>,
                                                   CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGBwdScalePtrs<DataType, ComputeType>,
                                                   CKArgs<DataType, ComputeType>>(problem);
    default:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGBwdDefaultPtrs<DataType, ComputeType>,
                                                   CKArgs<DataType, ComputeType>>(problem);
    }
}

template <typename DataType, typename ComputeType>
bool CheckCKApplicabilityByAlphaBeta(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::IsCKApplicable<DeviceOpGBwdBilinearPtrs<DataType, ComputeType>,
                                              CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return miopen::solver::IsCKApplicable<DeviceOpGBwdScalePtrs<DataType, ComputeType>,
                                              CKArgs<DataType, ComputeType>>(problem);
    default:
        return miopen::solver::IsCKApplicable<DeviceOpGBwdDefaultPtrs<DataType, ComputeType>,
                                              CKArgs<DataType, ComputeType>>(problem);
    }
}

template <typename DataType, typename ComputeType>
bool CheckIsArgSupportedByAlphaBeta(const ProblemDescription& problem, const std::string& kernel_id)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::IsCKArgsSupported<DeviceOpGBwdBilinearPtrs<DataType, ComputeType>,
                                                 CKArgs<DataType, ComputeType>>(problem, kernel_id);
    case SCALE:
        return miopen::solver::IsCKArgsSupported<DeviceOpGBwdScalePtrs<DataType, ComputeType>,
                                                 CKArgs<DataType, ComputeType>>(problem, kernel_id);
    default:
        return miopen::solver::IsCKArgsSupported<DeviceOpGBwdDefaultPtrs<DataType, ComputeType>,
                                                 CKArgs<DataType, ComputeType>>(problem, kernel_id);
    }
}

// ---------------------------------------------------------------------------
// Top-level dispatch by data type
// ---------------------------------------------------------------------------

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
            return FillValidKernelsByAlphaBeta<DataType, ck::tf32_t>(problem);
    }
    return FillValidKernelsByAlphaBeta<DataType, DataType>(problem);
}

template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 && CheckCKApplicabilityByAlphaBeta<DataType, ck::tf32_t>(problem))
            return true;
    }
    return CheckCKApplicabilityByAlphaBeta<DataType, DataType>(problem);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
            return CheckIsArgSupportedByAlphaBeta<DataType, ck::tf32_t>(problem, kernel_id);
    }
    return CheckIsArgSupportedByAlphaBeta<DataType, DataType>(problem, kernel_id);
}

} // anonymous namespace

// ===========================================================================
// Get all BWD kernel TypeStrings (exposed via extern "C" for dlopen)
// ===========================================================================

extern "C" ck_impl_status_t
ck_impl_3d_bwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        auto result = std::make_unique<CKKernelListHandle>();

        auto bilinear_ptrs = DeviceOpGBwdBilinearPtrs<float>::GetInstances();
        auto scale_ptrs    = DeviceOpGBwdScalePtrs<float>::GetInstances();
        auto default_ptrs  = DeviceOpGBwdDefaultPtrs<float>::GetInstances();

        result->kernels.reserve(bilinear_ptrs.size() + scale_ptrs.size() + default_ptrs.size());

        for(const auto& ptr : bilinear_ptrs)
            result->kernels.push_back(ptr->GetTypeString());
        for(const auto& ptr : scale_ptrs)
            result->kernels.push_back(ptr->GetTypeString());
        for(const auto& ptr : default_ptrs)
            result->kernels.push_back(ptr->GetTypeString());

        *out_handle = result.release();
    });
}

// ===========================================================================
// 3D BWD direction extern "C" functions
// ===========================================================================

using miopen::solver::InitInvokerFactoryBwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;

extern "C" ck_impl_status_t
ck_impl_3d_bwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result     = std::make_unique<CKKernelListHandle>();
        result->kernels = DispatchByDataType(data_type, [&](auto type_val) {
            return FillValidKernels<decltype(type_val)>(*problem, use_tf32);
        });
        *out_handle     = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_3d_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                             miopenDataType_t data_type,
                             bool use_tf32,
                             bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_result = DispatchByDataType(data_type, [&](auto type_val) {
            return CheckCKApplicability<decltype(type_val)>(*problem, use_tf32);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_3d_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                 const char* kernel_id,
                                 miopenDataType_t data_type,
                                 bool use_tf32,
                                 bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");
        std::string kid(kernel_id);
        *out_result = DispatchByDataType(data_type, [&](auto type_val) {
            return CheckIsArgSupported<decltype(type_val)>(*problem, kid, use_tf32);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_3d_bwd_get_workspace_size(const miopen::conv::ProblemDescription* /*problem*/,
                                  miopenDataType_t /*data_type*/,
                                  bool /*use_tf32*/,
                                  size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        // 3D BWD grouped convolution CK kernels do not use split-k and never
        // require CK-level workspace.  Layout transform workspace (NCHW to NHWC)
        // is computed independently at the solver level via
        // GetWorkspaceSizeLayoutTransformConv().
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_3d_bwd_get_solution(const miopen::ExecutionContext* ctx,
                            const miopen::conv::ProblemDescription* problem,
                            const char* kernel_id,
                            bool use_tf32,
                            miopen::solver::ConvSolution** out_solution)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_solution, CK_IMPL_STATUS_BAD_PARAM, "Null out_solution");
        CK_IMPL_THROW_IF_NULL(ctx, CK_IMPL_STATUS_BAD_PARAM, "Null ctx");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");
        std::string kid(kernel_id);

        auto solution = MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                switch(problem->GetAlphaBetaCase())
                {
                case BILINEAR:
                    return InitInvokerFactoryBwdNCHW<3,
                                                     false,
                                                     DeviceOpGBwdBilinearPtrs<T, TCompute>,
                                                     CKArgs<T, TCompute>,
                                                     miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                case SCALE:
                    return InitInvokerFactoryBwdNCHW<3,
                                                     false,
                                                     DeviceOpGBwdScalePtrs<T, TCompute>,
                                                     CKArgs<T, TCompute>,
                                                     miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                default:
                    return InitInvokerFactoryBwdNCHW<3,
                                                     false,
                                                     DeviceOpGBwdDefaultPtrs<T, TCompute>,
                                                     CKArgs<T, TCompute>,
                                                     miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                }
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                switch(problem->GetAlphaBetaCase())
                {
                case BILINEAR:
                    return InitInvokerFactoryNHWC<false,
                                                  DeviceOpGBwdBilinearPtrs<T, TCompute>,
                                                  CKArgs<T, TCompute>,
                                                  miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                case SCALE:
                    return InitInvokerFactoryNHWC<false,
                                                  DeviceOpGBwdScalePtrs<T, TCompute>,
                                                  CKArgs<T, TCompute>,
                                                  miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                default:
                    return InitInvokerFactoryNHWC<false,
                                                  DeviceOpGBwdDefaultPtrs<T, TCompute>,
                                                  CKArgs<T, TCompute>,
                                                  miopen::conv::DataInvokeParams>(
                        *ctx, *problem, kid);
                }
            },
            use_tf32);

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
