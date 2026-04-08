// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"
#include <miopen/solver/problem_description_interpreter.hpp>
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for fused Conv+ScaleAdd+Bias+ReLU
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

using CK_OutLayout = ck::tensor_layout::convolution::NDHWGK;

// DataType also applies to weights
// AccumDataType also applies to added z & bias tensors
template <typename DataType, typename AccumDataType = DataType>
using DeviceOp = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        3,
        ck::tensor_layout::convolution::NDHWGC,
        ck::tensor_layout::convolution::GKZYXC,
        ck::Tuple<CK_OutLayout, ck::tensor_layout::convolution::G_K>,
        CK_OutLayout,
        DataType,                                // in data type
        DataType,                                // wei data type
        ck::Tuple<AccumDataType, AccumDataType>, // z & bias tensors data type
        DataType,                                // out data type
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::
            ScaleAddScaleAddRelu>>; // end DeviceOperationInstanceFactory

// ---------------------------------------------------------------------------
// CKArgs -- fused Conv+ScaleAdd+Bias+ReLU (from solver lines 74-235).
// ---------------------------------------------------------------------------

struct CKArgs
{
    CKArgs(const miopen::conv::ProblemDescription& problem)
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

        in_lens      = {G, N, C, Di, Hi, Wi};
        out_lens     = {G, N, K, Do, Ho, Wo};
        wei_lens     = {G, K, C, Z, Y, X};
        bias_lens    = {G, 1, K, 1, 1, 1};
        bias_strides = {K, 0, 1, 0, 0, 0};

        // miopen filter_stride to CK filter_stride
        auto miopen_in_strides  = problem.GetIn().GetStrides();
        auto miopen_out_strides = problem.GetOut().GetStrides();
        auto miopen_wei_strides = problem.GetWeights().GetStrides();
        miopen_in_strides.insert(miopen_in_strides.begin(), C);
        miopen_out_strides.insert(miopen_out_strides.begin(), K);
        miopen_wei_strides.insert(miopen_wei_strides.begin(), K * miopen_wei_strides[0]);
        std::copy(miopen_in_strides.begin(), miopen_in_strides.end(), in_strides.begin());
        std::copy(miopen_out_strides.begin(), miopen_out_strides.end(), out_strides.begin());
        std::copy(miopen_wei_strides.begin(), miopen_wei_strides.end(), wei_strides.begin());

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                           ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadD(problem),
                           ProblemInterpreter::GetInputLeftPadH(problem),
                           ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                           ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                           ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    ConstData_t in_buf,
                    ConstData_t wei_buf,
                    Data_t out_buf,
                    ConstData_t z_buf,
                    ConstData_t bias_buf,
                    float alpha1,
                    float alpha2) const
    {
        using ScaleAddScaleAddRelu = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
        return op_ptr->MakeArgumentPointer(in_buf,
                                           wei_buf,
                                           {z_buf, bias_buf},
                                           out_buf,
                                           in_lens,
                                           in_strides,
                                           wei_lens,
                                           wei_strides,
                                           {out_lens, bias_lens},
                                           {out_strides, bias_strides},
                                           out_lens,
                                           out_strides,
                                           filter_stride,
                                           filter_dilation,
                                           lPadding,
                                           rPadding,
                                           {}, // PassThrough
                                           {}, // PassThrough
                                           ScaleAddScaleAddRelu{alpha1, alpha2});
    }

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    const miopen::fusion::FusionInvokeParams& data_ctx) const
    {
        const auto& conv_param =
            dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*data_ctx.op_args.params[0]);
        assert(&conv_param);

        const auto& z_param =
            dynamic_cast<miopen::fusion::TensorScaleAddOpInvokeParam&>(*data_ctx.op_args.params[1]);
        assert(&z_param);

        const auto& bias_param =
            dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*data_ctx.op_args.params[2]);
        assert(&bias_param);

        /// \todo: Support general activation functions.
        /// only relu activation supported and hardcoded for now
        [[maybe_unused]] const auto& activ_param =
            dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(*data_ctx.op_args.params[3]);
        assert(&activ_param);

        return MakeArgPtr(op_ptr,
                          data_ctx.in,
                          conv_param.weights,
                          data_ctx.out,
                          z_param.tensor_ptr,
                          bias_param.bdata,
                          conv_param.alpha,
                          z_param.alpha);
    }

    template <typename DevOpPtr>
    bool IsSupportedBy(const DevOpPtr& op_ptr) const
    {
        auto arg_ptr = MakeArgPtr(op_ptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0, 1.0);
        return op_ptr->IsSupportedArgument(arg_ptr.get());
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
    std::array<ck::index_t, 6> in_lens;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lens;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lens;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 6> bias_lens;
    std::array<ck::index_t, 6> bias_strides;
    std::array<ck::index_t, 3> filter_stride;
    std::array<ck::index_t, 3> filter_dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

// ---------------------------------------------------------------------------
// Template helpers
// ---------------------------------------------------------------------------

template <typename DataType, typename AccumDataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    return miopen::solver::FillValidKernelsIDs<DeviceOp<DataType, AccumDataType>, CKArgs>(problem);
}

template <typename DataType, typename AccumDataType>
bool CheckCKApplicability(const ProblemDescription& problem)
{
    return miopen::solver::IsCKApplicable<DeviceOp<DataType, AccumDataType>, CKArgs>(problem);
}

template <typename DataType, typename AccumDataType>
bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    return miopen::solver::IsCKArgsSupported<DeviceOp<DataType, AccumDataType>, CKArgs>(problem,
                                                                                        kernel_id);
}

// ---------------------------------------------------------------------------
// Data-type dispatch helpers.
//
// This solver supports int8_t with float accumulation, so the standard
// DispatchByDataType (which maps int8_t to int8_t AccumDataType) cannot
// be reused directly.  Instead, we dispatch manually with the correct
// (DataType, AccumDataType) pairs.
// ---------------------------------------------------------------------------

template <typename Fn>
auto DispatchFusedByDataType(miopenDataType_t dtype, Fn&& fn)
{
    switch(dtype)
    {
    case miopenHalf: return fn(ck::half_t{}, ck::half_t{});
    case miopenBFloat16: return fn(ck::bhalf_t{}, ck::bhalf_t{});
    case miopenInt8: return fn(int8_t{}, float{});
    default: return fn(float{}, float{});
    }
}

} // anonymous namespace

// ===========================================================================
// Fused Conv+ScaleAdd+Bias+ReLU extern "C" functions
// ===========================================================================

using miopen::solver::InitAnyInvokerFactory;

extern "C" ck_impl_status_t
ck_impl_fused_bias_res_add_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                                    miopenDataType_t data_type,
                                                    bool /*use_tf32*/,
                                                    CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result     = std::make_unique<CKKernelListHandle>();
        result->kernels = DispatchFusedByDataType(data_type, [&](auto data_val, auto accum_val) {
            return FillValidKernels<decltype(data_val), decltype(accum_val)>(*problem);
        });
        *out_handle     = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_bias_res_add_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                               miopenDataType_t data_type,
                                               bool /*use_tf32*/,
                                               bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_result = DispatchFusedByDataType(data_type, [&](auto data_val, auto accum_val) {
            return CheckCKApplicability<decltype(data_val), decltype(accum_val)>(*problem);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_bias_res_add_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
        *out_result = DispatchFusedByDataType(data_type, [&](auto data_val, auto accum_val) {
            return CheckIsArgSupported<decltype(data_val), decltype(accum_val)>(*problem, kid);
        });
    });
}

extern "C" ck_impl_status_t ck_impl_fused_bias_res_add_activ_get_workspace_size(
    const miopen::conv::ProblemDescription* /*problem*/,
    miopenDataType_t /*data_type*/,
    bool /*use_tf32*/,
    size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        // Fused Conv+ScaleAdd+Bias+ReLU CK kernels do not require CK-level workspace.
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_bias_res_add_activ_get_solution(const miopen::ExecutionContext* /*ctx*/,
                                              const miopen::conv::ProblemDescription* problem,
                                              const char* kernel_id,
                                              bool /*use_tf32*/,
                                              miopen::solver::ConvSolution** out_solution)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_solution, CK_IMPL_STATUS_BAD_PARAM, "Null out_solution");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");

        std::string kid(kernel_id);

        using ParamType = miopen::fusion::FusionInvokeParams;

        miopen::solver::ConvSolution solution;
        switch(problem->GetInDataType())
        {
        case miopenInt8:
            solution =
                InitAnyInvokerFactory<DeviceOp<int8_t, float>, CKArgs, ParamType>(*problem, kid);
            break;
        case miopenHalf:
            solution = InitAnyInvokerFactory<DeviceOp<ck::half_t, ck::half_t>, CKArgs, ParamType>(
                *problem, kid);
            break;
        case miopenFloat:
            solution =
                InitAnyInvokerFactory<DeviceOp<float, float>, CKArgs, ParamType>(*problem, kid);
            break;
        case miopenBFloat16:
            solution = InitAnyInvokerFactory<DeviceOp<ck::bhalf_t, ck::bhalf_t>, CKArgs, ParamType>(
                *problem, kid);
            break;
        default: throw CkImplException(CK_IMPL_STATUS_INVALID_VALUE, "Unsupported data type");
        }

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
