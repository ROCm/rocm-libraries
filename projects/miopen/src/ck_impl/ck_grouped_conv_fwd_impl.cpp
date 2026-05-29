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
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for grouped convolution forward
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGK,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGFwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwd<DataType, ComputeType>>;

// ---------------------------------------------------------------------------
// CKArgs — extracts convolution dimensions for CK argument construction.
//
// Each direction (fwd, bwd, wrw) has its own CKArgs with the same dimension
// members but direction-specific MakeArgPtr tensor ordering and IsSupportedBy
// logic.  FWD does not use split-k.  BWD/WRW additionally handle NHWC layout
// strides and split-k workspace queries.
// ---------------------------------------------------------------------------
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        auto d = ExtractConvDims(problem);
        G      = d.G;
        N      = d.N;
        K1     = d.K1;
        C1     = d.C1;
        C      = d.C;
        K      = d.K;
        Hi     = d.Hi;
        Wi     = d.Wi;
        Ho     = d.Ho;
        Wo     = d.Wo;
        Y      = d.Y;
        X      = d.X;

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        // strides from NHWGC to GNCHW layout
        in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
        out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
        wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        strides     = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation    = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                       ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding    = {ProblemInterpreter::GetInputLeftPadH(problem),
                       ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding    = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                       ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const miopen::ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K1;
    int C1;
    int K;
    int C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    return CheckCKApplicabilityCommon<DeviceOpGFwdPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    return FillValidKernelsCommon<DeviceOpGFwdPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    return CheckIsArgSupportedCommon<DeviceOpGFwdPtrs, CKArgs, DataType>(
        problem, kernel_id, use_tf32);
}

} // anonymous namespace

// ===========================================================================
// FWD direction functions
// ===========================================================================

using miopen::solver::InitInvokerFactoryFwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;

extern "C" ck_impl_status_t
ck_impl_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fwd_get_workspace_size(const miopen::conv::ProblemDescription* /*problem*/,
                               miopenDataType_t /*data_type*/,
                               bool /*use_tf32*/,
                               size_t* out_size)
{
    // FWD grouped convolution CK kernels do not use split-k and never
    // require CK-level workspace.  Layout transform workspace (NCHW to NHWC)
    // is computed independently at the solver level via
    // GetWorkspaceSizeLayoutTransformConv().
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_fwd_get_solution(const miopen::ExecutionContext* ctx,
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

        auto solution = MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryFwdNCHW<2,
                                                 false,
                                                 DeviceOpGFwdPtrs<T, TCompute>,
                                                 CKArgs,
                                                 miopen::conv::DataInvokeParams>(
                    *ctx, *problem, std::string(kernel_id));
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGFwdPtrs<T, TCompute>,
                                              CKArgs,
                                              miopen::conv::DataInvokeParams>(
                    *ctx, *problem, std::string(kernel_id));
            },
            use_tf32);
        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}

extern "C" ck_impl_status_t ck_impl_fwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        auto result = std::make_unique<CKKernelListHandle>();

        auto ptrs = DeviceOpGFwdPtrs<float>::GetInstances();
        result->kernels.reserve(ptrs.size());
        for(const auto& ptr : ptrs)
            result->kernels.push_back(ptr->GetTypeString());

        *out_handle = result.release();
    });
}
