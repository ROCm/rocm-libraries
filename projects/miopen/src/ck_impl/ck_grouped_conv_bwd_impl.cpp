// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>

using miopen::conv::ProblemDescription;
using miopen::solver::ConvSolution;
using miopen::solver::InitInvokerFactoryBwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;

using miopen::solver::conv::DeviceOpGBwdPtrs;

namespace {

// CKArgs — BWD direction.
// Inherits shared members and split-k methods from CKArgsSplitK.
// Provides only the direction-specific MakeArgPtr overloads.
struct CKArgs : CKArgsSplitK<CKArgs>
{
    CKArgs(const ProblemDescription& problem) : CKArgsSplitK<CKArgs>(problem) {}

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    Data_t in,
                    ConstData_t w,
                    ConstData_t out,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             output,
                                             out_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             input,
                                             in_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const miopen::ConvDataTensors& tensors,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in, alpha, beta, split_k);
    }
};

template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    return CheckCKApplicabilityCommon<DeviceOpGBwdPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    return FillValidKernelsCommon<DeviceOpGBwdPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    return CheckIsArgSupportedCommon<DeviceOpGBwdPtrs, CKArgs, DataType>(
        problem, kernel_id, use_tf32);
}

template <typename DataType>
size_t GetWorkspaceSize(const ProblemDescription& problem, bool use_tf32)
{
    return GetWorkspaceSizeCommon<DeviceOpGBwdPtrs, CKArgs, DataType>(problem, use_tf32);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// extern "C" BWD implementation
// ---------------------------------------------------------------------------

extern "C" CKKernelListHandle* ckgrpconv_bwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32)
{
    try
    {
        auto result     = std::make_unique<CKKernelListHandle>();
        result->kernels = DispatchByDataType(data_type, [&](auto type_val) {
            return FillValidKernels<decltype(type_val)>(*problem, use_tf32);
        });
        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

extern "C" bool ckgrpconv_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                                            miopenDataType_t data_type,
                                            bool use_tf32)
{
    try
    {
        return DispatchByDataType(data_type, [&](auto type_val) {
            return CheckCKApplicability<decltype(type_val)>(*problem, use_tf32);
        });
    }
    catch(...)
    {
        return false;
    }
}

extern "C" bool ckgrpconv_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                                const char* kernel_id,
                                                miopenDataType_t data_type,
                                                bool use_tf32)
{
    try
    {
        if(!kernel_id)
            return false;
        std::string kid(kernel_id);
        return DispatchByDataType(data_type, [&](auto type_val) {
            return CheckIsArgSupported<decltype(type_val)>(*problem, kid, use_tf32);
        });
    }
    catch(...)
    {
        return false;
    }
}

extern "C" size_t ckgrpconv_bwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                                   miopenDataType_t data_type,
                                                   bool use_tf32)
{
    try
    {
        return DispatchByDataType(data_type, [&](auto type_val) {
            return GetWorkspaceSize<decltype(type_val)>(*problem, use_tf32);
        });
    }
    catch(...)
    {
        return 0;
    }
}

extern "C" miopen::solver::ConvSolution*
ckgrpconv_bwd_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32)
{
    try
    {
        if(!ctx || !problem || !kernel_id)
            return nullptr;
        std::string kid(kernel_id);

        auto solution = MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryBwdNCHW<2,
                                                 false,
                                                 DeviceOpGBwdPtrs<T, TCompute>,
                                                 CKArgs,
                                                 miopen::conv::DataInvokeParams>(
                    *ctx, *problem, kid);
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdPtrs<T, TCompute>,
                                              CKArgs,
                                              miopen::conv::DataInvokeParams>(*ctx, *problem, kid);
            },
            use_tf32);

        return new ConvSolution(std::move(solution));
    }
    catch(...)
    {
        return nullptr;
    }
}
