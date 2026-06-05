// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"
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

extern "C" ck_impl_status_t
ck_impl_bwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
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
ck_impl_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
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
ck_impl_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
ck_impl_bwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_size = DispatchByDataType(data_type, [&](auto type_val) {
            return GetWorkspaceSize<decltype(type_val)>(*problem, use_tf32);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_bwd_get_solution(const miopen::ExecutionContext* ctx,
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

        *out_solution = new ConvSolution(std::move(solution));
    });
}

extern "C" ck_impl_status_t ck_impl_bwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        auto result = std::make_unique<CKKernelListHandle>();

        auto ptrs = DeviceOpGBwdPtrs<float>::GetInstances();
        result->kernels.reserve(ptrs.size());
        for(const auto& ptr : ptrs)
            result->kernels.push_back(ptr->GetTypeString());

        *out_handle = result.release();
    });
}
