// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_grouped_conv_common.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>

#include <vector>
#include <string>
#include <memory>

namespace {

using miopen::conv::ProblemDescription;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGWrwPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    miopen::solver::conv::DeviceOpGWrw<DataType, ComputeType>>;

// CKArgs — WRW direction.
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
                    ConstData_t x,
                    Data_t dw,
                    ConstData_t dy,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             output,
                                             out_strides,
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
                    const miopen::ConvWrwTensors& tensors,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        return MakeArgPtr(conv_ptr, tensors.x, tensors.dw, tensors.dy, alpha, beta, split_k);
    }
};

template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    return CheckCKApplicabilityCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    return FillValidKernelsCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
    return CheckIsArgSupportedCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(
        problem, kernel_id, use_tf32);
}

template <typename DataType>
size_t GetWorkspaceSize(const ProblemDescription& problem, bool use_tf32)
{
    return GetWorkspaceSizeCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// extern "C" WRW implementations
// ---------------------------------------------------------------------------

extern "C" {

ck_impl_status_t ck_impl_wrw_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
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

ck_impl_status_t ck_impl_wrw_is_applicable(const miopen::conv::ProblemDescription* problem,
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

ck_impl_status_t ck_impl_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
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

ck_impl_status_t ck_impl_wrw_get_workspace_size(const miopen::conv::ProblemDescription* problem,
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

ck_impl_status_t ck_impl_wrw_get_solution(const miopen::ExecutionContext* ctx,
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

        auto solution = miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops(
            *problem,
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return miopen::solver::InitInvokerFactoryWrwNCHW<2,
                                                                 false,
                                                                 DeviceOpGWrwPtrs<T, TCompute>,
                                                                 CKArgs,
                                                                 miopen::conv::WrWInvokeParams>(
                    *ctx, *problem, kid);
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                return miopen::solver::InitInvokerFactoryNHWC<false,
                                                              DeviceOpGWrwPtrs<T, TCompute>,
                                                              CKArgs,
                                                              miopen::conv::WrWInvokeParams>(
                    *ctx, *problem, kid);
            },
            use_tf32);

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}

ck_impl_status_t ck_impl_wrw_get_all_kernel_type_strings(CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        auto result = std::make_unique<CKKernelListHandle>();

        auto ptrs = DeviceOpGWrwPtrs<float>::GetInstances();
        result->kernels.reserve(ptrs.size());
        for(const auto& ptr : ptrs)
            result->kernels.push_back(ptr->GetTypeString());

        *out_handle = result.release();
    });
}
} // extern "C"
