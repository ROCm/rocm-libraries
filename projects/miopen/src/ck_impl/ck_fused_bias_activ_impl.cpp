// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include "implicitgemm_ck_util.hpp"
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>

// ---------------------------------------------------------------------------
// CK type aliases and instance function for Conv+Bias+ReLU fusion (FP16)
// ---------------------------------------------------------------------------

using DeviceConvFwdBiasReluPtr = ck::tensor_operation::device::DeviceConvFwdBiasActivationPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddRelu>;

// Forward-declare CK's explicit instance population function.
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdBiasReluPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

// ---------------------------------------------------------------------------
// CKArgs -- 2D fused Conv+Bias+ReLU (older CK API, std::vector-based).
// ---------------------------------------------------------------------------

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        input    = {ProblemInterpreter::GetInputHeightHi(problem),
                    ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(problem),
                    ProblemInterpreter::GetOutputWidthWo(problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(problem),
                    ProblemInterpreter::GetFilterWidthX(problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    int N;
    int K;
    int C;
    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> filter;
    std::vector<int> strides;
    std::vector<int> dilation;
    std::vector<int> lPadding;
    std::vector<int> rPadding;
};

// ---------------------------------------------------------------------------
// Helper: get the populated list of CK device operator instances.
// ---------------------------------------------------------------------------

std::vector<DeviceConvFwdBiasReluPtr> GetConvInstances()
{
    std::vector<DeviceConvFwdBiasReluPtr> conv_ptrs;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    return conv_ptrs;
}

// ---------------------------------------------------------------------------
// Template-free helpers for fill/applicable/args_supported.
// These replicate the logic from the solver but operate directly on the
// explicit instance list (no DeviceOperationInstanceFactory).
// ---------------------------------------------------------------------------

std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    const auto args = CKArgs{problem};
    auto conv_ptrs  = GetConvInstances();
    std::vector<std::string> valid_kernels;
    valid_kernels.reserve(conv_ptrs.size());
    for(const auto& it : conv_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.N,
                                                    args.K,
                                                    args.C,
                                                    args.input,
                                                    args.filter,
                                                    args.output,
                                                    args.strides,
                                                    args.dilation,
                                                    args.lPadding,
                                                    args.rPadding,
                                                    {},
                                                    {},
                                                    {});
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(it->GetTypeString());
        }
    }
    return valid_kernels;
}

bool CheckCKApplicability(const ProblemDescription& problem)
{
    const auto args = CKArgs{problem};
    auto conv_ptrs  = GetConvInstances();
    for(const auto& it : conv_ptrs)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    args.N,
                                                    args.K,
                                                    args.C,
                                                    args.input,
                                                    args.filter,
                                                    args.output,
                                                    args.strides,
                                                    args.dilation,
                                                    args.lPadding,
                                                    args.rPadding,
                                                    {},
                                                    {},
                                                    {});
        if(it->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    const auto args = CKArgs{problem};
    auto conv_ptrs  = GetConvInstances();
    for(size_t i = 0; i < conv_ptrs.size(); ++i)
    {
        if(conv_ptrs[i]->GetTypeString() == kernel_id)
        {
            auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  args.N,
                                                                  args.K,
                                                                  args.C,
                                                                  args.input,
                                                                  args.filter,
                                                                  args.output,
                                                                  args.strides,
                                                                  args.dilation,
                                                                  args.lPadding,
                                                                  args.rPadding,
                                                                  {},
                                                                  {},
                                                                  {});
            return conv_ptrs[i]->IsSupportedArgument(argument_ptr.get());
        }
    }
    return false;
}

} // anonymous namespace

// ===========================================================================
// Fused Conv+Bias+ReLU extern "C" functions
// ===========================================================================

extern "C" ck_impl_status_t
ck_impl_fused_bias_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fused_bias_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fused_bias_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
ck_impl_fused_bias_activ_get_workspace_size(const miopen::conv::ProblemDescription* /*problem*/,
                                            miopenDataType_t /*data_type*/,
                                            bool /*use_tf32*/,
                                            size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_bias_activ_get_solution(const miopen::ExecutionContext* ctx,
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
        solution.invoker_factory = [kid, conv_problem = *problem](
                                       const std::vector<miopen::Kernel>& kernels) {
            std::ignore = kernels;
            return [kid, conv_problem](const miopen::Handle& handle,
                                       const miopen::AnyInvokeParams& primitive_parameters) {
                const auto args = CKArgs{conv_problem};
                auto conv_ptrs  = GetConvInstances();

                // Find the instance matching the kernel_id.
                size_t id = 0;
                for(; id < conv_ptrs.size(); ++id)
                {
                    if(conv_ptrs[id]->GetTypeString() == kid)
                        break;
                }
                assert(id < conv_ptrs.size());
                auto& conv_ck = conv_ptrs.at(id);

                // Extract tensors from FusionInvokeParams.
                const auto& invoke_ctx =
                    primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
                const auto& wei_buf = dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(
                                          *invoke_ctx.op_args.params[0])
                                          .weights;
                const auto& bias_buf =
                    dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*invoke_ctx.op_args.params[1])
                        .bdata;

                auto argument_ptr = conv_ck->MakeArgumentPointer(
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(invoke_ctx.in)),
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(wei_buf)),
                    invoke_ctx.out,
                    const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                        static_cast<const void*>(bias_buf)),
                    args.N,
                    args.K,
                    args.C,
                    args.input,
                    args.filter,
                    args.output,
                    args.strides,
                    args.dilation,
                    args.lPadding,
                    args.rPadding,
                    {},
                    {},
                    {});

                auto invoker_ptr            = conv_ck->MakeInvokerPointer();
                const auto enable_profiling = handle.IsProfilingEnabled();

                float elapsed_time =
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
                if(enable_profiling)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed_time);
                }
            };
        };

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
