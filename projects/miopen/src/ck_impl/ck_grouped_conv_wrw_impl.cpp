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
// Older installed CK archives do not ship this instance; build native routing only with a
// matching CK header and enabled WMMA/BF16 registrar, leaving existing WRW staging unchanged.
#if defined(MIOPEN_CK_GFX1250_NCHW_WRW) && defined(CK_USE_WMMA) && defined(CK_ENABLE_BF16)
#if __has_include( \
    "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_nchw_pointwise_bf16_wmma.hpp")
#define MIOPEN_CK_NATIVE_POINTWISE_WRW 1
#endif
#endif
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp>
#endif

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <optional>

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
        // Large-tensor (>INT_MAX element stride) instances expose CK's int64
        // long_index_t MakeArgumentPointer overload; bind it with the int64
        // member arrays directly (they outlive the returned arg_ptr). This
        // mirrors the FWD path in ck_grouped_conv_fwd_impl.cpp.
        if(miopen::solver::IsLargeTensorCKInstance(conv_ptr))
        {
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
        // Sub-INT_MAX shapes: narrow to int32 at the boundary. The narrowed
        // bundle is a mutable member of CKArgs (populated by GetNarrowedArrays)
        // so its arrays outlive any arg_ptr referencing them -- CK's
        // MakeArgumentPointer captures references into the bundle.
        const auto& a = this->GetNarrowedArrays();
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             a.in_l,
                                             a.in_s,
                                             a.wei_l,
                                             a.wei_s,
                                             a.out_l,
                                             a.out_s,
                                             a.filter_strides,
                                             a.filter_dilations,
                                             a.lPadding,
                                             a.rPadding,
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
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
// The MIOpen CK archive contains the native registrar, but not the unrelated
// NGCHW XDL registrars instantiated by the complete NGCHW factory. Build the
// same typed operation vector through the single registered native entrypoint.
using NativeDeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::NGCHW,
    ck::tensor_layout::convolution::GKCYX,
    ck::tensor_layout::convolution::NGKHW,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::bhalf_t,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

constexpr char NativeKernelName[] = "DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>";

auto GetNativeInstances()
{
    std::vector<std::unique_ptr<NativeDeviceOp>> instances;
    ck::tensor_operation::device::instance::
        add_device_grouped_conv2d_bwd_weight_nchw_pointwise_bf16_wmma_instances(instances);
    return instances;
}

std::optional<int> NativeSplit(const std::string& kernel_id)
{
    const std::string name{NativeKernelName};
    if(kernel_id == name + "+1")
        return 1;
    if(kernel_id == name + "+0")
        return 0;
    if(kernel_id == name + "+-1")
        return -1;
    return std::nullopt;
}

// CKArgsSplitK's default-layout strides describe the existing NHWGC staging
// buffers, not the caller's physical NCHW tensors. Keep those strides intact
// for the fallback, and pass the real packed NCHW strides only to this op.
struct NativeCKArgs : CKArgs
{
    explicit NativeCKArgs(const ProblemDescription& problem) : CKArgs(problem)
    {
        const auto in_s  = Hi * Wi;
        const auto out_s = Ho * Wo;
        in_strides       = {N * C * in_s, C * in_s, in_s, Wi, 1};
        out_strides      = {N * K * out_s, K * out_s, out_s, Wo, 1};
        wei_strides      = {K * C, C, 1, 1, 1};
    }
};

bool NativeProblemShape(const ProblemDescription& problem)
{
    if(!problem.IsDirectionBackwardWrW() || !problem.Is2d() || !problem.IsLayoutDefault() ||
       !problem.IsBfp16() || problem.HasNonPackedTensors() ||
       problem.GetAlphaBetaCase() != DEFAULT || problem.GetAlpha().GetAsFloat() != 1.0f ||
       problem.GetBeta().GetAsFloat() != 0.0f || problem.GetConv().mode != miopenConvolution)
        return false;

    const auto d = ExtractConvDims(problem);
    if(d.Hi < 1 || d.Wi < 1 || d.Hi > 76800 || d.Wi > 76800)
        return false;
    const auto s = d.Hi * d.Wi;
    return d.G == 1 && d.N > 0 && d.N <= 42 && d.C >= 8 && d.C <= 24 && d.C % 8 == 0 && d.K >= 16 &&
           d.K <= 128 && d.K % 16 == 0 && d.Y == 1 && d.X == 1 && d.Hi == d.Ho && d.Wi == d.Wo &&
           s >= 1024 && s <= 76800 && s % 512 == 0 && problem.GetKernelStrideH() == 1 &&
           problem.GetKernelStrideW() == 1 && problem.GetDilationH() == 1 &&
           problem.GetDilationW() == 1 && problem.GetPadH() == 0 && problem.GetPadW() == 0 &&
           problem.GetConv().trans_output_pads.size() == 2 &&
           problem.GetConv().trans_output_pads[0] == 0 &&
           problem.GetConv().trans_output_pads[1] == 0;
}

std::optional<std::string> StagedFallbackID(const ProblemDescription& problem)
{
    const CKArgs args{problem};
    auto instances = DeviceOpGWrwPtrs<ck::bhalf_t>::GetInstances();
    for(const auto& instance : instances)
    {
        if(args.IsSupportedBySplitK(instance, 1))
            return instance->GetTypeString() + "+1";
    }
    return std::nullopt;
}

bool NativeProblemSupported(const ProblemDescription& problem, int split)
{
    if(!NativeProblemShape(problem) || !StagedFallbackID(problem))
        return false;

    const NativeCKArgs args{problem};
    auto instances = GetNativeInstances();
    auto instance  = miopen::solver::FindConvPtrByID(instances, NativeKernelName);
    if(instance == instances.end())
        return false;
    // No invented device pointer is installed in the dry argument. CK validates
    // the dimensions, strides, split, architecture, and data type here.
    const auto argument = args.MakeArgPtr(*instance, nullptr, nullptr, nullptr, 1.0f, 0.0f, split);
    return (*instance)->IsSupportedArgument(argument.get());
}

miopen::solver::ConvSolution MakeNativeSolution(const miopen::ExecutionContext& ctx,
                                                const ProblemDescription& problem,
                                                const std::string& kernel_id)
{
    const auto split = NativeSplit(kernel_id);
    if(!split || !NativeProblemSupported(problem, *split))
        return {miopenStatusInvalidValue};

    const auto fallback_id = StagedFallbackID(problem);
    auto staged            = miopen::solver::InitInvokerFactoryWrwNCHW<2,
                                                                       false,
                                                                       DeviceOpGWrwPtrs<ck::bhalf_t>,
                                                                       CKArgs,
                                                                       miopen::conv::WrWInvokeParams>(
        ctx, problem, *fallback_id);
    if(!staged.Succeeded() || !staged.invoker_factory)
        return {miopenStatusInvalidValue};

    auto instances = GetNativeInstances();
    auto instance  = miopen::solver::FindConvPtrByID(instances, NativeKernelName);
    if(instance == instances.end())
        return {miopenStatusInvalidValue};

    NativeCKArgs args{problem};
    auto dry_argument = args.MakeArgPtr(*instance, nullptr, nullptr, nullptr, 1.0f, 0.0f, *split);
    const auto native_workspace = (*instance)->GetWorkSpaceSize(dry_argument.get());
    staged.workspace_sz         = std::max(staged.workspace_sz, native_workspace);

    staged.invoker_factory = [fallback_factory = std::move(*staged.invoker_factory),
                              native_op = std::shared_ptr<NativeDeviceOp>{std::move(*instance)},
                              args      = std::move(args),
                              x_desc    = problem.GetOut(),
                              dw_desc   = problem.GetWeights(),
                              dy_desc   = problem.GetIn(),
                              kernel_id,
                              split          = *split,
                              workspace_size = staged.workspace_sz](
                                 const std::vector<miopen::Kernel>& kernels) mutable {
        auto fallback = fallback_factory(kernels);
        return [fallback  = std::move(fallback),
                native_op = std::move(native_op),
                args      = std::move(args),
                x_desc    = std::move(x_desc),
                dw_desc   = std::move(dw_desc),
                dy_desc   = std::move(dy_desc),
                kernel_id,
                split,
                workspace_size](const miopen::Handle& handle,
                                const miopen::AnyInvokeParams& primitive_parameters) mutable {
            const auto& invoke   = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors  = invoke.tensors;
            const auto workspace = invoke.workSpace;
            const auto workspace_aligned = reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0;
            // The staged converter also uses caller workspace. Neither the old
            // converter nor CK supports overlapping or missing staging scratch.
            const bool valid_workspace =
                workspace != nullptr && workspace_aligned &&
                invoke.workSpaceSize >= workspace_size &&
                miopen::solver::internal::DisjointByteRanges(
                    tensors.x, tensors.xDesc.GetNumBytes(), workspace, workspace_size) &&
                miopen::solver::internal::DisjointByteRanges(
                    tensors.dy, tensors.dyDesc.GetNumBytes(), workspace, workspace_size) &&
                miopen::solver::internal::DisjointByteRanges(
                    tensors.dw, tensors.dwDesc.GetNumBytes(), workspace, workspace_size);
            MIOPEN_THROW_IF(!valid_workspace,
                            "Insufficient, misaligned, or overlapping WRW workspace");

            const bool native_eligible = handle.GetDeviceName() == "gfx1250" &&
                                         invoke.alpha.GetAsFloat() == 1.0f &&
                                         invoke.beta.GetAsFloat() == 0.0f &&
                                         reinterpret_cast<std::uintptr_t>(workspace) % 256 == 0 &&
                                         miopen::solver::internal::CanBorrowNCHWOperands(
                                             static_cast<miopen::ConvTensors>(tensors),
                                             x_desc,
                                             dw_desc,
                                             dy_desc,
                                             workspace,
                                             workspace_size);
            if(!native_eligible)
            {
                fallback(handle, primitive_parameters);
                return;
            }

            auto argument =
                args.MakeArgPtr(native_op, tensors.x, tensors.dw, tensors.dy, 1.0f, 0.0f, split);
            native_op->SetWorkSpacePointer(argument.get(), workspace);
            if(!native_op->IsSupportedArgument(argument.get()))
            {
                fallback(handle, primitive_parameters);
                return;
            }

            handle.ResetKernelTime();
            auto invoker = native_op->MakeInvokerPointer();
            {
                miopen::solver::WorkAroundHipEventProfiler profiler(handle);
                MIOPEN_LOG_I2("kernel_name = " << kernel_id);
                invoker->Run(argument.get(), {handle.GetStream(), false});
            }
            if(handle.IsProfilingEnabled())
            {
                const float elapsed = handle.GetKernelTime();
                if(miopen::IsLoggingKernel())
                    miopen::AddKernelToJsonAccumulator(kernel_id, elapsed, false);
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    return staged;
}
#endif


template <typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem, bool use_tf32)
{
    return CheckCKApplicabilityCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
}

template <typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem, bool use_tf32)
{
    auto kernels = FillValidKernelsCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
    if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        if(NativeProblemSupported(problem, 1) &&
           std::find(kernels.begin(), kernels.end(), NativeKernelName) == kernels.end())
            kernels.emplace_back(NativeKernelName);
    }
#endif
    return kernels;
}

template <typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem,
                         const std::string& kernel_id,
                         bool use_tf32)
{
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
    if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        if(const auto split = NativeSplit(kernel_id))
            return NativeProblemSupported(problem, *split);
    }
#endif
    return CheckIsArgSupportedCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(
        problem, kernel_id, use_tf32);
}

template <typename DataType>
size_t GetWorkspaceSize(const ProblemDescription& problem, bool use_tf32)
{
    auto workspace = GetWorkspaceSizeCommon<DeviceOpGWrwPtrs, CKArgs, DataType>(problem, use_tf32);
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
    if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        if(NativeProblemSupported(problem, 1))
        {
            NativeCKArgs args{problem};
            auto instances = GetNativeInstances();
            auto instance  = miopen::solver::FindConvPtrByID(instances, NativeKernelName);
            auto argument  = args.MakeArgPtr(*instance, nullptr, nullptr, nullptr, 1.0f, 0.0f, 1);
            workspace      = std::max(workspace, (*instance)->GetWorkSpaceSize(argument.get()));
        }
    }
#endif
    return workspace;
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
#ifdef MIOPEN_CK_NATIVE_POINTWISE_WRW
        if(kid.rfind(NativeKernelName, 0) == 0)
        {
            *out_solution =
                new miopen::solver::ConvSolution(MakeNativeSolution(*ctx, *problem, kid));
            return;
        }
#endif

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
