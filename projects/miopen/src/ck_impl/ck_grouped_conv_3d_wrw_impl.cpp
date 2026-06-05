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
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include "implicitgemm_ck_util.hpp"

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;
using Scale       = ck::tensor_operation::element_wise::Scale;

template <typename DataType, typename ComputeType = DataType>
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G               = ProblemInterpreter::GetGroupCountG(problem);
        N               = ProblemInterpreter::GetBatchN(problem);
        K1              = ProblemInterpreter::GetOutputChannelK(problem);
        C1              = ProblemInterpreter::GetInputChannelC(problem);
        C               = C1 / G;
        K               = K1 / G;
        Hi              = ProblemInterpreter::GetInputHeightHi(problem);
        Wi              = ProblemInterpreter::GetInputWidthWi(problem);
        Ho              = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo              = ProblemInterpreter::GetOutputWidthWo(problem);
        Y               = ProblemInterpreter::GetFilterHeightY(problem);
        X               = ProblemInterpreter::GetFilterWidthX(problem);
        Di              = ProblemInterpreter::GetInputDepthDi(problem);
        Do              = ProblemInterpreter::GetOutputDepthDo(problem);
        Z               = ProblemInterpreter::GetFilterDepthZ(problem);
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);

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
                    ConstData_t x,
                    Data_t dw,
                    ConstData_t dy,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        using DeviceP = std::remove_pointer_t<decltype(conv_ptr.get())>;
        if constexpr(std::is_same_v<
                         DeviceP,
                         miopen::solver::conv::DeviceOpGBwdWeightBilinear<DataType, ComputeType>>)
        {
            return MakeBilinearArgPtr(conv_ptr, x, dw, dy, alpha, beta, split_k);
        }
        else if constexpr(std::is_same_v<
                              DeviceP,
                              miopen::solver::conv::DeviceOpGBwdWeightScale<DataType, ComputeType>>)
        {
            (void)beta;
            return MakeScaleArgPtr(conv_ptr, x, dw, dy, alpha, split_k);
        }
        else
        {
            (void)alpha;
            (void)beta;
            static_assert(
                std::is_same_v<
                    DeviceP,
                    miopen::solver::conv::DeviceOpGBwdWeightDefault<DataType, ComputeType>>,
                "Default should be wrw pass through");
            return MakeDefaultArgPtr(conv_ptr, x, dw, dy, split_k);
        }
    }

    template <typename ConvPtr>
    auto MakeBilinearArgPtr(const ConvPtr& conv_ptr,
                            ConstData_t x,
                            Data_t dw,
                            ConstData_t dy,
                            float alpha,
                            float beta,
                            int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             {dw},
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             {wei_lengths},
                                             {wei_strides},
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             Bilinear{alpha, beta},
                                             PassThrough{},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeScaleArgPtr(const ConvPtr& conv_ptr,
                         ConstData_t x,
                         Data_t dw,
                         ConstData_t dy,
                         float alpha,
                         int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             {},
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             {},
                                             {},
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             Scale{alpha},
                                             PassThrough{},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeDefaultArgPtr(
        const ConvPtr& conv_ptr, ConstData_t x, Data_t dw, ConstData_t dy, int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             PassThrough{},
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

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr        = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, 1);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr        = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    std::size_t GetCKSplitkWorkspaceSize(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        return conv_ptr->GetWorkSpaceSize(arg_ptr.get());
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
    miopenAlphaBetaCase_t alpha_beta_case;
    miopenDataType_t data_type;
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

template <typename DataType, typename ComputeType>
std::vector<std::string> FillValidKernelsByAlphaBeta(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::FillValidKernelsIDs<
            miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return miopen::solver::FillValidKernelsIDs<
            miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    default:
        return miopen::solver::FillValidKernelsIDs<
            miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    }
}

template <typename DataType, typename ComputeType>
bool CheckCKApplicabilityByAlphaBeta(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::IsCKApplicable<
            miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return miopen::solver::IsCKApplicable<
            miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    default:
        return miopen::solver::IsCKApplicable<
            miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    }
}

template <typename DataType, typename ComputeType>
bool CheckIsArgSupportedByAlphaBeta(const ProblemDescription& problem, const std::string& kernel_id)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::IsCKArgsSupported<
            miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem, kernel_id);
    case SCALE:
        return miopen::solver::IsCKArgsSupported<
            miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem, kernel_id);
    default:
        return miopen::solver::IsCKArgsSupported<
            miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem, kernel_id);
    }
}

template <typename DataType, typename ComputeType>
std::size_t GetWorkspaceSizeByAlphaBeta(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::GetCKSplitkMaxWorkspaceSize<
            miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return miopen::solver::GetCKSplitkMaxWorkspaceSize<
            miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    default:
        return miopen::solver::GetCKSplitkMaxWorkspaceSize<
            miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<DataType, ComputeType>,
            CKArgs<DataType, ComputeType>>(problem);
    }
}

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

template <typename DataType>
std::size_t GetWorkspaceSize(const ProblemDescription& problem, bool use_tf32)
{
    auto ws = GetWorkspaceSizeByAlphaBeta<DataType, DataType>(problem);
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
        {
            ws = std::max(ws, GetWorkspaceSizeByAlphaBeta<DataType, ck::tf32_t>(problem));
        }
    }
    return ws;
}

} // namespace

extern "C" ck_impl_status_t
ck_impl_3d_wrw_get_all_kernel_type_strings(CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        auto result = std::make_unique<CKKernelListHandle>();

        auto bilinear_ptrs =
            miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<float>::GetInstances();
        auto scale_ptrs = miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<float>::GetInstances();
        auto default_ptrs =
            miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<float>::GetInstances();

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

using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::InitInvokerFactoryWrwNCHW;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;

extern "C" ck_impl_status_t
ck_impl_3d_wrw_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
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
ck_impl_3d_wrw_is_applicable(const miopen::conv::ProblemDescription* problem,
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
ck_impl_3d_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
ck_impl_3d_wrw_get_workspace_size(const miopen::conv::ProblemDescription* problem,
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
ck_impl_3d_wrw_get_solution(const miopen::ExecutionContext* ctx,
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
                    return InitInvokerFactoryWrwNCHW<
                        3,
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                case SCALE:
                    return InitInvokerFactoryWrwNCHW<
                        3,
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                default:
                    return InitInvokerFactoryWrwNCHW<
                        3,
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                }
            },
            [&](auto data_type_val, auto compute_type_val) {
                using T        = decltype(data_type_val);
                using TCompute = decltype(compute_type_val);
                switch(problem->GetAlphaBetaCase())
                {
                case BILINEAR:
                    return InitInvokerFactoryNHWC<
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightBilinearPtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                case SCALE:
                    return InitInvokerFactoryNHWC<
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightScalePtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                default:
                    return InitInvokerFactoryNHWC<
                        false,
                        miopen::solver::conv::DeviceOpGBwdWeightDefaultPtrs<T, TCompute>,
                        CKArgs<T, TCompute>,
                        miopen::conv::WrWInvokeParams>(*ctx, *problem, kid);
                }
            },
            use_tf32);

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
