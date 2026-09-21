// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

#include "ck_grouped_conv_common.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp>

// ---------------------------------------------------------------------------
// CK type aliases for fused grouped conv + activation (Clamp)
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::Clamp;

const auto in_element_op  = InElementOp{};
const auto wei_element_op = WeiElementOp{};

// ---------------------------------------------------------------------------
// Layout selectors for 2D and 3D
// ---------------------------------------------------------------------------

template <ck::index_t NDimSpatial>
struct LayoutsSelector;

template <>
struct LayoutsSelector<2>
{
    using InLayout  = ck::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKYXC;
    using OutLayout = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct LayoutsSelector<3>
{
    using InLayout  = ck::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck::tensor_layout::convolution::NDHWGK;
};

// ---------------------------------------------------------------------------
// DeviceOp template aliases (same as in solver)
// ---------------------------------------------------------------------------

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename InLayout     = ck::tensor_layout::convolution::NHWGC,
          typename WeiLayout    = ck::tensor_layout::convolution::GKYXC,
          typename OutLayout    = ck::tensor_layout::convolution::NHWGK>
using DeviceOpGFwdAct = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                                      InLayout,
                                                                                      WeiLayout,
                                                                                      ck::Tuple<>,
                                                                                      OutLayout,
                                                                                      InDataType,
                                                                                      WeiDataType,
                                                                                      ck::Tuple<>,
                                                                                      OutDataType,
                                                                                      InElementOp,
                                                                                      WeiElementOp,
                                                                                      OutElementOp,
                                                                                      AComputeType,
                                                                                      BComputeType>;

template <ck::index_t NumDimSpatial,
          typename DataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
using DeviceOpGFwdActPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwdAct<NumDimSpatial,
                    DataType,
                    DataType,
                    DataType,
                    DataType,
                    DataType,
                    InLayout,
                    WeiLayout,
                    OutLayout>>;

// ---------------------------------------------------------------------------
// CKArgs<NDimSpatial, DataType> — handles 2D/3D via runtime check
// ---------------------------------------------------------------------------

template <int NDimSpatial, typename DataType>
struct CKArgs
{
    using OutputElementOpType = OutElementOp;
    using OutputDataType      = DataType;

    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G;
        K  = K1 / G;

        if(problem.Is3d())
        {
            Di = ProblemInterpreter::GetInputDepthDi(problem);
            Do = ProblemInterpreter::GetOutputDepthDo(problem);
            Z  = ProblemInterpreter::GetFilterDepthZ(problem);
            Hi = ProblemInterpreter::GetInputHeightHi(problem);
            Wi = ProblemInterpreter::GetInputWidthWi(problem);
            Ho = ProblemInterpreter::GetOutputHeightHo(problem);
            Wo = ProblemInterpreter::GetOutputWidthWo(problem);
            Y  = ProblemInterpreter::GetFilterHeightY(problem);
            X  = ProblemInterpreter::GetFilterWidthX(problem);

            in_lens  = {G, N, C, Di, Hi, Wi};
            out_lens = {G, N, K, Do, Ho, Wo};
            wei_lens = {G, K, C, Z, Y, X};

            in_strides  = {C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
            out_strides = {K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};

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
        else
        {

            Hi = ProblemInterpreter::GetInputHeightHi(problem);
            Wi = ProblemInterpreter::GetInputWidthWi(problem);
            Ho = ProblemInterpreter::GetOutputHeightHo(problem);
            Wo = ProblemInterpreter::GetOutputWidthWo(problem);
            Y  = ProblemInterpreter::GetFilterHeightY(problem);
            X  = ProblemInterpreter::GetFilterWidthX(problem);

            in_lens  = {G, N, C, Hi, Wi};
            out_lens = {G, N, K, Ho, Wo};
            wei_lens = {G, K, C, Y, X};

            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};

            filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                               ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
            filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                               ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
            lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                               ProblemInterpreter::GetInputLeftPadW(problem)};
            rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                               ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
        }
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    ConstData_t bias,
                    Data_t out,
                    float alpha,
                    float beta,
                    OutElementOp clampOp) const
    {
        (void)alpha;
        (void)beta;
        (void)bias;
        constexpr bool is3DConv = (NDimSpatial == 3);

        if constexpr(is3DConv)
        {
            return conv_ptr->MakeArgumentPointer(in,
                                                 w,
                                                 {},
                                                 out,
                                                 in_lens,
                                                 in_strides,
                                                 wei_lens,
                                                 wei_strides,
                                                 {},
                                                 {},
                                                 out_lens,
                                                 out_strides,
                                                 filter_stride,
                                                 filter_dilation,
                                                 lPadding,
                                                 rPadding,
                                                 in_element_op,
                                                 wei_element_op,
                                                 clampOp);
        }
        else
        {
            std::array<ck::index_t, 5> adjusted_in_lens{};
            std::array<ck::index_t, 5> adjusted_out_lens{};
            std::array<ck::index_t, 5> adjusted_wei_lens{};

            std::copy(in_lens.begin(), in_lens.begin() + 5, adjusted_in_lens.begin());
            std::copy(out_lens.begin(), out_lens.begin() + 5, adjusted_out_lens.begin());
            std::copy(wei_lens.begin(), wei_lens.begin() + 5, adjusted_wei_lens.begin());

            std::array<ck::index_t, 5> adjusted_in_strides{};
            std::array<ck::index_t, 5> adjusted_out_strides{};
            std::array<ck::index_t, 5> adjusted_wei_strides{};
            std::copy(in_strides.begin(), in_strides.begin() + 5, adjusted_in_strides.begin());
            std::copy(out_strides.begin(), out_strides.begin() + 5, adjusted_out_strides.begin());
            std::copy(wei_strides.begin(), wei_strides.begin() + 5, adjusted_wei_strides.begin());

            std::array<ck::index_t, 2> adjusted_filter_stride{};
            std::array<ck::index_t, 2> adjusted_filter_dilation{};
            std::array<ck::index_t, 2> adjusted_lPadding{};
            std::array<ck::index_t, 2> adjusted_rPadding{};

            std::copy(
                filter_stride.begin(), filter_stride.begin() + 2, adjusted_filter_stride.begin());
            std::copy(filter_dilation.begin(),
                      filter_dilation.begin() + 2,
                      adjusted_filter_dilation.begin());
            std::copy(lPadding.begin(), lPadding.begin() + 2, adjusted_lPadding.begin());
            std::copy(rPadding.begin(), rPadding.begin() + 2, adjusted_rPadding.begin());

            return conv_ptr->MakeArgumentPointer(in,
                                                 w,
                                                 {},
                                                 out,
                                                 adjusted_in_lens,
                                                 adjusted_in_strides,
                                                 adjusted_wei_lens,
                                                 adjusted_wei_strides,
                                                 {},
                                                 {},
                                                 adjusted_out_lens,
                                                 adjusted_out_strides,
                                                 adjusted_filter_stride,
                                                 adjusted_filter_dilation,
                                                 adjusted_lPadding,
                                                 adjusted_rPadding,
                                                 in_element_op,
                                                 wei_element_op,
                                                 clampOp);
        }
    }

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    const miopen::fusion::FusionInvokeParams& data_ctx) const
    {
        const auto& conv_param =
            dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*data_ctx.op_args.params[0]);
        assert(&conv_param);

        const auto& activ_param =
            dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(*data_ctx.op_args.params[1]);

        return MakeArgPtr(op_ptr,
                          data_ctx.in,
                          conv_param.weights,
                          nullptr,
                          data_ctx.out,
                          conv_param.alpha,
                          conv_param.beta,
                          miopen::solver::GetOutElementOp<DataType, OutElementOp>(activ_param));
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  1.0f,
                                  0.0f,
                                  OutElementOp{0, ck::NumericLimits<DataType>::Max()});
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
    int Di = 0;
    int Do = 0;
    int Z  = 0;
    std::array<ck::index_t, 6> in_lens;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lens;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lens;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> filter_stride;
    std::array<ck::index_t, 3> filter_dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

// ---------------------------------------------------------------------------
// Template helpers
// ---------------------------------------------------------------------------

template <int NDim, typename DataType>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    using Layouts = LayoutsSelector<NDim>;
    return miopen::solver::FillValidKernelsIDs<DeviceOpGFwdActPtrs<NDim,
                                                                   DataType,
                                                                   typename Layouts::InLayout,
                                                                   typename Layouts::WeiLayout,
                                                                   typename Layouts::OutLayout>,
                                               CKArgs<NDim, DataType>>(problem);
}

template <int NDim, typename DataType>
bool CheckCKApplicability(const ProblemDescription& problem)
{
    using Layouts = LayoutsSelector<NDim>;
    return miopen::solver::IsCKApplicable<DeviceOpGFwdActPtrs<NDim,
                                                              DataType,
                                                              typename Layouts::InLayout,
                                                              typename Layouts::WeiLayout,
                                                              typename Layouts::OutLayout>,
                                          CKArgs<NDim, DataType>>(problem);
}

template <int NDim, typename DataType>
bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    using Layouts = LayoutsSelector<NDim>;
    return miopen::solver::IsCKArgsSupported<DeviceOpGFwdActPtrs<NDim,
                                                                 DataType,
                                                                 typename Layouts::InLayout,
                                                                 typename Layouts::WeiLayout,
                                                                 typename Layouts::OutLayout>,
                                             CKArgs<NDim, DataType>>(problem, kernel_id);
}

// ---------------------------------------------------------------------------
// Dispatch helpers that select 2D or 3D based on problem->Is3d()
// ---------------------------------------------------------------------------

template <typename DataType>
std::vector<std::string> FillValidKernelsForDim(const ProblemDescription& problem)
{
    if(problem.Is3d())
        return FillValidKernels<3, DataType>(problem);
    else
        return FillValidKernels<2, DataType>(problem);
}

template <typename DataType>
bool CheckCKApplicabilityForDim(const ProblemDescription& problem)
{
    if(problem.Is3d())
        return CheckCKApplicability<3, DataType>(problem);
    else
        return CheckCKApplicability<2, DataType>(problem);
}

template <typename DataType>
bool CheckIsArgSupportedForDim(const ProblemDescription& problem, const std::string& kernel_id)
{
    if(problem.Is3d())
        return CheckIsArgSupported<3, DataType>(problem, kernel_id);
    else
        return CheckIsArgSupported<2, DataType>(problem, kernel_id);
}

} // anonymous namespace

// ===========================================================================
// Fused grouped conv + activation (Clamp) extern "C" functions
// ===========================================================================

using miopen::solver::InitInvokerFactoryFwdNCHW;
using miopen::solver::InitInvokerFactoryNHWC;
using miopen::solver::MakeSolutionGroupConvImplicitGemmXdlops;

extern "C" ck_impl_status_t
ck_impl_fused_grp_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                           miopenDataType_t data_type,
                                           bool /*use_tf32*/,
                                           CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result     = std::make_unique<CKKernelListHandle>();
        result->kernels = DispatchByDataType(data_type, [&](auto type_val) {
            return FillValidKernelsForDim<decltype(type_val)>(*problem);
        });
        *out_handle     = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_grp_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                      miopenDataType_t data_type,
                                      bool /*use_tf32*/,
                                      bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_result = DispatchByDataType(data_type, [&](auto type_val) {
            return CheckCKApplicabilityForDim<decltype(type_val)>(*problem);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_grp_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
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
        *out_result = DispatchByDataType(data_type, [&](auto type_val) {
            return CheckIsArgSupportedForDim<decltype(type_val)>(*problem, kid);
        });
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_grp_activ_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                           miopenDataType_t /*data_type*/,
                                           bool /*use_tf32*/,
                                           size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        *out_size = miopen::solver::GetWorkspaceSizeLayoutTransformConv(*problem);
    });
}

extern "C" ck_impl_status_t
ck_impl_fused_grp_activ_get_solution(const miopen::ExecutionContext* ctx,
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

        auto make_solution = [&](auto ndim_constant) {
            constexpr int NDim = decltype(ndim_constant)::value;
            using Layouts      = LayoutsSelector<NDim>;
            return MakeSolutionGroupConvImplicitGemmXdlops(
                *problem,
                [&](auto data_type_val, [[maybe_unused]] auto compute_type_val) {
                    (void)data_type_val;
                    using T = decltype(data_type_val);
                    return InitInvokerFactoryFwdNCHW<
                        NDim,
                        false,
                        DeviceOpGFwdActPtrs<NDim,
                                            T,
                                            typename Layouts::InLayout,
                                            typename Layouts::WeiLayout,
                                            typename Layouts::OutLayout>,
                        CKArgs<NDim, T>,
                        miopen::fusion::FusionInvokeParams>(*ctx, *problem, kid);
                },
                [&](auto data_type_val, [[maybe_unused]] auto compute_type_val) {
                    (void)data_type_val;
                    using T = decltype(data_type_val);
                    return InitInvokerFactoryNHWC<false,
                                                  DeviceOpGFwdActPtrs<NDim,
                                                                      T,
                                                                      typename Layouts::InLayout,
                                                                      typename Layouts::WeiLayout,
                                                                      typename Layouts::OutLayout>,
                                                  CKArgs<NDim, T>,
                                                  miopen::fusion::FusionInvokeParams>(
                        *ctx, *problem, kid);
                });
        };

        miopen::solver::ConvSolution solution;
        if(problem->Is3d())
            solution = make_solution(std::integral_constant<int, 3>{});
        else
            solution = make_solution(std::integral_constant<int, 2>{});

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
