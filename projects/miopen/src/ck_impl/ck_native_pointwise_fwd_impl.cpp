// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_native_pointwise_fwd_impl.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_nchw_pointwise_instance.hpp"

namespace miopen {
namespace solver {
namespace conv {
namespace {

struct NativeFactory
{
    static auto GetInstances()
    {
        std::vector<std::unique_ptr<ck::tensor_operation::device::instance::DeviceNCHWPointwiseFwd>>
            instances;
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP16)
        instances.reserve(1);
#endif
        ck::tensor_operation::device::instance::
            add_device_grouped_conv2d_fwd_nchw_pointwise_f16_instances(instances);
        return instances;
    }
};

bool IsEligible(const miopen::conv::ProblemDescription& problem)
{
    return problem.IsDirectionForward() && problem.Is2d() && problem.IsLayoutDefault() &&
           !problem.HasMixedDataTypes() && !problem.IsTensorsCasted() &&
           !problem.HasNonPackedTensors() && problem.GetGroupCount() == 1 &&
           problem.GetInDataType() == miopenHalf && problem.GetAlphaBetaCase() == DEFAULT;
}

struct NativeArgs : CKArgsSplitK<NativeArgs>
{
    NativeArgs(const miopen::conv::ProblemDescription& problem) : CKArgsSplitK<NativeArgs>(problem)
    {
        const auto& x = problem.GetIn().GetStrides();
        const auto& y = problem.GetOut().GetStrides();
        const auto& w = problem.GetWeights().GetStrides();
        std::copy(x.begin(), x.end(), in_strides.begin() + 1);
        std::copy(y.begin(), y.end(), out_strides.begin() + 1);
        std::copy(w.begin(), w.end(), wei_strides.begin() + 1);
        in_strides[0]  = C * in_strides[2];
        out_strides[0] = K * out_strides[2];
        wei_strides[0] = K * wei_strides[1];
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& op,
                    ConstData_t x,
                    ConstData_t w,
                    Data_t y,
                    float alpha,
                    float beta,
                    int split = 1) const
    {
        MIOPEN_THROW_IF(alpha != 1.0f || beta != 0.0f || split != 1,
                        "Native pointwise forward requires alpha=1, beta=0, split=1");
        return op->MakeArgumentPointer(x,
                                       w,
                                       {},
                                       y,
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
    auto MakeArgPtr(const ConvPtr& op,
                    const ConvDataTensors& tensors,
                    float alpha,
                    float beta,
                    int split = 1) const
    {
        return MakeArgPtr(op, tensors.in, tensors.w, tensors.out, alpha, beta, split);
    }
};

} // namespace

std::vector<std::string> NativePointwiseFwdKernels(const miopen::conv::ProblemDescription& problem)
{
    if(!IsEligible(problem))
        return {};
    const NativeArgs args{problem};
    std::vector<std::string> kernels;
    for(const auto& op : NativeFactory::GetInstances())
        if(args.IsSupportedBy(op))
            kernels.push_back(op->GetTypeString());
    return kernels;
}

bool IsNativePointwiseFwdKernel(const std::string& kernel_id)
{
    return kernel_id.find("DeviceGroupedConvFwdNCHWPointwise<") == 0;
}

bool IsNativePointwiseFwdSupported(const miopen::conv::ProblemDescription& problem,
                                   const std::string& kernel_id)
{
    const auto kernels = NativePointwiseFwdKernels(problem);
    return std::find(kernels.begin(), kernels.end(), kernel_id) != kernels.end();
}

ConvSolution NativePointwiseFwdSolution(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const std::string& kernel_id)
{
    if(!IsNativePointwiseFwdSupported(problem, kernel_id))
        return ConvSolution{miopenStatusInvalidValue};
    // Direct invocation: NativeArgs passes original NCHW tensors, no conversion/workspace.
    return InitInvokerFactoryNHWC<false, NativeFactory, NativeArgs, miopen::conv::DataInvokeParams>(
        ctx, problem, kernel_id);
}

} // namespace conv
} // namespace solver
} // namespace miopen
