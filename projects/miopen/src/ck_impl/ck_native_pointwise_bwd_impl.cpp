// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_native_pointwise_bwd_impl.hpp"
#include "ck_grouped_conv_impl_helpers.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_nchw_pointwise_instance.hpp"

namespace miopen {
namespace solver {
namespace conv {
namespace {

// Use the registered native family without pulling in the optional NGCHW XDL archive.
struct NativeFactory
{
    static auto GetInstances()
    {
        std::vector<
            std::unique_ptr<ck::tensor_operation::device::instance::DeviceNCHWPointwiseBwdData>>
            instances;
#if defined(CK_USE_GFX1250) && defined(CK_ENABLE_FP32)
        instances.reserve(2);
#endif
        ck::tensor_operation::device::instance::
            add_device_grouped_conv2d_bwd_data_nchw_pointwise_f32_instances(instances);
        return instances;
    }
};

bool IsEligible(const miopen::conv::ProblemDescription& problem)
{
    return problem.IsDirectionBackwardData() && problem.Is2d() && problem.IsLayoutDefault() &&
           !problem.HasMixedDataTypes() && !problem.IsTensorsCasted() &&
           !problem.HasNonPackedTensors() && problem.GetGroupCount() == 1 &&
           problem.GetInDataType() == miopenFloat && problem.GetAlphaBetaCase() == DEFAULT;
}

struct NativeArgs : CKArgsSplitK<NativeArgs>
{
    NativeArgs(const miopen::conv::ProblemDescription& problem) : CKArgsSplitK<NativeArgs>(problem)
    {
        // For BWD, GetIn describes dY and GetOut describes dX. Preserve the caller's
        // NCHW strides rather than CKArgsSplitK's converted NHWC strides.
        const auto& dx = problem.GetOut().GetStrides();
        const auto& dy = problem.GetIn().GetStrides();
        const auto& w  = problem.GetWeights().GetStrides();
        std::copy(dx.begin(), dx.end(), in_strides.begin() + 1);
        std::copy(dy.begin(), dy.end(), out_strides.begin() + 1);
        std::copy(w.begin(), w.end(), wei_strides.begin() + 1);
        in_strides[0]  = C * in_strides[2];
        out_strides[0] = K * out_strides[2];
        wei_strides[0] = K * wei_strides[1];
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& op,
                    Data_t dx,
                    ConstData_t w,
                    ConstData_t dy,
                    float alpha,
                    float beta,
                    int split = 1) const
    {
        MIOPEN_THROW_IF(alpha != 1.0f || beta != 0.0f || split != 1,
                        "Native pointwise backward data requires alpha=1, beta=0, split=1");
        return op->MakeArgumentPointer(dy,
                                       w,
                                       {},
                                       dx,
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
                                       split);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& op,
                    const ConvDataTensors& tensors,
                    float alpha,
                    float beta,
                    int split = 1) const
    {
        return MakeArgPtr(op, tensors.out, tensors.w, tensors.in, alpha, beta, split);
    }
};

std::string BaseKernelID(const std::string& id)
{
    const auto pos = id.find('+');
    if(pos == std::string::npos)
        return id;
    // Do not accept unsupported split values or ambiguous/malformed suffixes.
    return id.substr(pos) == "+1" ? id.substr(0, pos) : std::string{};
}

} // namespace

std::vector<std::string> NativePointwiseBwdKernels(const miopen::conv::ProblemDescription& problem)
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

bool IsNativePointwiseBwdKernel(const std::string& kernel_id)
{
    return kernel_id.find("DeviceGroupedConvBwdDataNCHWPointwise<") == 0;
}

bool IsNativePointwiseBwdSupported(const miopen::conv::ProblemDescription& problem,
                                   const std::string& kernel_id)
{
    if(!IsEligible(problem))
        return false;
    const auto id      = BaseKernelID(kernel_id);
    const auto kernels = NativePointwiseBwdKernels(problem);
    return std::find(kernels.begin(), kernels.end(), id) != kernels.end();
}

ConvSolution NativePointwiseBwdSolution(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const std::string& kernel_id)
{
    if(!IsNativePointwiseBwdSupported(problem, kernel_id))
        return ConvSolution{miopenStatusInvalidValue};
    // This helper enqueues the CK invoker directly. Despite its historical NHWC name,
    // it does not transpose anything: NativeArgs binds the original NCHW buffers.
    return InitInvokerFactoryNHWC<false, NativeFactory, NativeArgs, miopen::conv::DataInvokeParams>(
        ctx, problem, kernel_id);
}

} // namespace conv
} // namespace solver
} // namespace miopen
