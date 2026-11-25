// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvRdnaFusedEpilogue.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

namespace
{

constexpr std::string_view B_UID_VALUE = "b.uid";
constexpr std::string_view ACTIVATION_VALUE = "activation";
constexpr std::array<std::string_view, 3> ACTIVATION_PARAMETER_VALUES{
    "activation.p0", "activation.p1", "activation.p2"};

/// The pointwise modes the kernels apply, and the kernels' number for each.
struct ServedActivation
{
    data_objects::PointwiseMode pointwise;
    ActivationMode kernel;
};
constexpr std::array<ServedActivation, 9> SERVED_ACTIVATIONS{{
    {data_objects::PointwiseMode::RELU_FWD, ActivationMode::RELU},
    {data_objects::PointwiseMode::SIGMOID_FWD, ActivationMode::SIGMOID},
    {data_objects::PointwiseMode::TANH_FWD, ActivationMode::TANH},
    {data_objects::PointwiseMode::ELU_FWD, ActivationMode::ELU},
    {data_objects::PointwiseMode::GELU_FWD, ActivationMode::GELU},
    {data_objects::PointwiseMode::GELU_APPROX_TANH_FWD, ActivationMode::GELU_APPROX_TANH},
    {data_objects::PointwiseMode::SWISH_FWD, ActivationMode::SWISH},
    {data_objects::PointwiseMode::SOFTPLUS_FWD, ActivationMode::SOFTPLUS},
    {data_objects::PointwiseMode::IDENTITY, ActivationMode::IDENTITY},
}};

std::optional<ActivationMode> kernelMode(data_objects::PointwiseMode mode)
{
    for(const auto& served : SERVED_ACTIVATIONS)
    {
        if(served.pointwise == mode)
        {
            return served.kernel;
        }
    }
    return std::nullopt;
}

std::optional<data_objects::PointwiseMode> pointwiseMode(ActivationMode mode)
{
    for(const auto& served : SERVED_ACTIVATIONS)
    {
        if(served.kernel == mode)
        {
            return served.pointwise;
        }
    }
    return std::nullopt;
}

/// Whether @p activation is one the kernels compute as the reference does: finite parameters,
/// and a nonzero softplus beta (the reference divides by it).
bool isServable(const Activation& activation)
{
    if(!std::isfinite(activation.p0) || !std::isfinite(activation.p1)
       || !std::isfinite(activation.p2))
    {
        return false;
    }
    return activation.mode != ActivationMode::SOFTPLUS || activation.p0 != 0.0F;
}

/// The activation of a unary pointwise node, with the parameters an unset attribute takes in
/// hipDNN's reference: lower clip 0, upper clip the largest float, slope 0, and alpha / beta 1.
std::optional<Activation> activationOf(const data_objects::PointwiseAttributes& node)
{
    const auto mode = kernelMode(node.operation());
    if(!mode)
    {
        return std::nullopt;
    }
    Activation activation;
    activation.mode = *mode;
    switch(*mode)
    {
    case ActivationMode::RELU:
        activation.p0 = node.relu_lower_clip().value_or(0.0F);
        activation.p1 = node.relu_upper_clip().value_or(std::numeric_limits<float>::max());
        activation.p2 = node.relu_lower_clip_slope().value_or(0.0F);
        break;
    case ActivationMode::ELU:
        activation.p0 = node.elu_alpha().value_or(1.0F);
        break;
    case ActivationMode::SWISH:
        activation.p0 = node.swish_beta().value_or(1.0F);
        break;
    case ActivationMode::SOFTPLUS:
        activation.p0 = node.softplus_beta().value_or(1.0F);
        break;
    default:
        break;
    }
    if(!isServable(activation))
    {
        return std::nullopt;
    }
    return activation;
}

bool sameDims(const data_objects::TensorAttributes& a, const data_objects::TensorAttributes& b)
{
    const auto* aDims = a.dims();
    const auto* bDims = b.dims();
    if(aDims == nullptr || bDims == nullptr || aDims->size() != bDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t i = 0; i < aDims->size(); ++i)
    {
        if(aDims->Get(i) != bDims->Get(i))
        {
            return false;
        }
    }
    return true;
}

/// An intermediate the kernel keeps in registers: virtual, the shape of the final output, and
/// either fp32 or the data dtype (the kernel keeps it in fp32 either way).
bool isFusedIntermediate(const data_objects::TensorAttributes& tensor,
                         const data_objects::TensorAttributes& output)
{
    return tensor.virtual_() && sameDims(tensor, output)
           && (tensor.data_type() == data_objects::DataType::FLOAT
               || tensor.data_type() == output.data_type());
}

/// A device tensor of @p problem's dtype holding one value per output channel: dims
/// (1, K, 1...) of the convolution's rank, the channels contiguous.
bool isChannelBias(const data_objects::TensorAttributes& bias,
                   const ConvProblem& problem,
                   flatbuffers::uoffset_t rank)
{
    if(bias.virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&bias)
       || bias.data_type() != problem.dtype)
    {
        return false;
    }
    const auto* dims = bias.dims();
    const auto* strides = bias.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != rank || strides->size() != rank)
    {
        return false;
    }
    for(flatbuffers::uoffset_t i = 0; i < rank; ++i)
    {
        if(dims->Get(i) != (i == 1 ? int64_t{problem.k} : 1))
        {
            return false;
        }
    }
    return strides->Get(1) == 1;
}

} // namespace

std::optional<ConvEpilogueMatch> matchConvBiasActivation(const MatchContext& context)
{
    // conv -> ADD(bias) -> activation, or conv -> activation.
    const uint32_t nodes = context.graph.nodeCount();
    if(nodes != 2 && nodes != 3)
    {
        return std::nullopt;
    }
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    std::vector<const data_objects::PointwiseAttributes*> pointwise;
    for(uint32_t i = 0; i < nodes; ++i)
    {
        const auto& node = context.graph.getNodeWrapper(i);
        // The kernels accumulate the convolution and apply the activation in fp32, so a graph
        // that asks for another compute type for either is declined rather than computed at a
        // different precision.
        if(node.attributesType() == data_objects::NodeAttributes::ConvolutionFwdAttributes
           && node.computeDataType() == data_objects::DataType::FLOAT)
        {
            conv = &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
        }
        else if(node.attributesType() == data_objects::NodeAttributes::PointwiseAttributes)
        {
            const auto& attributes = node.attributesAs<data_objects::PointwiseAttributes>();
            if(attributes.operation() != data_objects::PointwiseMode::ADD
               && node.computeDataType() != data_objects::DataType::FLOAT)
            {
                return std::nullopt;
            }
            pointwise.push_back(&attributes);
        }
    }
    if(conv == nullptr || pointwise.size() != nodes - 1)
    {
        return std::nullopt;
    }

    // With a bias, one pointwise node is the ADD of the bias and the other the activation that
    // reads its sum; without one, the activation reads the convolution output.
    const int64_t convOutUid = conv->y_tensor_uid();
    const data_objects::PointwiseAttributes* add = nullptr;
    const auto* act = pointwise[0];
    if(nodes == 3)
    {
        add = pointwise[0];
        act = pointwise[1];
        if(act->operation() == data_objects::PointwiseMode::ADD)
        {
            std::swap(add, act);
        }
        if(add->operation() != data_objects::PointwiseMode::ADD
           || !add->in_1_tensor_uid().has_value() || add->in_2_tensor_uid().has_value())
        {
            return std::nullopt;
        }
    }
    const int64_t actInUid = add != nullptr ? add->out_0_tensor_uid() : convOutUid;
    if(act->in_1_tensor_uid().has_value() || act->in_2_tensor_uid().has_value()
       || act->in_0_tensor_uid() != actInUid)
    {
        return std::nullopt;
    }
    const auto activation = activationOf(*act);
    if(!activation)
    {
        return std::nullopt;
    }

    ConvEpilogueMatch matched;
    const data_objects::TensorAttributes* sum = nullptr;
    if(add != nullptr)
    {
        // One ADD operand is the convolution output, the other the bias, in either order.
        const int64_t in0 = add->in_0_tensor_uid();
        const int64_t in1 = *add->in_1_tensor_uid();
        if((in0 == convOutUid) == (in1 == convOutUid))
        {
            return std::nullopt;
        }
        matched.bUid = in0 == convOutUid ? in1 : in0;
        matched.b = findTensor(context, *matched.bUid);
        sum = findTensor(context, actInUid);
        if(matched.b == nullptr || sum == nullptr)
        {
            return std::nullopt;
        }
    }
    matched.conv = conv;
    matched.activation = *activation;
    matched.yUid = act->out_0_tensor_uid();
    matched.x = findTensor(context, conv->x_tensor_uid());
    matched.w = findTensor(context, conv->w_tensor_uid());
    matched.y = findTensor(context, matched.yUid);
    const auto* convOut = findTensor(context, convOutUid);
    if(matched.x == nullptr || matched.w == nullptr || matched.y == nullptr || convOut == nullptr)
    {
        return std::nullopt;
    }

    // Neither the convolution output nor the biased sum is stored, and nothing else reads them.
    if(!isFusedIntermediate(*convOut, *matched.y)
       || (sum != nullptr && !isFusedIntermediate(*sum, *matched.y)))
    {
        return std::nullopt;
    }

    // The convolution as the kernel runs it, with the activation output as its stored output.
    const ConvAttributes attributes{conv->conv_mode(),
                                    conv->stride(),
                                    conv->dilation(),
                                    conv->pre_padding(),
                                    conv->post_padding()};
    const auto problem = convProblem(*matched.x, *matched.w, *matched.y, attributes);
    if(!problem
       || (matched.b != nullptr && !isChannelBias(*matched.b, *problem, matched.x->dims()->size())))
    {
        return std::nullopt;
    }
    matched.problem = *problem;
    return matched;
}

void addEpilogueLaunchValues(MetadataValues& values,
                             std::optional<int64_t> bUid,
                             const Activation& activation)
{
    if(bUid)
    {
        values[std::string(B_UID_VALUE)] = *bUid;
    }
    const auto mode = pointwiseMode(activation.mode);
    values[std::string(ACTIVATION_VALUE)]
        = std::string(mode ? data_objects::EnumNamePointwiseMode(*mode) : "");
    const std::array<float, 3> parameters{activation.p0, activation.p1, activation.p2};
    for(size_t i = 0; i < parameters.size(); ++i)
    {
        values[std::string(ACTIVATION_PARAMETER_VALUES[i])] = static_cast<double>(parameters[i]);
    }
}

std::vector<std::string_view> epilogueLaunchValueNames()
{
    std::vector<std::string_view> names{B_UID_VALUE, ACTIVATION_VALUE};
    names.insert(
        names.end(), ACTIVATION_PARAMETER_VALUES.begin(), ACTIVATION_PARAMETER_VALUES.end());
    return names;
}

EpilogueLaunchInputs readEpilogueLaunchValues(const SavedLaunchInputs& inputs)
{
    EpilogueLaunchInputs epilogue;
    if(inputs.values.count(std::string(B_UID_VALUE)) != 0)
    {
        epilogue.bUid = requireLaunchInt(inputs, B_UID_VALUE);
    }

    const auto modeName = requireLaunchString(inputs, ACTIVATION_VALUE);
    std::optional<ActivationMode> mode;
    for(const auto& served : SERVED_ACTIVATIONS)
    {
        if(modeName == data_objects::EnumNamePointwiseMode(served.pointwise))
        {
            mode = served.kernel;
        }
    }
    std::array<double, 3> parameters{};
    for(size_t i = 0; i < parameters.size(); ++i)
    {
        parameters[i] = requireLaunchDouble(inputs, ACTIVATION_PARAMETER_VALUES[i]);
    }
    const auto fitsFloat = [](double value) {
        return std::isfinite(value)
               && std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max());
    };
    if(mode)
    {
        epilogue.activation.mode = *mode;
        epilogue.activation.p0 = static_cast<float>(parameters[0]);
        epilogue.activation.p1 = static_cast<float>(parameters[1]);
        epilogue.activation.p2 = static_cast<float>(parameters[2]);
    }
    if(!mode || !fitsFloat(parameters[0]) || !fitsFloat(parameters[1]) || !fitsFloat(parameters[2])
       || !isServable(epilogue.activation))
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "launch values of dispatch '" + inputs.dispatchSymbol
                                              + "' give activation '" + modeName
                                              + "' with parameters the fused engines do not serve");
    }
    return epilogue;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
