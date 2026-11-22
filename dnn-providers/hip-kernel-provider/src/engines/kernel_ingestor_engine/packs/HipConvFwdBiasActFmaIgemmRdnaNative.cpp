// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFmaIgemm.hpp"
#include "engines/kernel_ingestor_engine/packs/HipConvFwdBiasActFmaIgemmRdnaLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file HipConvFwdBiasActFmaIgemmRdnaNative.cpp
 * @brief The native half of `hipkernel:hip_conv_fwd_bias_act_fma_igemm_rdna`
 *        (descriptors/hip/conv_bias_act_rdna/): matching, scoring, dispatch, and the function
 *        that registers them.
 *
 * The engine serves, on the gfx11 and gfx12 RDNA targets, a conv-forward node whose virtual
 * output feeds a pointwise ADD with a per-output-channel bias, whose virtual output feeds a
 * forward activation; or, without the bias, a conv-forward node whose virtual output feeds the
 * activation directly. It runs the FMA implicit-GEMM kernel of
 * `hip_conv_fwd_fma_igemm_rdna` (ConvRdnaFmaIgemm.hpp) built with its bias and activation
 * epilogues, so neither intermediate is stored: the bias is added and the activation applied
 * in fp32, and the result rounded once. The convolution itself is served under the same rules
 * as the unfused engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using conv_rdna::ConvProblem;
namespace fma_igemm = conv_rdna::fma_igemm;
using fma_igemm::Activation;
using fma_igemm::ActivationMode;

// The contract with hip_conv_fwd_bias_act_fma_igemm_rdna.*.json.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL
    = "hipkernel.hip_conv_fwd_bias_act_fma_igemm_rdna.dispatch";

// The names of the saved launch values (HipConvFwdBiasActFmaIgemmRdnaLaunchValues.hpp).
constexpr std::string_view DISPATCH_SYMBOL_V1
    = HIP_CONV_FWD_BIAS_ACT_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1;
constexpr std::string_view X_UID_VALUE = "x.uid";
constexpr std::string_view W_UID_VALUE = "w.uid";
constexpr std::string_view B_UID_VALUE = "b.uid";
constexpr std::string_view Y_UID_VALUE = "y.uid";
constexpr std::string_view ACTIVATION_VALUE = "activation";
constexpr std::array<std::string_view, 3> ACTIVATION_PARAMETER_VALUES{
    "activation.p0", "activation.p1", "activation.p2"};

constexpr std::string_view TOKEN_PREFIX = "conv_fwd_bias_act";
constexpr std::string_view X_TOKEN = "conv_fwd_bias_act.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd_bias_act.w.uid";
constexpr std::string_view B_TOKEN = "conv_fwd_bias_act.b.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd_bias_act.y.uid";

using BiasActBinding = HipConvFwdBiasActFmaIgemmRdnaBinding;
using LaunchInputs = HipConvFwdBiasActFmaIgemmRdnaLaunchInputs;

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// The pointwise modes the kernel applies, and the kernel's number for each.
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

/// Whether @p activation is one the kernel computes as the reference does: finite parameters,
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

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The conv node, the tensors the kernel reads and writes, the problem and the activation.
struct MatchedConvBiasAct
{
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::TensorAttributes* x = nullptr;
    const data_objects::TensorAttributes* w = nullptr;
    /// The bias; null for a graph without one.
    const data_objects::TensorAttributes* b = nullptr;
    /// The output of the activation: what the kernel stores.
    const data_objects::TensorAttributes* y = nullptr;
    std::optional<int64_t> bUid;
    int64_t yUid = 0;
    ConvProblem problem;
    Activation activation;
};

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

std::optional<MatchedConvBiasAct> matchConvBiasAct(const MatchContext& context)
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
        // The kernel accumulates the convolution and applies the activation in fp32, so a
        // graph that asks for another compute type for either is declined rather than computed
        // at a different precision.
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

    MatchedConvBiasAct matched;
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
        matched.b = conv_rdna::findTensor(context, *matched.bUid);
        sum = conv_rdna::findTensor(context, actInUid);
        if(matched.b == nullptr || sum == nullptr)
        {
            return std::nullopt;
        }
    }
    matched.conv = conv;
    matched.activation = *activation;
    matched.yUid = act->out_0_tensor_uid();
    matched.x = conv_rdna::findTensor(context, conv->x_tensor_uid());
    matched.w = conv_rdna::findTensor(context, conv->w_tensor_uid());
    matched.y = conv_rdna::findTensor(context, matched.yUid);
    const auto* convOut = conv_rdna::findTensor(context, convOutUid);
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
    const conv_rdna::ConvAttributes attributes{conv->conv_mode(),
                                               conv->stride(),
                                               conv->dilation(),
                                               conv->pre_padding(),
                                               conv->post_padding()};
    const auto problem = conv_rdna::convProblem(*matched.x, *matched.w, *matched.y, attributes);
    if(!problem
       || (matched.b != nullptr && !isChannelBias(*matched.b, *problem, matched.x->dims()->size())))
    {
        return std::nullopt;
    }
    matched.problem = *problem;
    return matched;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchConvBiasAct(context);
    if(!matched || !fma_igemm::serves(matched->problem))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    conv_rdna::bindConvFeatures(bound,
                                TOKEN_PREFIX,
                                matched->problem,
                                matched->conv->x_tensor_uid(),
                                matched->conv->w_tensor_uid(),
                                matched->yUid,
                                *matched->x,
                                *matched->w,
                                *matched->y);
    if(matched->bUid)
    {
        bound[std::string(B_TOKEN)] = *matched->bUid;
    }
    return bound;
}

/// The variant of the graph's problem (fma_igemm::isVariantOf); every kernel of this engine
/// carries the bias and activation epilogues.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto matched = matchConvBiasAct(context);
    return matched && fma_igemm::isVariantOf(kernel, matched->problem);
}

/// The kernel matcher leaves a single candidate to rank.
double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 1.0;
}

BiasActBinding biasActBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_act_fma_igemm_rdna dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold a tensor uid");
        }
        return *value;
    };
    BiasActBinding binding;
    binding.x = read(X_TOKEN);
    binding.w = read(W_TOKEN);
    // No bias token for a graph without a bias.
    binding.b = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, B_TOKEN);
    binding.y = read(Y_TOKEN);
    return binding;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class PreparedHipConvFwdBiasActFmaIgemmRdna : public IngestorPreparedDispatch
{
public:
    PreparedHipConvFwdBiasActFmaIgemmRdna(IngestorKernelCode code,
                                          BiasActBinding binding,
                                          ConvProblem problem,
                                          Activation activation)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _activation(activation)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const BiasActBinding& binding() const
    {
        return _binding;
    }

    const ConvProblem& problem() const
    {
        return _problem;
    }

    const Activation& activation() const
    {
        return _activation;
    }

private:
    IngestorKernelCode _code;
    BiasActBinding _binding;
    ConvProblem _problem;
    Activation _activation;
};

/// Applies the launch geometry to @p code and builds the prepared dispatch. A plan built from
/// a graph and a plan restored from a saved payload both come through here.
std::unique_ptr<PreparedDispatch> makePrepared(IngestorKernelCode code,
                                               const BiasActBinding& binding,
                                               const ConvProblem& problem,
                                               const Activation& activation)
{
    // Both grid dimensions were bounded by fma_igemm::serves or by readLaunchInputs.
    code.setBlockSize(fma_igemm::BLOCK_SIZE, 1, 1);
    code.setGridSize(static_cast<unsigned int>(fma_igemm::gridXBlocks(problem)),
                     static_cast<unsigned int>(fma_igemm::gridY(problem)),
                     1);
    return std::make_unique<PreparedHipConvFwdBiasActFmaIgemmRdna>(
        std::move(code), binding, problem, activation);
}

/// @see hipConvFwdBiasActFmaIgemmRdnaLaunchValues
MetadataValues launchValues(const BiasActBinding& binding,
                            const ConvProblem& problem,
                            const Activation& activation)
{
    MetadataValues values{{std::string(X_UID_VALUE), binding.x},
                          {std::string(W_UID_VALUE), binding.w},
                          {std::string(Y_UID_VALUE), binding.y}};
    if(binding.b)
    {
        values[std::string(B_UID_VALUE)] = *binding.b;
    }
    fma_igemm::addProblemLaunchValues(values, problem);
    const auto mode = pointwiseMode(activation.mode);
    values[std::string(ACTIVATION_VALUE)]
        = std::string(mode ? data_objects::EnumNamePointwiseMode(*mode) : "");
    const std::array<float, 3> parameters{activation.p0, activation.p1, activation.p2};
    for(size_t i = 0; i < parameters.size(); ++i)
    {
        values[std::string(ACTIVATION_PARAMETER_VALUES[i])] = static_cast<double>(parameters[i]);
    }
    return values;
}

/// @see readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs
LaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH_SYMBOL_V1);
    std::vector<std::string_view> names{
        X_UID_VALUE, W_UID_VALUE, B_UID_VALUE, Y_UID_VALUE, ACTIVATION_VALUE};
    names.insert(
        names.end(), ACTIVATION_PARAMETER_VALUES.begin(), ACTIVATION_PARAMETER_VALUES.end());
    const auto problemNames = fma_igemm::problemLaunchValueNames();
    names.insert(names.end(), problemNames.begin(), problemNames.end());
    requireOnlyLaunchValues(inputs, names);

    LaunchInputs launch;
    launch.binding.x = requireLaunchInt(inputs, X_UID_VALUE);
    launch.binding.w = requireLaunchInt(inputs, W_UID_VALUE);
    if(inputs.values.count(std::string(B_UID_VALUE)) != 0)
    {
        launch.binding.b = requireLaunchInt(inputs, B_UID_VALUE);
    }
    launch.binding.y = requireLaunchInt(inputs, Y_UID_VALUE);
    launch.problem = fma_igemm::readProblemLaunchValues(inputs);

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
        launch.activation.mode = *mode;
        launch.activation.p0 = static_cast<float>(parameters[0]);
        launch.activation.p1 = static_cast<float>(parameters[1]);
        launch.activation.p2 = static_cast<float>(parameters[2]);
    }
    if(!mode || !fitsFloat(parameters[0]) || !fitsFloat(parameters[1]) || !fitsFloat(parameters[2])
       || !isServable(launch.activation))
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "launch values of dispatch '" + inputs.dispatchSymbol
                                              + "' give activation '" + modeName
                                              + "' with parameters this engine does not serve");
    }
    return launch;
}

class HipConvFwdBiasActFmaIgemmRdnaDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    HipConvFwdBiasActFmaIgemmRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The weights are read in place and no intermediate is stored.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = biasActBinding(bound);
        const auto matched = matchConvBiasAct(context);
        if(!matched || !fma_igemm::serves(matched->problem))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hip_conv_fwd_bias_act_fma_igemm_rdna prepare() got a graph its matcher does not "
                "admit");
        }
        const auto& problem = matched->problem;

        // Same defines as the UKD's build, for a kernel compiled at run time.
        compilation::KernelCompileOptions options(matched->x, context.deviceProperties.gcnArchName);
        fma_igemm::addCompileOptions(options, problem, /*hasBias=*/true, /*hasActivation=*/true);

        auto code = buildIngestorKernelCode(
            _kernelCompiler,
            _kpackLoader,
            context,
            kernel,
            options,
            fma_igemm::kernelSignature(/*hasBias=*/true, /*hasActivation=*/true));
        return makePrepared(std::move(code), binding, problem, matched->activation);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored = restoreIngestorLaunch<LaunchInputs>(
            inputs,
            std::move(code),
            fma_igemm::kernelSignature(/*hasBias=*/true, /*hasActivation=*/true),
            &readLaunchInputs,
            deviceOrdinal);
        return makePrepared(std::move(restored.code),
                            restored.launch.binding,
                            restored.launch.problem,
                            restored.launch.activation);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasActFmaIgemmRdna&>(prepared);
        return SavedLaunchInputs{std::string(DISPATCH_SYMBOL_V1),
                                 launchValues(conv.binding(), conv.problem(), conv.activation())};
    }

    bool supportsSaving() const override
    {
        return true;
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& conv = dynamic_cast<const PreparedHipConvFwdBiasActFmaIgemmRdna&>(prepared);
        const auto& binding = conv.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const void* b = nullptr;
        if(binding.b)
        {
            b = hipdnn_plugin_sdk::findDeviceBuffer(*binding.b, deviceBuffers, numDeviceBuffers)
                    .ptr;
        }
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        fma_igemm::launch(conv.kernelForStream(handle.getStream()),
                          handle.getStream(),
                          x.ptr,
                          w.ptr,
                          y.ptr,
                          b,
                          binding.b.has_value(),
                          conv.problem(),
                          &conv.activation());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

const HipConvFwdBiasActFmaIgemmRdnaDispatchHandler& dispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache());
    static const HipConvFwdBiasActFmaIgemmRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                                s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

MetadataValues
    hipConvFwdBiasActFmaIgemmRdnaLaunchValues(const HipConvFwdBiasActFmaIgemmRdnaBinding& binding,
                                              const conv_rdna::ConvProblem& problem,
                                              const conv_rdna::fma_igemm::Activation& activation)
{
    return launchValues(binding, problem, activation);
}

HipConvFwdBiasActFmaIgemmRdnaLaunchInputs
    readHipConvFwdBiasActFmaIgemmRdnaLaunchInputs(const SavedLaunchInputs& inputs)
{
    return readLaunchInputs(inputs);
}

compilation::KpackModuleCache& hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetHipConvFwdBiasActFmaIgemmRdnaModuleCache()
{
    hipConvFwdBiasActFmaIgemmRdnaKpackModuleCache().clear();
}

void registerHipConvFwdBiasActFmaIgemmRdnaSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &score);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
    scope.add(std::string(DISPATCH_SYMBOL_V1), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
