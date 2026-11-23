// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/HiprtcKernelCompiler.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"

/**
 * @file BatchnormInferencePointwiseNative.cpp
 * @brief The hipkernel:BatchnormInferencePointwise engine's native half: a fused two-node
 *        graph match, a kernel-scoped matcher, dispatch, and the function that registers
 *        them.
 *
 * A test fixture (ALMIOPEN-3070): the first ingestor engine whose graph match spans two
 * nodes. It serves [BatchnormInference, Pointwise] where the Pointwise node consumes the
 * batchnorm output, the way the HIP MLOps engine fuses batchnorm inference with an
 * activation. The activation is classified as the MLOps engine classifies it
 * (core::utils::parseActivation) and bound as a token; the kernel-scoped matcher admits
 * only the kernel whose `activation` metadata names it. The clip bounds travel as the
 * `alpha` and `beta` tokens.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.batchnorm_inference_pointwise.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.batchnorm_inference_pointwise.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.batchnorm_inference_pointwise.dispatch";

constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view ACTIVATION_FIELD = "activation";

constexpr std::string_view X_TOKEN = "batchnorm_inference_pointwise.x.uid";
constexpr std::string_view MEAN_TOKEN = "batchnorm_inference_pointwise.mean.uid";
constexpr std::string_view INV_VARIANCE_TOKEN = "batchnorm_inference_pointwise.inv_variance.uid";
constexpr std::string_view SCALE_TOKEN = "batchnorm_inference_pointwise.scale.uid";
constexpr std::string_view BIAS_TOKEN = "batchnorm_inference_pointwise.bias.uid";
constexpr std::string_view INTERMEDIATE_TOKEN = "batchnorm_inference_pointwise.intermediate.uid";
constexpr std::string_view OUTPUT_TOKEN = "batchnorm_inference_pointwise.output.uid";
constexpr std::string_view ACTIVATION_TOKEN = "batchnorm_inference_pointwise.activation";
constexpr std::string_view ALPHA_TOKEN = "batchnorm_inference_pointwise.alpha";
constexpr std::string_view BETA_TOKEN = "batchnorm_inference_pointwise.beta";

// The activation names the kernels' `activation` metadata uses, and the value each passes to
// the kernel as HIP_PLUGIN_BNPW_ACTIVATION.
constexpr std::string_view IDENTITY = "IDENTITY";
constexpr std::string_view RELU = "RELU";
constexpr std::string_view CLIPPED_RELU = "CLIPPED_RELU";
constexpr std::string_view CLAMP = "CLAMP";
constexpr std::array<std::string_view, 4> ACTIVATIONS = {IDENTITY, RELU, CLIPPED_RELU, CLAMP};

constexpr uint32_t SUPPORTED_RANK = 4;
constexpr unsigned int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The extents of a rank-4 tensor whose strides are packed NCHW, or nullopt. Runs on an
/// unvalidated graph, so it checks every pointer and length before indexing.
std::optional<std::array<int64_t, SUPPORTED_RANK>>
    packedNchwDims(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SUPPORTED_RANK
       || strides->size() != SUPPORTED_RANK)
    {
        return std::nullopt;
    }

    std::array<int64_t, SUPPORTED_RANK> extents{};
    for(uint32_t i = 0; i < SUPPORTED_RANK; ++i)
    {
        extents[i] = dims->Get(i);
        if(extents[i] <= 0)
        {
            return std::nullopt;
        }
    }

    int64_t expected = 1;
    for(uint32_t i = SUPPORTED_RANK; i-- > 0;)
    {
        if(strides->Get(i) != expected)
        {
            return std::nullopt;
        }
        expected *= extents[i];
    }
    return extents;
}

/// A device tensor this engine reads or writes: real, not a pass-by-value scalar.
bool isDeviceOperand(const data_objects::TensorAttributes& tensor)
{
    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/// The activation the Pointwise node applies, with its clip bounds, or nullopt for one this
/// engine does not serve. Follows core::utils::parseActivation, which the HIP MLOps engine
/// uses, so the fused graphs it accepts are the ones that engine accepts.
struct Activation
{
    std::string_view name;
    double alpha = 0.0;
    double beta = 0.0;
};

std::optional<Activation> classifyActivation(const data_objects::PointwiseAttributes& pointwise)
{
    if(pointwise.operation() == data_objects::PointwiseMode::IDENTITY)
    {
        return Activation{IDENTITY};
    }
    if(pointwise.operation() != data_objects::PointwiseMode::RELU_FWD)
    {
        return std::nullopt;
    }

    const auto lower = pointwise.relu_lower_clip();
    const auto upper = pointwise.relu_upper_clip();
    if(lower.has_value() && upper.has_value())
    {
        return Activation{
            CLAMP, static_cast<double>(lower.value()), static_cast<double>(upper.value())};
    }
    if(upper.has_value())
    {
        return Activation{CLIPPED_RELU, 0.0, static_cast<double>(upper.value())};
    }
    // Leaky ReLU, and plain ReLU with a non-zero lower clip, are not served.
    if(pointwise.relu_lower_clip_slope().has_value()
       || (lower.has_value() && lower.value() != 0.0f))
    {
        return std::nullopt;
    }
    return Activation{RELU};
}

/// The two nodes this engine serves, or nullopt when the graph is not exactly
/// [BatchnormInference, Pointwise] in that order.
struct FusedNodes
{
    const data_objects::BatchnormInferenceAttributes* batchnorm;
    const data_objects::PointwiseAttributes* pointwise;
};

std::optional<FusedNodes> fusedNodes(const MatchContext& context)
{
    if(context.graph.nodeCount() != 2)
    {
        return std::nullopt;
    }

    const auto& first = context.graph.getNodeWrapper(0);
    const auto& second = context.graph.getNodeWrapper(1);
    if(first.attributesType() != data_objects::NodeAttributes::BatchnormInferenceAttributes
       || second.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }

    return FusedNodes{&first.attributesAs<data_objects::BatchnormInferenceAttributes>(),
                      &second.attributesAs<data_objects::PointwiseAttributes>()};
}

/**
 * @brief Graph-scoped applicability: is this a batchnorm inference whose output feeds one
 *        unary activation, over packed NCHW FLOAT tensors?
 *
 * Checks the edge by uid rather than by node order alone: the Pointwise node must read the
 * batchnorm output, which must be virtual (an intermediate the fused kernel never writes),
 * while every other operand is a real device tensor.
 */
std::optional<BoundTokens> batchnormInferencePointwiseGraphMatches(const MatchContext& context)
{
    const auto nodes = fusedNodes(context);
    if(!nodes.has_value())
    {
        return std::nullopt;
    }
    const auto& batchnorm = *nodes->batchnorm;
    const auto& pointwise = *nodes->pointwise;

    // Unary, and reading exactly the batchnorm output.
    if(pointwise.in_1_tensor_uid().has_value() || pointwise.in_2_tensor_uid().has_value()
       || pointwise.in_0_tensor_uid() != batchnorm.y_tensor_uid())
    {
        return std::nullopt;
    }

    const auto activation = classifyActivation(pointwise);
    if(!activation.has_value())
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, batchnorm.x_tensor_uid());
    const auto* mean = findTensor(context, batchnorm.mean_tensor_uid());
    const auto* invVariance = findTensor(context, batchnorm.inv_variance_tensor_uid());
    const auto* scale = findTensor(context, batchnorm.scale_tensor_uid());
    const auto* bias = findTensor(context, batchnorm.bias_tensor_uid());
    const auto* intermediate = findTensor(context, batchnorm.y_tensor_uid());
    const auto* output = findTensor(context, pointwise.out_0_tensor_uid());
    if(x == nullptr || mean == nullptr || invVariance == nullptr || scale == nullptr
       || bias == nullptr || intermediate == nullptr || output == nullptr)
    {
        return std::nullopt;
    }

    if(!intermediate->virtual_())
    {
        return std::nullopt;
    }
    for(const auto* tensor : {x, mean, invVariance, scale, bias, output})
    {
        if(!isDeviceOperand(*tensor) || tensor->data_type() != data_objects::DataType::FLOAT)
        {
            return std::nullopt;
        }
    }

    const auto xDims = packedNchwDims(*x);
    const auto intermediateDims = packedNchwDims(*intermediate);
    const auto outputDims = packedNchwDims(*output);
    if(!xDims.has_value() || intermediateDims != xDims || outputDims != xDims)
    {
        return std::nullopt;
    }

    // One value per channel: [1, C, 1, 1].
    const std::array<int64_t, SUPPORTED_RANK> channelDims = {1, (*xDims)[1], 1, 1};
    for(const auto* tensor : {mean, invVariance, scale, bias})
    {
        if(packedNchwDims(*tensor) != channelDims)
        {
            return std::nullopt;
        }
    }

    // The kernel indexes with int.
    const int64_t elementCount = (*xDims)[0] * (*xDims)[1] * (*xDims)[2] * (*xDims)[3];
    if(elementCount > std::numeric_limits<int>::max())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = batchnorm.x_tensor_uid();
    bound[std::string(MEAN_TOKEN)] = batchnorm.mean_tensor_uid();
    bound[std::string(INV_VARIANCE_TOKEN)] = batchnorm.inv_variance_tensor_uid();
    bound[std::string(SCALE_TOKEN)] = batchnorm.scale_tensor_uid();
    bound[std::string(BIAS_TOKEN)] = batchnorm.bias_tensor_uid();
    bound[std::string(INTERMEDIATE_TOKEN)] = batchnorm.y_tensor_uid();
    bound[std::string(OUTPUT_TOKEN)] = pointwise.out_0_tensor_uid();
    bound[std::string(ACTIVATION_TOKEN)] = std::string(activation->name);
    bound[std::string(ALPHA_TOKEN)] = activation->alpha;
    bound[std::string(BETA_TOKEN)] = activation->beta;
    return bound;
}

/// The string token @p token, or nullopt when absent or not a string.
std::optional<std::string> boundString(const BoundTokens& bound, std::string_view token)
{
    const auto entry = bound.find(std::string(token));
    if(entry == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

/**
 * @brief Kernel-scoped applicability: does this kernel apply the graph's activation, at
 *        the graph's dtype? The activation is the field that separates this engine's
 *        kernels, so without it every graph would reach the identity kernel.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto activation = boundString(bound, ACTIVATION_TOKEN);
    const auto xUid = tryGetBoundInt(bound, X_TOKEN);
    if(!activation.has_value() || !xUid.has_value())
    {
        return false;
    }
    const auto* x = findTensor(context, *xUid);
    if(x == nullptr)
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(ACTIVATION_FIELD)) == *activation
           && kernel.getStringMetadata(std::string(DTYPE_FIELD))
                  == data_objects::EnumNameDataType(x->data_type());
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The operand uids a match bound, read back once at prepare().
struct Binding
{
    int64_t x = 0;
    int64_t mean = 0;
    int64_t invVariance = 0;
    int64_t scale = 0;
    int64_t bias = 0;
    int64_t output = 0;
};

/// The double token @p token.
///
/// @throws HipdnnPluginException when the graph match did not bind it.
double boundDouble(const BoundTokens& bound, std::string_view token)
{
    const auto entry = bound.find(std::string(token));
    const auto* value = entry == bound.end() ? nullptr : std::get_if<double>(&entry->second);
    if(value == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "batchnorm inference pointwise dispatch is missing bound token '" + std::string(token)
                + "', or it does not hold a number");
    }
    return *value;
}

/// @throws HipdnnPluginException when the graph match did not bind a uid.
Binding readBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "batchnorm inference pointwise dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN),
            read(MEAN_TOKEN),
            read(INV_VARIANCE_TOKEN),
            read(SCALE_TOKEN),
            read(BIAS_TOKEN),
            read(OUTPUT_TOKEN)};
}

/// The HIP_PLUGIN_BNPW_ACTIVATION value of the kernel's `activation` metadata.
///
/// @throws HipdnnPluginException for a name the kernel source does not implement.
unsigned int activationOrdinal(const KernelDefinition& kernel)
{
    const auto& activation = kernel.getStringMetadata(std::string(ACTIVATION_FIELD));
    for(unsigned int ordinal = 0; ordinal < ACTIVATIONS.size(); ++ordinal)
    {
        if(ACTIVATIONS[ordinal] == activation)
        {
            return ordinal;
        }
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "kernel '" + toString(kernel.kernelId)
                                                       + "' declares unsupported activation '"
                                                       + activation + "'");
}

/// The compiled kernel and everything launch() needs, resolved once from the graph.
class PreparedBatchnormInferencePointwise : public IngestorPreparedDispatch
{
public:
    PreparedBatchnormInferencePointwise(IngestorKernelCode code,
                                        Binding binding,
                                        float alpha,
                                        float beta,
                                        int elementCount,
                                        int channels,
                                        int spatialSize)
        : _code(std::move(code))
        , _binding(binding)
        , _alpha(alpha)
        , _beta(beta)
        , _elementCount(elementCount)
        , _channels(channels)
        , _spatialSize(spatialSize)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    const Binding& binding() const
    {
        return _binding;
    }

    void launch(hipStream_t stream,
                const void* x,
                const void* mean,
                const void* invVariance,
                const void* scale,
                const void* bias,
                void* output) const
    {
        // Changing this argument list means changing the kernel source with it.
        _code.kernelForStream(stream).launch(stream,
                                             x,
                                             mean,
                                             invVariance,
                                             scale,
                                             bias,
                                             output,
                                             _alpha,
                                             _beta,
                                             _elementCount,
                                             _channels,
                                             _spatialSize);
    }

private:
    IngestorKernelCode _code;
    Binding _binding;
    float _alpha;
    float _beta;
    int _elementCount;
    int _channels;
    int _spatialSize;
};

/// The tensor @p uid of a graph the match accepted.
const data_objects::TensorAttributes& matchedTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched batchnorm inference pointwise graph has no tensor for uid "
                + std::to_string(uid));
    }
    return *tensor;
}

/**
 * @brief The native dispatch behind this engine's UDD: compiles the fused kernel from
 *        source at prepare() and launches it once per execution.
 *
 * The kernel is embedded source, so plans built through this handler cannot be saved,
 * which is the default.
 */
class BatchnormInferencePointwiseDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler and @p kpackLoader must outlive this handler; all three are
    /// process-lifetime.
    BatchnormInferencePointwiseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                               const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        // workspace_policy: none -- the fused kernel never materialises the intermediate.
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = readBinding(bound);
        const auto& x = matchedTensor(context, binding.x);
        const auto* dims = x.dims();
        const int64_t channels = dims->Get(1);
        const int64_t spatialSize = dims->Get(2) * dims->Get(3);
        const int64_t elementCount = dims->Get(0) * channels * spatialSize;

        compilation::KernelCompileOptions options(&x, context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_BNPW_TYPE", std::string("float"));
        options.add("HIP_PLUGIN_BNPW_ACTIVATION", activationOrdinal(kernel));

        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options, {});
        const auto gridSize
            = static_cast<unsigned int>((elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE);
        code.setBlockSize(BLOCK_SIZE, 1, 1);
        code.setGridSize(gridSize, 1, 1);

        return std::make_unique<PreparedBatchnormInferencePointwise>(
            std::move(code),
            binding,
            static_cast<float>(boundDouble(bound, ALPHA_TOKEN)),
            static_cast<float>(boundDouble(bound, BETA_TOKEN)),
            static_cast<int>(elementCount),
            static_cast<int>(channels),
            static_cast<int>(spatialSize));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& fused = dynamic_cast<const PreparedBatchnormInferencePointwise&>(prepared);
        const auto& binding = fused.binding();
        const auto buffer = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };

        fused.launch(handle.getStream(),
                     buffer(binding.x),
                     buffer(binding.mean),
                     buffer(binding.invVariance),
                     buffer(binding.scale),
                     buffer(binding.bias),
                     buffer(binding.output));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

/// This engine's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer, while a provider's Container is created and destroyed per handle, so it (and
/// the compiler and loader it holds) must outlive every Container. The loader is never
/// reached -- every kernel is embedded source -- but buildIngestorKernelCode takes one.
const BatchnormInferencePointwiseDispatchHandler& batchnormInferencePointwiseDispatchHandler()
{
    static const compilation::HiprtcKernelCompiler s_kernelCompiler;
    static compilation::KpackModuleCache s_kpackModuleCache;
    static const compilation::KpackKernelLoader s_kpackLoader(s_kpackModuleCache);
    static const BatchnormInferencePointwiseDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                              s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerBatchnormInferencePointwiseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &batchnormInferencePointwiseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &batchnormInferencePointwiseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
