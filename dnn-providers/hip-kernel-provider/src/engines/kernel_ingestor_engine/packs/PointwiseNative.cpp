// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file PointwiseNative.cpp
 * @brief The pointwise engine's native half: matching, scoring, dispatch, and the one
 *        function that registers them.
 *
 * Three packs (add, mul, sub) share one applicability matcher -- node shape, binary,
 * one element, uniform dtype -- evaluated once per graph; each pack adds only an
 * operation check, so RFC 0017 §6 keeps them disjoint without duplicating that work.
 * Symbol names are restated here rather than shared via a header, since a descriptor
 * can't reference a C++ constant; the loader pre-flights every symbol a descriptor
 * names, so a mismatched string is caught without becoming a compile error.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.pointwise.graph_match";
constexpr std::string_view ADD_MATCHER_SYMBOL = "hipkernel.pointwise.add_match";
constexpr std::string_view MUL_MATCHER_SYMBOL = "hipkernel.pointwise.mul_match";
constexpr std::string_view SUB_MATCHER_SYMBOL = "hipkernel.pointwise.sub_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.pointwise.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.pointwise.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.pointwise.dispatch";

constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view INPUT_A_TOKEN = "input_a";
constexpr std::string_view INPUT_B_TOKEN = "input_b";
constexpr std::string_view OUTPUT_TOKEN = "output";

/// Scratch reported by the larger-block kernel; keeps max-across-survivors non-zero.
constexpr size_t LARGE_BLOCK_WORKSPACE_BYTES = 1024;
constexpr int64_t LARGE_BLOCK_SIZE = 256;

constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched pointwise graph binds, in argument order.
struct PointwiseBinding
{
    int64_t inputA = 0;
    int64_t inputB = 0;
    int64_t output = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// Packedness alone is not enough: require one element and a supported rank and layout.
/// Use the shared stride order for singleton axes.
bool isSupportedOperand(const BoundTokens& bound, std::string_view root)
{
    const auto rank = tryGetBoundInt(bound, tensorField(root, "rank"));
    if(!rank || *rank < MIN_SUPPORTED_RANK || *rank > MAX_SUPPORTED_RANK)
    {
        return false;
    }
    for(int64_t i = 0; i < *rank; ++i)
    {
        if(tryGetBoundInt(bound, tensorElement(root, "dims", static_cast<size_t>(i))) != 1)
        {
            return false;
        }
    }
    const auto* order
        = tryGetBoundValue<std::vector<int64_t>>(bound, tensorField(root, "stride_order"));
    using hipdnn_data_sdk::utilities::TensorLayout;
    if(order == nullptr
       || (*rank == 4 && *order != TensorLayout::NCHW.strideOrder
           && *order != TensorLayout::NHWC.strideOrder)
       || (*rank == 5 && *order != TensorLayout::NCDHW.strideOrder
           && *order != TensorLayout::NDHWC.strideOrder))
    {
        return false;
    }
    const auto* isVirtual = tryGetBoundValue<bool>(bound, tensorField(root, "virtual"));
    const auto* runtime
        = tryGetBoundValue<bool>(bound, tensorField(root, "is_runtime_pass_by_value"));
    return isVirtual != nullptr && !*isVirtual && runtime != nullptr && !*runtime
           && tryGetBoundValue<double>(bound, tensorField(root, "value_f32")) == nullptr;
}

bool publishPointwiseNode(BoundTokens& bound,
                          const data_objects::PointwiseAttributes& attributes,
                          data_objects::DataType computeType)
{
    const auto* operation = data_objects::EnumNamePointwiseMode(attributes.operation());
    const auto* compute = data_objects::EnumNameDataType(computeType);
    if(operation == nullptr || *operation == '\0' || compute == nullptr || *compute == '\0')
    {
        return false;
    }
    bound.emplace("pointwise.operation", std::string(operation));
    bound.emplace("pointwise.compute_data_type", std::string(compute));
    const auto publishFloat = [&bound](const char* key, flatbuffers::Optional<float> value) {
        if(!value.has_value())
        {
            return true;
        }
        if(!std::isfinite(*value))
        {
            return false;
        }
        bound.emplace(key, static_cast<double>(*value));
        return true;
    };
    if(attributes.axis_tensor_uid().has_value())
    {
        bound.emplace("pointwise.axis_tensor_uid", *attributes.axis_tensor_uid());
    }
    return publishFloat("pointwise.relu_lower_clip", attributes.relu_lower_clip())
           && publishFloat("pointwise.relu_upper_clip", attributes.relu_upper_clip())
           && publishFloat("pointwise.relu_lower_clip_slope", attributes.relu_lower_clip_slope())
           && publishFloat("pointwise.swish_beta", attributes.swish_beta())
           && publishFloat("pointwise.elu_alpha", attributes.elu_alpha())
           && publishFloat("pointwise.softplus_beta", attributes.softplus_beta());
}

/// Return attributes for a single-node pointwise graph, or nullptr.
const data_objects::PointwiseAttributes* pointwiseNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes
       || node.attributes() == nullptr)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::PointwiseAttributes>();
}

/**
 * @brief Graph-scoped applicability shared by every pack: is this a single-node binary
 *        pointwise op over 1-element tensors this engine can launch? Says nothing about
 *        which operation -- every pack lists this matcher, so it runs once per (graph,
 *        device) and the memoized verdict is reused, with only the operation check left
 *        to each pack.
 */
std::optional<BoundTokens> pointwiseGraphMatches(const MatchContext& context)
{

    // Exactly one node: this engine's kernels each serve one complete graph.
    const auto* attributesPtr = pointwiseNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // Binary: a second operand is required, a third would be a different operation.
    if(!attributes.in_1_tensor_uid().has_value() || attributes.in_2_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto* inputA = findTensor(context, attributes.in_0_tensor_uid());
    const auto* inputB = findTensor(context, attributes.in_1_tensor_uid().value());
    const auto* output = findTensor(context, attributes.out_0_tensor_uid());
    if(inputA == nullptr || inputB == nullptr || output == nullptr)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    if(!publishGraph(bound, context.graph) || !publishTensor(bound, INPUT_A_TOKEN, inputA)
       || !publishTensor(bound, INPUT_B_TOKEN, inputB)
       || !publishTensor(bound, OUTPUT_TOKEN, output)
       || !publishPointwiseNode(
           bound, attributes, context.graph.getNodeWrapper(0).computeDataType()))
    {
        return std::nullopt;
    }
    if(!isSupportedOperand(bound, INPUT_A_TOKEN) || !isSupportedOperand(bound, INPUT_B_TOKEN)
       || !isSupportedOperand(bound, OUTPUT_TOKEN))
    {
        return std::nullopt;
    }
    if(*tryGetBoundValue<std::string>(bound, "input_a.dtype")
           != *tryGetBoundValue<std::string>(bound, "input_b.dtype")
       || *tryGetBoundValue<std::string>(bound, "input_a.dtype")
              != *tryGetBoundValue<std::string>(bound, "output.dtype"))
    {
        return std::nullopt;
    }
    return bound;
}

/**
 * @brief Graph-scoped operation check: the one fact that separates this engine's packs.
 *
 * Listing this second matcher is the whole cost of a pack, and two packs claiming the
 * same operation would be the authoring mistake, not two packs sharing the graph check.
 */
bool pointwiseOperationMatches(const BoundTokens& bound, data_objects::PointwiseMode operation)
{
    const auto* published = tryGetBoundValue<std::string>(bound, "pointwise.operation");
    return published != nullptr && *published == data_objects::EnumNamePointwiseMode(operation);
}

bool pointwiseAddMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return pointwiseOperationMatches(bound, data_objects::PointwiseMode::ADD);
}

bool pointwiseMulMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return pointwiseOperationMatches(bound, data_objects::PointwiseMode::MUL);
}

bool pointwiseSubMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return pointwiseOperationMatches(bound, data_objects::PointwiseMode::SUB);
}

/// Match each kernel using the published dtype.
bool pointwiseKernelMatches(const MatchContext& /*context*/,
                            const BoundTokens& bound,
                            const KernelDefinition& kernel)
{
    const auto* dataType = tryGetBoundValue<std::string>(bound, "input_a.dtype");
    return dataType != nullptr && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dataType;
}

double pointwiseScore(const MatchContext& /*context*/,
                      const BoundTokens& /*bound*/,
                      const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
PointwiseBinding pointwiseBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "pointwise dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(INPUT_A_TOKEN), read(INPUT_B_TOKEN), read(OUTPUT_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus the operand uids it launches with: everything read from the
/// graph, resolved once, owning nothing that points back into it.
class PreparedPointwise : public PreparedDispatch
{
public:
    PreparedPointwise(IngestorKernelCode code, PointwiseBinding binding)
        : _code(std::move(code))
        , _binding(binding)
    {
    }

    /// The kernel for the device this dispatch is running on. Resolved here rather than
    /// at prepare() because a plan outlives the handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const PointwiseBinding& binding() const
    {
        return _binding;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so a module
    // outlives every function resolved from it for the plan's lifetime.
    IngestorKernelCode _code;
    PointwiseBinding _binding;
};

std::string elementTypeFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "kernel '" + toString(kernel.kernelId) + "' declares unsupported dtype '" + dtype + "'");
}

const data_objects::TensorAttributes& firstInput(const MatchContext& context,
                                                 const PointwiseBinding& binding)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(binding.inputA);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched pointwise graph has no tensor for uid " + std::to_string(binding.inputA));
    }
    return *it->second;
}

/// The argument list this pack marshals: three device pointers, in operand order, matching
/// the launch() below one for one. It sits here rather than in the adapter that consumes it
/// so that it is edited alongside that launch -- a stale copy rejects the correct kernel
/// rather than the drifted one.
///
/// Names are empty and offsets zero because neither is compared for a HIP-produced kernel;
/// see requireSignatureMatch.
const std::vector<KernelArgument>& pointwiseKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_buffer, s_buffer, s_buffer};
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches a pointwise
 *        kernel. Shared across packs -- the operation is just the selected kernel's
 *        entry point. Splits per RFC 0017 §8.5: everything graph/kernel-derived
 *        resolves once at prepare(); execute() only resolves buffers and launches, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class PointwiseDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Same must-outlive contract. Which of the two is consulted is
    /// the selected kernel's source kind, decided in buildIngestorKernelCode.
    /// Device properties aren't held here -- they arrive per call via MatchContext,
    /// so each call compiles for the device it's actually for.
    PointwiseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)) == LARGE_BLOCK_SIZE
                   ? LARGE_BLOCK_WORKSPACE_BYTES
                   : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        // Reads the operand uids the graph match bound rather than re-deriving them.
        const auto binding = pointwiseBinding(bound);

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&firstInput(context, binding),
                                                  context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_POINTWISE_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_POINTWISE_BLOCK_SIZE", blockSize);

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, pointwiseKernelSignature());

        code.setBlockSize(blockSize, 1, 1);
        code.setGridSize(1, 1, 1);

        return std::make_unique<PreparedPointwise>(std::move(code), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedPointwise = dynamic_cast<const PreparedPointwise&>(prepared);
        const auto& binding = preparedPointwise.binding();

        const auto inputA
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputA, deviceBuffers, numDeviceBuffers);
        const auto inputB
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputB, deviceBuffers, numDeviceBuffers);
        const auto output
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.output, deviceBuffers, numDeviceBuffers);

        // Changing this argument list means changing pointwiseKernelSignature() with it.
        preparedPointwise.kernelForStream(handle.getStream())
            .launch(handle.getStream(), inputA.ptr, inputB.ptr, output.ptr);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& pointwiseKpackModuleCache()
{
    // Process-lifetime, and exposed rather than hidden inside the loader so a test can
    // observe that two dispatches over the same (archive, toc_key, arch) loaded one
    // module -- which is otherwise unobservable.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetPointwiseModuleCache()
{
    pointwiseKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so
/// it (and the compiler and loader it holds) must outlive every Container.
const PointwiseDispatchHandler& pointwiseDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(pointwiseKpackModuleCache());
    static const PointwiseDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerPointwiseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &pointwiseGraphMatches);
    scope.add(std::string(ADD_MATCHER_SYMBOL), &pointwiseAddMatches);
    scope.add(std::string(MUL_MATCHER_SYMBOL), &pointwiseMulMatches);
    scope.add(std::string(SUB_MATCHER_SYMBOL), &pointwiseSubMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &pointwiseKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &pointwiseScore);
    scope.add(std::string(DISPATCH_SYMBOL), &pointwiseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
