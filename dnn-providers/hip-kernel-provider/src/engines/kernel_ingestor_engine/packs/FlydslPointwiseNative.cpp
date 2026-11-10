// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
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
#include "engines/kernel_ingestor_engine/packs/FlydslPointwiseCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"

/**
 * @file FlydslPointwiseNative.cpp
 * @brief The flyDSL pointwise pack's native half: matching, scoring, dispatch, and the one
 *        function that registers them.
 *
 * Every object is one elementwise operation -- sigmoid, swish, exp, softplus, neg,
 * identity, add, sub, mul or div -- for one combination of bf16/f16/f32 storage types,
 * computing in f32. Shapes, strides and broadcasting are kernel arguments: prepare()
 * collapses the output to at most four axes and hands each operand an element stride per
 * axis, 0 where it broadcasts (flydsl_pointwise::collapse). What is baked, and so a matcher
 * obligation here, is the operation and the three storage types.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_pointwise::PointwiseGeometry;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_pointwise.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_pointwise.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_pointwise.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_pointwise.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view OPERATION_FIELD = "operation";
constexpr std::string_view IN1_DTYPE_FIELD = "in1_dtype";
constexpr std::string_view OUT_DTYPE_FIELD = "out_dtype";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view VEC_ELEMS_FIELD = "vec_elems";
/// `in1_dtype` of a unary object, which has no second input.
constexpr std::string_view NO_INPUT = "none";

constexpr std::string_view IN0_TOKEN = "flydsl_pointwise.in0.uid";
/// Bound only for a binary graph.
constexpr std::string_view IN1_TOKEN = "flydsl_pointwise.in1.uid";
constexpr std::string_view OUT_TOKEN = "flydsl_pointwise.out.uid";
constexpr std::string_view ELEMENTS_TOKEN = "flydsl_pointwise.elements";

/// The node this pack's matchers read, or nullptr if the graph isn't a single pointwise node.
const data_objects::PointwiseAttributes* pointwiseNode(const MatchContext& context)
{
    // Shapes are read from the graph at prepare, so override-shape graphs are declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::PointwiseAttributes>();
}

/// The f32 the kernel's `param` slot carries: the swish or softplus beta (1 when the graph
/// sets none), 1 for every other operation. nullopt for a beta the kernel cannot use: a
/// non-finite one, or a softplus beta that is not positive (it divides by it).
std::optional<float> operationParameter(const data_objects::PointwiseAttributes& attributes)
{
    switch(attributes.operation())
    {
    case data_objects::PointwiseMode::SWISH_FWD:
    {
        const float beta = attributes.swish_beta().value_or(1.0F);
        return std::isfinite(beta) ? std::optional<float>(beta) : std::nullopt;
    }
    case data_objects::PointwiseMode::SOFTPLUS_FWD:
    {
        const float beta = attributes.softplus_beta().value_or(1.0F);
        return std::isfinite(beta) && beta > 0.0F ? std::optional<float>(beta) : std::nullopt;
    }
    default:
        return 1.0F;
    }
}

/// The operands of a matched graph, read back from the node.
struct PointwiseOperands
{
    const data_objects::TensorAttributes* in0 = nullptr;
    /// nullptr for a unary operation.
    const data_objects::TensorAttributes* in1 = nullptr;
    const data_objects::TensorAttributes* out = nullptr;
};

/// The node's operands when their count fits the operation, else nullopt: a unary operation
/// names in_0 alone, a binary one in_0 and in_1; neither names in_2.
std::optional<PointwiseOperands> operands(const MatchContext& context,
                                          const data_objects::PointwiseAttributes& attributes)
{
    const bool binary = flydsl_pointwise::isBinary(attributes.operation());
    if(attributes.in_2_tensor_uid().has_value()
       || attributes.in_1_tensor_uid().has_value() != binary)
    {
        return std::nullopt;
    }
    PointwiseOperands result;
    result.in0 = flydsl_norm::findTensor(context, attributes.in_0_tensor_uid());
    result.out = flydsl_norm::findTensor(context, attributes.out_0_tensor_uid());
    if(binary)
    {
        result.in1 = flydsl_norm::findTensor(context, *attributes.in_1_tensor_uid());
        if(result.in1 == nullptr)
        {
            return std::nullopt;
        }
    }
    if(result.in0 == nullptr || result.out == nullptr)
    {
        return std::nullopt;
    }
    return result;
}

/// The geometry the kernel runs @p operands with, or nullopt if it cannot address them.
std::optional<PointwiseGeometry> geometryOf(const PointwiseOperands& operands)
{
    const auto out = flydsl_pointwise::operandShape(*operands.out);
    const auto in0 = flydsl_pointwise::operandShape(*operands.in0);
    if(!out.has_value() || !in0.has_value())
    {
        return std::nullopt;
    }
    if(operands.in1 == nullptr)
    {
        return flydsl_pointwise::collapse(*out, *in0, nullptr);
    }
    const auto in1 = flydsl_pointwise::operandShape(*operands.in1);
    if(!in1.has_value())
    {
        return std::nullopt;
    }
    return flydsl_pointwise::collapse(*out, *in0, &*in1);
}

/**
 * @brief Graph-scoped applicability: is this one pointwise node whose operation, storage
 *        types and shapes a shipped object computes exactly?
 */
std::optional<BoundTokens> flydslPointwiseGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = pointwiseNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    if(!flydsl_pointwise::operationName(attributes.operation()).has_value()
       || !operationParameter(attributes).has_value())
    {
        return std::nullopt;
    }

    // The objects compute in f32: at least the precision a FLOAT, HALF or BFLOAT16 compute
    // type asks for. DOUBLE or any other declines.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    const auto found = operands(context, attributes);
    if(!found.has_value())
    {
        return std::nullopt;
    }
    for(const auto* tensor : {found->in0, found->in1, found->out})
    {
        if(tensor != nullptr
           && (!flydsl_norm::isDeviceOperand(*tensor)
               || !flydsl_norm::isSupportedDataType(tensor->data_type())))
        {
            return std::nullopt;
        }
    }

    const auto geometry = geometryOf(*found);
    if(!geometry.has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(IN0_TOKEN)] = attributes.in_0_tensor_uid();
    if(found->in1 != nullptr)
    {
        bound[std::string(IN1_TOKEN)] = *attributes.in_1_tensor_uid();
    }
    bound[std::string(OUT_TOKEN)] = attributes.out_0_tensor_uid();
    bound[std::string(ELEMENTS_TOKEN)] = geometry->elements;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: the object built for this operation and these three
 *        storage types. Exactly one object fits a graph the graph matcher accepted.
 */
bool flydslPointwiseKernelMatches(const MatchContext& context,
                                  const BoundTokens& /*bound*/,
                                  const KernelDefinition& kernel)
{
    const auto* attributes = pointwiseNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto operation = flydsl_pointwise::operationName(attributes->operation());
    const auto found = operands(context, *attributes);
    if(!operation.has_value() || !found.has_value())
    {
        return false;
    }
    const auto in0 = flydsl_norm::kernelDataTypeName(found->in0->data_type());
    const auto out = flydsl_norm::kernelDataTypeName(found->out->data_type());
    const auto in1 = found->in1 == nullptr
                         ? std::optional<std::string>(std::string(NO_INPUT))
                         : flydsl_norm::kernelDataTypeName(found->in1->data_type());
    if(!in0.has_value() || !in1.has_value() || !out.has_value())
    {
        return false;
    }
    return kernel.getStringMetadata(std::string(OPERATION_FIELD)) == *operation
           && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *in0
           && kernel.getStringMetadata(std::string(IN1_DTYPE_FIELD)) == *in1
           && kernel.getStringMetadata(std::string(OUT_DTYPE_FIELD)) == *out;
}

/// The descriptor priority: one object serves each graph, so there is nothing to rank.
double flydslPointwiseScore(const MatchContext& /*context*/,
                            const BoundTokens& /*bound*/,
                            const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The uids a matched graph binds, in argument order.
struct FlydslPointwiseBinding
{
    int64_t in0 = 0;
    /// Absent for a unary operation.
    std::optional<int64_t> in1;
    int64_t out = 0;
};

FlydslPointwiseBinding flydslPointwiseBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_pointwise dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(IN0_TOKEN), tryGetBoundInt(bound, IN1_TOKEN), read(OUT_TOKEN)};
}

/// The kernel and every argument but the three pointers, resolved once at prepare().
class PreparedFlydslPointwise : public PreparedDispatch
{
public:
    PreparedFlydslPointwise(IngestorKernelCode code,
                            FlydslPointwiseBinding binding,
                            const PointwiseGeometry& geometry,
                            bool vectorPath,
                            float parameter)
        : _code(std::move(code))
        , _binding(binding)
        , _geometry(geometry)
        , _vectorPath(vectorPath)
        , _parameter(parameter)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslPointwiseBinding& binding() const
    {
        return _binding;
    }

    const PointwiseGeometry& geometry() const
    {
        return _geometry;
    }

    bool vectorPath() const
    {
        return _vectorPath;
    }

    float parameter() const
    {
        return _parameter;
    }

private:
    IngestorKernelCode _code;
    FlydslPointwiseBinding _binding;
    PointwiseGeometry _geometry;
    bool _vectorPath;
    float _parameter;
};

/**
 * @brief The argument list launch() passes, one for one, and the `signature` block every
 *        one of this pack's UKDs carries: Out, A, B (pointers); n, vec, d1, d2, d3 (int32);
 *        the output's, A's and B's four element strides (int64); param (f32).
 */
const std::vector<KernelArgument>& flydslPointwiseKernelSignature()
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_int32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_int64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const KernelArgument s_float{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const std::vector<KernelArgument> s_signature = [] {
        std::vector<KernelArgument> signature{s_buffer, s_buffer, s_buffer};
        signature.insert(signature.end(), 5, s_int32);
        signature.insert(signature.end(), 3 * flydsl_pointwise::KERNEL_AXES, s_int64);
        signature.push_back(s_float);
        return signature;
    }();
    return s_signature;
}

/// Every operand promises the 16-byte base alignment one dense-path access spans.
bool declaresVectorAlignment(const PointwiseOperands& operands)
{
    for(const auto* tensor : {operands.in0, operands.in1, operands.out})
    {
        if(tensor != nullptr && !flydsl_norm::declaresVectorAlignment(*tensor))
        {
            return false;
        }
    }
    return true;
}

class FlydslPointwiseDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslPointwiseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                   const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Elementwise: no scratch.
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
        const auto binding = flydslPointwiseBinding(bound);
        const auto* attributes = pointwiseNode(context);
        const auto found = attributes != nullptr ? operands(context, *attributes) : std::nullopt;
        const auto geometry = found.has_value() ? geometryOf(*found) : std::nullopt;
        const auto parameter
            = attributes != nullptr ? operationParameter(*attributes) : std::nullopt;
        if(!geometry.has_value() || !parameter.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_pointwise: prepare() given a graph its matcher would decline");
        }

        const auto blockThreads = kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD));
        const auto vecElems = kernel.getIntMetadata(std::string(VEC_ELEMS_FIELD));
        // The dense path reads and writes 16 bytes at a time from each base.
        const bool vectorPath = geometry->dense && declaresVectorAlignment(*found);

        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslPointwiseKernelSignature());

        // `one_thread_per_element_or_vector`.
        const auto perThread = vectorPath ? vecElems : int64_t{1};
        const auto threads = (geometry->elements + perThread - 1) / perThread;
        code.setBlockSize(static_cast<unsigned int>(blockThreads), 1, 1);
        code.setGridSize(
            static_cast<unsigned int>((threads + blockThreads - 1) / blockThreads), 1, 1);

        return std::make_unique<PreparedFlydslPointwise>(
            std::move(code), binding, *geometry, vectorPath, *parameter);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedPointwise = dynamic_cast<const PreparedFlydslPointwise&>(prepared);
        const auto& binding = preparedPointwise.binding();
        const auto& geometry = preparedPointwise.geometry();

        void* out
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.out, deviceBuffers, numDeviceBuffers).ptr;
        void* in0
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.in0, deviceBuffers, numDeviceBuffers).ptr;
        // A unary object never reads its second input; it is given the first's pointer
        // rather than a null.
        void* in1 = binding.in1.has_value() ? hipdnn_plugin_sdk::findDeviceBuffer(
                                                  *binding.in1, deviceBuffers, numDeviceBuffers)
                                                  .ptr
                                            : in0;

        const auto elements = static_cast<int32_t>(geometry.elements);
        const int32_t vec = preparedPointwise.vectorPath() ? 1 : 0;
        const auto d1 = static_cast<int32_t>(geometry.extents[1]);
        const auto d2 = static_cast<int32_t>(geometry.extents[2]);
        const auto d3 = static_cast<int32_t>(geometry.extents[3]);
        const auto& so = geometry.outStrides;
        const auto& sa = geometry.aStrides;
        const auto& sb = geometry.bStrides;
        const float parameter = preparedPointwise.parameter();
        // Changing this argument list means changing flydslPointwiseKernelSignature() with it.
        preparedPointwise.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    out,
                    in0,
                    in1,
                    elements,
                    vec,
                    d1,
                    d2,
                    d3,
                    so[0],
                    so[1],
                    so[2],
                    so[3],
                    sa[0],
                    sa[1],
                    sa[2],
                    sa[3],
                    sb[0],
                    sb[1],
                    sb[2],
                    sb[3],
                    parameter);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslPointwiseKpackModuleCache()
{
    // Process-lifetime and this pack's own; see flydslRmsNormKpackModuleCache().
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslPointwiseModuleCache()
{
    flydslPointwiseKpackModuleCache().clear();
}

namespace
{

const FlydslPointwiseDispatchHandler& flydslPointwiseDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslPointwiseKpackModuleCache());
    static const FlydslPointwiseDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslPointwiseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslPointwiseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslPointwiseKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslPointwiseScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslPointwiseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
