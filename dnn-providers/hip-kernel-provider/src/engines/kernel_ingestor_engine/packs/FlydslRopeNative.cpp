// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRopeCommon.hpp"

/**
 * @file FlydslRopeNative.cpp
 * @brief The flyDSL RoPE pack's native half: matching, scoring, dispatch, and the one
 *        function that registers them.
 *
 * Every kernel this pack selects is a pre-built HSACO (flydsl/kernels_src/kernels/rope),
 * one per x/y dtype. The kernel takes head_dim, rope_dim, rope_offset, output_scale and
 * every stride of x, freqs and y as arguments, so the only fact baked at pack time is the
 * element type of x and y, which kernel_match pairs with the graph. Everything else the
 * kernel cannot do is declined by graph_match (resolveRopeProblem()).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using flydsl_rope::RopeProblem;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_rope.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_rope.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_rope.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_rope.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";

constexpr std::string_view X_TOKEN = "flydsl_rope.x.uid";
constexpr std::string_view FREQS_TOKEN = "flydsl_rope.freqs.uid";
constexpr std::string_view Y_TOKEN = "flydsl_rope.y.uid";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched RoPE graph binds, in argument order.
struct FlydslRopeBinding
{
    int64_t x = 0;
    int64_t freqs = 0;
    int64_t y = 0;
};

/// The node this pack's matchers read, or nullptr if the graph isn't a single RoPE node.
/// Override-shape graphs are declined: the problem is resolved from the graph's shapes at
/// prepare.
const data_objects::RopeAttributes* ropeNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::RopeAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::RopeAttributes>();
}

/// Whether @p tensor declares a base alignment of at least, and a multiple of, @p bytes:
/// every access is one element wide.
bool declaresElementAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// The problem for the graph's single RoPE node, or nullopt if the kernels do not serve it.
std::optional<RopeProblem> ropeProblem(const MatchContext& context,
                                       const data_objects::RopeAttributes& attributes)
{
    const auto* x = flydsl_norm::findTensor(context, attributes.x_tensor_uid());
    const auto* freqs = flydsl_norm::findTensor(context, attributes.freqs_tensor_uid());
    const auto* y = flydsl_norm::findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || freqs == nullptr || y == nullptr)
    {
        return std::nullopt;
    }
    if(!flydsl_norm::isDeviceOperand(*x) || !flydsl_norm::isDeviceOperand(*freqs)
       || !flydsl_norm::isDeviceOperand(*y))
    {
        return std::nullopt;
    }
    // x and y in one of the built dtypes; freqs raw f32 angles, as cuDNN defines them.
    if(!flydsl_norm::isSupportedDataType(x->data_type()) || y->data_type() != x->data_type()
       || freqs->data_type() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto elementBytes = flydsl_norm::elementBytes(x->data_type());
    if(!declaresElementAlignment(*x, elementBytes) || !declaresElementAlignment(*y, elementBytes)
       || !declaresElementAlignment(*freqs, 4))
    {
        return std::nullopt;
    }
    return flydsl_rope::resolveRopeProblem(attributes, *x, *freqs, *y);
}

/**
 * @brief Graph-scoped applicability: a single RoPE node whose problem the pre-built
 *        kernels compute exactly.
 *
 * The kernels compute in f32: at least the precision an UNSET, FLOAT, HALF or BFLOAT16
 * compute type asks for; any other compute type declines. Rotation is rotate-half (the
 * only form the schema defines). y may alias x: each element is read and written by one
 * thread, read first.
 */
std::optional<BoundTokens> flydslRopeGraphMatches(const MatchContext& context)
{
    const auto* attributes = ropeNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    if(!ropeProblem(context, *attributes).has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes->x_tensor_uid();
    bound[std::string(FREQS_TOKEN)] = attributes->freqs_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes->y_tensor_uid();
    return bound;
}

/// Kernel-scoped applicability: the object built for x's dtype. Shape is runtime.
bool flydslRopeKernelMatches(const MatchContext& context,
                             const BoundTokens& /*bound*/,
                             const KernelDefinition& kernel)
{
    const auto* attributes = ropeNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* x = flydsl_norm::findTensor(context, attributes->x_tensor_uid());
    if(x == nullptr)
    {
        return false;
    }
    const auto dtype = flydsl_norm::kernelDataTypeName(x->data_type());
    return dtype.has_value() && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dtype;
}

/// One object per dtype, so at most one candidate survives kernel_match; the descriptor
/// priority is reproduced rather than a second opinion invented.
double flydslRopeScore(const MatchContext& /*context*/,
                       const BoundTokens& /*bound*/,
                       const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
FlydslRopeBinding flydslRopeBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_rope dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(X_TOKEN), read(FREQS_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything read from the graph, resolved once, owning nothing
/// that points back into it.
class PreparedFlydslRope : public PreparedDispatch
{
public:
    PreparedFlydslRope(IngestorKernelCode code,
                       FlydslRopeBinding binding,
                       RopeProblem problem,
                       float outputScale)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _outputScale(outputScale)
    {
    }

    /// The kernel for the device this dispatch is running on; see FlydslRmsNormNative.cpp.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslRopeBinding& binding() const
    {
        return _binding;
    }

    const RopeProblem& problem() const
    {
        return _problem;
    }

    float outputScale() const
    {
        return _outputScale;
    }

private:
    IngestorKernelCode _code;
    FlydslRopeBinding _binding;
    RopeProblem _problem;
    float _outputScale;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one
 *        and the `signature` every one of this pack's UKDs carries: x, freqs and y
 *        pointers; the four element strides of x, of freqs and of y (int64); head_dim,
 *        rope_dim and rope_offset (int32); output_scale (f32).
 *
 * Names are empty and offsets zero because neither is compared for this producer; see
 * requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslRopeKernelSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_pointer,
                                                         s_pointer,
                                                         s_pointer,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i32,
                                                         s_i32,
                                                         s_i32,
                                                         s_f32};
    return s_signature;
}

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and launches, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslRopeDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler, @param kpackLoader Must outlive this handler; see
    /// FlydslRmsNormNative.cpp.
    FlydslRopeDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

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
        const auto binding = flydslRopeBinding(bound);
        const auto* attributes = ropeNode(context);
        const auto problem
            = attributes == nullptr ? std::nullopt : ropeProblem(context, *attributes);
        if(!problem.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_rope: prepare() given a graph its matcher would decline");
        }

        const auto blockThreads
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)));

        // Built from the arch alone; see FlydslRmsNormNative.cpp for why not from x.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslRopeKernelSignature());

        // `one_block_per_row_d2_d1_d0`: x padded to [d0, d1, d2, D], a workgroup per row.
        code.setBlockSize(blockThreads, 1, 1);
        code.setGridSize(static_cast<unsigned int>(problem->dims[2]),
                         static_cast<unsigned int>(problem->dims[1]),
                         static_cast<unsigned int>(problem->dims[0]));

        return std::make_unique<PreparedFlydslRope>(
            std::move(code), binding, *problem, attributes->output_scale());
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedRope = dynamic_cast<const PreparedFlydslRope&>(prepared);
        const auto& binding = preparedRope.binding();
        const auto& problem = preparedRope.problem();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto freqs
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.freqs, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // Changing this argument list means changing flydslRopeKernelSignature() with it.
        preparedRope.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    x.ptr,
                    freqs.ptr,
                    y.ptr,
                    problem.xStrides[0],
                    problem.xStrides[1],
                    problem.xStrides[2],
                    problem.xStrides[3],
                    problem.freqsStrides[0],
                    problem.freqsStrides[1],
                    problem.freqsStrides[2],
                    problem.freqsStrides[3],
                    problem.yStrides[0],
                    problem.yStrides[1],
                    problem.yStrides[2],
                    problem.yStrides[3],
                    problem.headDim,
                    problem.ropeDim,
                    problem.ropeOffset,
                    preparedRope.outputScale());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslRopeKpackModuleCache()
{
    // Process-lifetime and this pack's own; see FlydslRmsNormNative.cpp.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslRopeModuleCache()
{
    flydslRopeKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime; see FlydslRmsNormNative.cpp.
const FlydslRopeDispatchHandler& flydslRopeDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslRopeKpackModuleCache());
    static const FlydslRopeDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslRopeSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslRopeGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslRopeKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslRopeScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslRopeDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
