// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslRmsNormNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/FlydslMoeRoutingCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"

/**
 * @file FlydslMoeRoutingNative.cpp
 * @brief The flyDSL MoE routing pack's native half (`hipkernel:flydsl_moe_routing`):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * Each object (`flydsl/kernels_src/kernels/moe_routing/moe_routing_kernel.py`, ported from
 * FlyDSL's `moe_sorting_kernel.py`) is a three-kernel counting sort over the row-major
 * `(s, k)` pairs of `topk_ids [S, top_k]`, launched in order on the plan's stream:
 *
 *   - the histogram kernel, one 256-thread block per span of pairs, writes each block's
 *     per-expert counts into the workspace (expert-major);
 *   - the prefix kernel, one 1024-thread block, scans them in place into each block's
 *     first row per expert, and writes `first_token_offset` (int32 or int64);
 *   - the scatter kernel, the histogram's grid, writes `token_index` and `token_ks`.
 *
 * Rows come out ordered by expert, then token, then slot, exactly as the CPU reference
 * orders them (a stable counting sort), and the same on every launch: the kernels use no
 * atomics. An id outside `[0, num_experts)` is not routed and the rows past
 * `first_token_offset[E]` are left untouched. Every shape is a kernel argument; an object
 * bounds only `num_experts` (`e_max`, its LDS table's size).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using flydsl_norm::Desc1D;
using flydsl_norm::findTensor;
using flydsl_norm::hasUsableShape;
using flydsl_norm::isDeviceOperand;
using flydsl_norm::isPackedRowMajor;
using namespace flydsl_moe_routing;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_moe_routing.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_moe_routing.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_moe_routing.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_moe_routing.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// The largest num_experts the object's LDS table holds.
constexpr std::string_view E_MAX_FIELD = "e_max";
/// The prefix and scatter kernels' symbols in the same object.
constexpr std::string_view PREFIX_SYMBOL_FIELD = "prefix_symbol";
constexpr std::string_view SCATTER_SYMBOL_FIELD = "scatter_symbol";

constexpr std::string_view TOPK_IDS_TOKEN = "flydsl_moe_routing.topk_ids.uid";
constexpr std::string_view OFFSETS_TOKEN = "flydsl_moe_routing.first_token_offset.uid";
constexpr std::string_view TOKEN_INDEX_TOKEN = "flydsl_moe_routing.token_index.uid";
constexpr std::string_view TOKEN_KS_TOKEN = "flydsl_moe_routing.token_ks.uid";
constexpr std::string_view TOKENS_TOKEN = "flydsl_moe_routing.tokens";
constexpr std::string_view TOP_K_TOKEN = "flydsl_moe_routing.top_k";
constexpr std::string_view EXPERTS_TOKEN = "flydsl_moe_routing.num_experts";
constexpr std::string_view OFFSET_IS_I64_TOKEN = "flydsl_moe_routing.offset_is_i64";

/// The tensor uids and extents a matched graph binds.
struct FlydslMoeRoutingBinding
{
    int64_t topkIds = 0;
    int64_t offsets = 0;
    int64_t tokenIndex = 0;
    int64_t tokenKs = 0;
    int64_t tokens = 0;
    int64_t topK = 0;
    int64_t experts = 0;
    bool offsetIsI64 = false;
};

const data_objects::MoeRoutingAttributes* moeRoutingNode(const MatchContext& context)
{
    // The extents are bound from the graph's shapes at prepare, which an override-shape
    // graph may replace at execute.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::MoeRoutingAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::MoeRoutingAttributes>();
}

/// Whether @p tensor declares a base alignment of at least, and a multiple of, @p bytes.
bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// Whether @p tensor's dims are exactly @p expected.
bool hasDims(const data_objects::TensorAttributes& tensor, const std::vector<int64_t>& expected)
{
    const auto& dims = *tensor.dims();
    if(dims.size() != expected.size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < dims.size(); ++axis)
    {
        if(dims.Get(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Graph-scoped applicability: one MoE routing node whose operands the objects
 *        address -- packed int32 ids, offsets and rows within the kernels' 32-bit offsets.
 */
std::optional<BoundTokens> flydslMoeRoutingGraphMatches(const MatchContext& context)
{
    const auto* attributes = moeRoutingNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto topkUid = attributes->topk_ids_tensor_uid();
    const auto offsetsUid = attributes->first_token_offset_tensor_uid();
    const auto indexUid = attributes->token_index_tensor_uid();
    const auto ksUid = attributes->token_ks_tensor_uid();
    // Four distinct tensors: the kernels read topk_ids while writing the other three.
    const std::vector<int64_t> uids = {topkUid, offsetsUid, indexUid, ksUid};
    for(size_t i = 0; i < uids.size(); ++i)
    {
        for(size_t j = i + 1; j < uids.size(); ++j)
        {
            if(uids[i] == uids[j])
            {
                return std::nullopt;
            }
        }
    }

    const auto* topkIds = findTensor(context, topkUid);
    const auto* offsets = findTensor(context, offsetsUid);
    const auto* tokenIndex = findTensor(context, indexUid);
    const auto* tokenKs = findTensor(context, ksUid);
    if(topkIds == nullptr || offsets == nullptr || tokenIndex == nullptr || tokenKs == nullptr)
    {
        return std::nullopt;
    }
    for(const auto* tensor : {topkIds, offsets, tokenIndex, tokenKs})
    {
        if(!isDeviceOperand(*tensor) || !hasUsableShape(*tensor, 1) || !isPackedRowMajor(*tensor))
        {
            return std::nullopt;
        }
    }

    // int32 ids and rows, read and written one 32-bit word at a time; the offsets as
    // int32 or int64 (written as two words).
    const bool offsetIsI64 = offsets->data_type() == data_objects::DataType::INT64;
    if(topkIds->data_type() != data_objects::DataType::INT32
       || tokenIndex->data_type() != data_objects::DataType::INT32
       || tokenKs->data_type() != data_objects::DataType::INT32
       || (!offsetIsI64 && offsets->data_type() != data_objects::DataType::INT32))
    {
        return std::nullopt;
    }
    if(!declaresAlignment(*topkIds, 4) || !declaresAlignment(*tokenIndex, 4)
       || !declaresAlignment(*tokenKs, 4) || !declaresAlignment(*offsets, offsetIsI64 ? 8 : 4))
    {
        return std::nullopt;
    }

    // topk_ids [S, top_k]; first_token_offset [E + 1, 1, 1]; the rows [1, S * top_k, 1].
    const auto& idDims = *topkIds->dims();
    if(idDims.size() != 2)
    {
        return std::nullopt;
    }
    const int64_t tokens = idDims.Get(0);
    const int64_t topK = idDims.Get(1);
    const int64_t experts = attributes->num_experts();
    if(tokens < 1 || topK < 1 || experts < 1 || experts > MAX_EXPERTS || tokens > MAX_PAIRS / topK)
    {
        return std::nullopt;
    }
    const int64_t pairs = tokens * topK;
    if(!hasDims(*offsets, {experts + 1, 1, 1}) || !hasDims(*tokenIndex, {1, pairs, 1})
       || !hasDims(*tokenKs, {1, pairs, 1}))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(TOPK_IDS_TOKEN)] = topkUid;
    bound[std::string(OFFSETS_TOKEN)] = offsetsUid;
    bound[std::string(TOKEN_INDEX_TOKEN)] = indexUid;
    bound[std::string(TOKEN_KS_TOKEN)] = ksUid;
    bound[std::string(TOKENS_TOKEN)] = tokens;
    bound[std::string(TOP_K_TOKEN)] = topK;
    bound[std::string(EXPERTS_TOKEN)] = experts;
    bound[std::string(OFFSET_IS_I64_TOKEN)] = static_cast<int64_t>(offsetIsI64 ? 1 : 0);
    return bound;
}

/// Kernel-scoped applicability: int32 ids, and num_experts within the object's table.
bool flydslMoeRoutingKernelMatches(const MatchContext& /*context*/,
                                   const BoundTokens& bound,
                                   const KernelDefinition& kernel)
{
    const auto experts = tryGetBoundInt(bound, EXPERTS_TOKEN);
    return experts.has_value() && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == "i32"
           && *experts <= kernel.getIntMetadata(std::string(E_MAX_FIELD));
}

/// One object per arch; the priority is the only order there is.
double flydslMoeRoutingScore(const MatchContext& /*context*/,
                             const BoundTokens& /*bound*/,
                             const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

FlydslMoeRoutingBinding flydslMoeRoutingBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_moe_routing dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    return {read(TOPK_IDS_TOKEN),
            read(OFFSETS_TOKEN),
            read(TOKEN_INDEX_TOKEN),
            read(TOKEN_KS_TOKEN),
            read(TOKENS_TOKEN),
            read(TOP_K_TOKEN),
            read(EXPERTS_TOKEN),
            read(OFFSET_IS_I64_TOKEN) != 0};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The three kernels plus everything read from the graph, resolved once.
class PreparedFlydslMoeRouting : public PreparedDispatch
{
public:
    PreparedFlydslMoeRouting(IngestorKernelCode histogram,
                             IngestorKernelCode prefix,
                             IngestorKernelCode scatter,
                             FlydslMoeRoutingBinding binding,
                             RoutingSplit split)
        : _histogram(std::move(histogram))
        , _prefix(std::move(prefix))
        , _scatter(std::move(scatter))
        , _binding(binding)
        , _split(split)
    {
    }

    compilation::IRunnableKernel& histogramForStream(hipStream_t stream) const
    {
        return _histogram.kernelForStream(stream);
    }

    compilation::IRunnableKernel& prefixForStream(hipStream_t stream) const
    {
        return _prefix.kernelForStream(stream);
    }

    compilation::IRunnableKernel& scatterForStream(hipStream_t stream) const
    {
        return _scatter.kernelForStream(stream);
    }

    const FlydslMoeRoutingBinding& binding() const
    {
        return _binding;
    }

    const RoutingSplit& split() const
    {
        return _split;
    }

private:
    IngestorKernelCode _histogram;
    IngestorKernelCode _prefix;
    IngestorKernelCode _scatter;
    FlydslMoeRoutingBinding _binding;
    RoutingSplit _split;
};

KernelArgument bufferArgument()
{
    return {"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
}

KernelArgument scalarArgument(uint32_t bytes)
{
    return {"by_value", bytes, 0, ""};
}

/// histogram(TopkIds, Counts, n_pairs, num_experts, span, num_blocks): each tensor a
/// pointer and its 1-D layout.
const std::vector<KernelArgument>& histogramSignature()
{
    static const std::vector<KernelArgument> s_signature{bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t))};
    return s_signature;
}

/// prefix(Counts, FirstTokenOffset, num_experts, num_blocks, offset_is_i64).
const std::vector<KernelArgument>& prefixSignature()
{
    static const std::vector<KernelArgument> s_signature{bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t))};
    return s_signature;
}

/// scatter(TopkIds, Counts, TokenIndex, TokenKs, n_pairs, num_experts, span, num_blocks,
/// top_k).
const std::vector<KernelArgument>& scatterSignature()
{
    static const std::vector<KernelArgument> s_signature{bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         bufferArgument(),
                                                         scalarArgument(sizeof(Desc1D)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t)),
                                                         scalarArgument(sizeof(int32_t))};
    return s_signature;
}

RoutingSplit splitFor(const MatchContext& context, const FlydslMoeRoutingBinding& binding)
{
    return routingSplit(binding.tokens * binding.topK,
                        binding.experts,
                        context.deviceProperties.multiProcessorCount);
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes the counts workspace and
 *        launches the histogram, prefix and scatter kernels.
 */
class FlydslMoeRoutingDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslMoeRoutingDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                    const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One int32 count per (expert, block), padded to the prefix kernel's 16-byte loads.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& /*kernel*/) const override
    {
        const auto binding = flydslMoeRoutingBinding(bound);
        return workspaceBytesFor(binding.experts, splitFor(context, binding).blocks);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslMoeRoutingBinding(bound);
        const auto split = splitFor(context, binding);

        const auto prefixSymbol = kernel.getStringMetadata(std::string(PREFIX_SYMBOL_FIELD));
        const auto scatterSymbol = kernel.getStringMetadata(std::string(SCATTER_SYMBOL_FIELD));
        if(prefixSymbol.empty() || scatterSymbol.empty())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_moe_routing: kernel '" + kernel.name
                    + "' names no prefix or scatter kernel");
        }

        // Built from the arch alone; see FlydslRmsNormNative.cpp's prepare().
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto histogram = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, histogramSignature());
        // The other two kernels are the same code object under other symbols: the same
        // archive entry, each with its own argument list.
        const auto sibling
            = [&](const std::string& symbol, const std::vector<KernelArgument>& signature) {
                  KernelDefinition definition = kernel;
                  definition.source.symbol = symbol;
                  definition.source.signature = signature;
                  return buildIngestorKernelCode(
                      _kernelCompiler, _kpackLoader, context, definition, options, signature);
              };
        auto prefix = sibling(prefixSymbol, prefixSignature());
        auto scatter = sibling(scatterSymbol, scatterSignature());

        const auto blocks = static_cast<unsigned int>(split.blocks);
        histogram.setBlockSize(TILE_THREADS, 1, 1);
        histogram.setGridSize(blocks, 1, 1);
        prefix.setBlockSize(PREFIX_THREADS, 1, 1);
        prefix.setGridSize(1, 1, 1);
        scatter.setBlockSize(TILE_THREADS, 1, 1);
        scatter.setGridSize(blocks, 1, 1);

        return std::make_unique<PreparedFlydslMoeRouting>(
            std::move(histogram), std::move(prefix), std::move(scatter), binding, split);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& routing = dynamic_cast<const PreparedFlydslMoeRouting&>(prepared);
        const auto& binding = routing.binding();
        const auto& split = routing.split();
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "flydsl_moe_routing needs its counts workspace");
        }

        const auto buffer = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        void* topkIds = buffer(binding.topkIds);
        void* offsets = buffer(binding.offsets);
        void* tokenIndex = buffer(binding.tokenIndex);
        void* tokenKs = buffer(binding.tokenKs);

        // The matcher bounds every extent within int32.
        const auto pairs = static_cast<int32_t>(binding.tokens * binding.topK);
        const auto experts = static_cast<int32_t>(binding.experts);
        const auto topK = static_cast<int32_t>(binding.topK);
        const auto span = static_cast<int32_t>(split.span);
        const auto blocks = static_cast<int32_t>(split.blocks);
        const int32_t offsetIsI64 = binding.offsetIsI64 ? 1 : 0;
        const Desc1D pairsDesc{pairs};
        const Desc1D countsDesc{experts * blocks};
        const Desc1D offsetsDesc{(experts + 1) * (binding.offsetIsI64 ? 2 : 1)};

        const auto stream = handle.getStream();
        // Changing any argument list means changing its signature function with it.
        routing.histogramForStream(stream).launch(
            stream, topkIds, pairsDesc, workspace, countsDesc, pairs, experts, span, blocks);
        routing.prefixForStream(stream).launch(
            stream, workspace, countsDesc, offsets, offsetsDesc, experts, blocks, offsetIsI64);
        routing.scatterForStream(stream).launch(stream,
                                                topkIds,
                                                pairsDesc,
                                                workspace,
                                                countsDesc,
                                                tokenIndex,
                                                pairsDesc,
                                                tokenKs,
                                                pairsDesc,
                                                pairs,
                                                experts,
                                                span,
                                                blocks,
                                                topK);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

/// This pack's own module cache; see flydslRmsNormKpackModuleCache().
compilation::KpackModuleCache& flydslMoeRoutingKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// Process-lifetime, as the other packs' handlers are.
const FlydslMoeRoutingDispatchHandler& flydslMoeRoutingDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslMoeRoutingKpackModuleCache());
    static const FlydslMoeRoutingDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void resetFlydslMoeRoutingModuleCache()
{
    flydslMoeRoutingKpackModuleCache().clear();
}

void registerFlydslMoeRoutingSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslMoeRoutingGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslMoeRoutingKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslMoeRoutingScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslMoeRoutingDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
