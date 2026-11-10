// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
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
#include "engines/kernel_ingestor_engine/packs/FlydslCausalConv1dCommon.hpp"

/**
 * @file FlydslCausalConv1dNative.cpp
 * @brief The flyDSL causal-conv1d pack's native half (`hipkernel:flydsl_causal_conv1d`):
 *        matching, scoring, dispatch, and the function that registers them.
 *
 * The kernel (`flydsl/kernels_src/kernels/causal_conv1d/causal_conv1d_kernel.py`, AITER's
 * FlyDSL prefill and decode convolutions restructured into one launch) computes the op as
 * causal_conv1d_attributes.fbs and CpuFpReferenceCausalConv1d define it: token-packed
 * sequences, an optional bias, IDENTITY or SILU, an optional initial state (zeros when
 * absent), an optional final state, an optional state_indices pool, and final_state bound to
 * initial_state's buffer for an in-place update. A decode step (T = N) and a prefill are the
 * same launch. T, N, C, the pool size, every stride and which optional operands are bound
 * are kernel arguments; an object bakes the dtype, the kernel width, the activation and its
 * schedule (`lanes` channels per thread, which C must be a multiple of).
 *
 * The offsets' order and the indices' range and distinctness are execute-time values the
 * caller guarantees, as the CPU reference requires (it throws). The kernel cannot report a
 * violation; a row outside [0, P) is neither read nor written, so a bad index stays in
 * bounds.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

using namespace flydsl_causal_conv1d;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_causal_conv1d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_causal_conv1d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_causal_conv1d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_causal_conv1d.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view WIDTH_FIELD = "width";
/// `identity` or `silu`.
constexpr std::string_view ACTIVATION_FIELD = "activation";
/// Contiguous channels per thread; C must be a multiple of it.
constexpr std::string_view LANES_FIELD = "lanes";
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view TILE_TOKENS_FIELD = "tile_tokens";

constexpr std::string_view X_TOKEN = "flydsl_causal_conv1d.x.uid";
constexpr std::string_view WEIGHT_TOKEN = "flydsl_causal_conv1d.weight.uid";
constexpr std::string_view CU_SEQLENS_TOKEN = "flydsl_causal_conv1d.cu_seqlens.uid";
constexpr std::string_view Y_TOKEN = "flydsl_causal_conv1d.y.uid";
constexpr std::string_view BIAS_TOKEN = "flydsl_causal_conv1d.bias.uid";
constexpr std::string_view INITIAL_STATE_TOKEN = "flydsl_causal_conv1d.initial_state.uid";
constexpr std::string_view STATE_INDICES_TOKEN = "flydsl_causal_conv1d.state_indices.uid";
constexpr std::string_view FINAL_STATE_TOKEN = "flydsl_causal_conv1d.final_state.uid";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The node this pack's matchers read, or nullptr if the graph isn't a single causal-conv1d
/// node. Extents are bound from the graph's shapes at prepare, so override-shape graphs are
/// declined.
const data_objects::CausalConv1dAttributes* causalConv1dNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::CausalConv1dAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::CausalConv1dAttributes>();
}

std::optional<CausalConv1dProblem> problemFor(const MatchContext& context)
{
    const auto* attributes = causalConv1dNode(context);
    return attributes == nullptr ? std::nullopt : resolveCausalConv1dProblem(context, *attributes);
}

/**
 * @brief Graph-scoped applicability: a single causal-conv1d node the shipped objects compute
 *        exactly (resolveCausalConv1dProblem lists the conditions). Binds every operand uid.
 */
std::optional<BoundTokens> flydslCausalConv1dGraphMatches(const MatchContext& context)
{
    const auto* attributes = causalConv1dNode(context);
    if(attributes == nullptr || !resolveCausalConv1dProblem(context, *attributes).has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes->x_tensor_uid();
    bound[std::string(WEIGHT_TOKEN)] = attributes->weight_tensor_uid();
    bound[std::string(CU_SEQLENS_TOKEN)] = attributes->cu_seqlens_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes->y_tensor_uid();
    const std::array<std::pair<std::string_view, flatbuffers::Optional<int64_t>>, 4> optional{{
        {BIAS_TOKEN, attributes->bias_tensor_uid()},
        {INITIAL_STATE_TOKEN, attributes->initial_state_tensor_uid()},
        {STATE_INDICES_TOKEN, attributes->state_indices_tensor_uid()},
        {FINAL_STATE_TOKEN, attributes->final_state_tensor_uid()},
    }};
    for(const auto& [token, uid] : optional)
    {
        if(uid.has_value())
        {
            bound[std::string(token)] = *uid;
        }
    }
    return bound;
}

/**
 * @brief Kernel-scoped applicability: the object's dtype, kernel width and activation are the
 *        graph's, C is a multiple of its `lanes`, and its grid fits the device's limits.
 */
bool flydslCausalConv1dKernelMatches(const MatchContext& context,
                                     const BoundTokens& /*bound*/,
                                     const KernelDefinition& kernel)
{
    const auto problem = problemFor(context);
    if(!problem.has_value())
    {
        return false;
    }
    const auto dtype = flydsl_norm::kernelDataTypeName(problem->dataType);
    const std::string activation = problem->silu ? "silu" : "identity";
    if(!dtype.has_value() || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype
       || kernel.getIntMetadata(std::string(WIDTH_FIELD)) != problem->width
       || kernel.getStringMetadata(std::string(ACTIVATION_FIELD)) != activation)
    {
        return false;
    }
    const auto lanes = kernel.getIntMetadata(std::string(LANES_FIELD));
    if(lanes <= 0 || problem->channels % lanes != 0)
    {
        return false;
    }
    return launchGrid(*problem,
                      kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)),
                      lanes,
                      kernel.getIntMetadata(std::string(TILE_TOKENS_FIELD)))
        .has_value();
}

/// The descriptors' priority: the 4-lane objects (100) above the 1-lane ones (10).
double flydslCausalConv1dScore(const MatchContext& /*context*/,
                               const BoundTokens& /*bound*/,
                               const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

int64_t requireBound(const BoundTokens& bound, std::string_view token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "flydsl_causal_conv1d dispatch is missing bound token '" + std::string(token)
                + "', or it does not hold an integer");
    }
    return *value;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The tensor uids a matched graph binds; the optional ones are absent when unbound.
struct FlydslCausalConv1dBinding
{
    int64_t x = 0;
    int64_t weight = 0;
    int64_t cuSeqlens = 0;
    int64_t y = 0;
    std::optional<int64_t> bias;
    std::optional<int64_t> initialState;
    std::optional<int64_t> stateIndices;
    std::optional<int64_t> finalState;
};

/// The compiled kernel plus everything read from the graph, resolved once.
class PreparedFlydslCausalConv1d : public PreparedDispatch
{
public:
    PreparedFlydslCausalConv1d(IngestorKernelCode code,
                               FlydslCausalConv1dBinding binding,
                               CausalConv1dProblem problem)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslCausalConv1dBinding& binding() const
    {
        return _binding;
    }

    const CausalConv1dProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    FlydslCausalConv1dBinding _binding;
    CausalConv1dProblem _problem;
};

/**
 * @brief The argument list this pack launches with, mirroring launch() below one for one:
 *        eight addresses (x, weight, bias, initial_state, final_state, cu_seqlens,
 *        state_indices, y) and eleven int64 element strides, all by value, then seven int32s
 *        (T, N, C, the state rows, the two index widths in words, the flags). Names are empty
 *        and offsets zero because neither is compared for this producer; see
 *        requireSignatureMatch.
 */
const std::vector<KernelArgument>& flydslCausalConv1dKernelSignature()
{
    static const KernelArgument s_long{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const KernelArgument s_int{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature = [] {
        std::vector<KernelArgument> signature(19, s_long);
        signature.insert(signature.end(), 7, s_int);
        return signature;
    }();
    return s_signature;
}

/// One operand's bytes as [begin, end), for the aliasing checks at launch.
struct ByteRange
{
    uintptr_t begin = 0;
    uintptr_t end = 0;

    bool overlaps(const ByteRange& other) const
    {
        return begin < other.end && other.begin < end;
    }
};

ByteRange rangeOf(const void* pointer,
                  const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& strides,
                  int64_t elementBytes)
{
    int64_t last = 0;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        last += (dims[axis] - 1) * strides[axis];
    }
    const auto begin = reinterpret_cast<uintptr_t>(pointer);
    return {begin, begin + static_cast<uintptr_t>((last + 1) * elementBytes)};
}

int64_t addressOf(const void* pointer)
{
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(pointer));
}

/**
 * @brief Sizes and launches a flyDSL causal-conv1d kernel. Everything graph/kernel-derived
 *        resolves at prepare(); launch() resolves buffers, checks the aliasing only it can
 *        see, and launches.
 */
class FlydslCausalConv1dDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    FlydslCausalConv1dDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                      const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One pass, no scratch: the sequence blocks own the state, the token tiles read x only.
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
        FlydslCausalConv1dBinding binding;
        binding.x = requireBound(bound, X_TOKEN);
        binding.weight = requireBound(bound, WEIGHT_TOKEN);
        binding.cuSeqlens = requireBound(bound, CU_SEQLENS_TOKEN);
        binding.y = requireBound(bound, Y_TOKEN);
        binding.bias = tryGetBoundInt(bound, BIAS_TOKEN);
        binding.initialState = tryGetBoundInt(bound, INITIAL_STATE_TOKEN);
        binding.stateIndices = tryGetBoundInt(bound, STATE_INDICES_TOKEN);
        binding.finalState = tryGetBoundInt(bound, FINAL_STATE_TOKEN);

        const auto problem = problemFor(context);
        const auto grid = problem.has_value()
                              ? launchGrid(*problem,
                                           kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD)),
                                           kernel.getIntMetadata(std::string(LANES_FIELD)),
                                           kernel.getIntMetadata(std::string(TILE_TOKENS_FIELD)))
                              : std::nullopt;
        if(!grid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_causal_conv1d: prepare() given a graph or kernel '" + kernel.name
                    + "' its matchers would decline");
        }

        // Build defines are ignored on the kpack path; built from the arch alone, as the
        // other flyDSL packs do.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslCausalConv1dKernelSignature());

        // `sequences_plus_token_tiles`: grid x is the N sequence blocks, then the token
        // tiles; grid y the channel blocks.
        code.setBlockSize(
            static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_THREADS_FIELD))),
            1,
            1);
        code.setGridSize(
            static_cast<unsigned int>(grid->first), static_cast<unsigned int>(grid->second), 1);

        return std::make_unique<PreparedFlydslCausalConv1d>(std::move(code), binding, *problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedConv = dynamic_cast<const PreparedFlydslCausalConv1d&>(prepared);
        const auto& binding = preparedConv.binding();
        const auto& problem = preparedConv.problem();

        const auto find = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        const auto findOptional = [&](const std::optional<int64_t>& uid) -> void* {
            return uid.has_value() ? find(*uid) : nullptr;
        };
        void* x = find(binding.x);
        void* weight = find(binding.weight);
        void* cuSeqlens = find(binding.cuSeqlens);
        void* y = find(binding.y);
        void* bias = findOptional(binding.bias);
        void* initialState = findOptional(binding.initialState);
        void* stateIndices = findOptional(binding.stateIndices);
        void* finalState = findOptional(binding.finalState);

        checkAliasing(problem, x, y, initialState, finalState);
        // The offsets and indices are read through scalar loads, which need word alignment.
        if(addressOf(cuSeqlens) % 4 != 0
           || (stateIndices != nullptr && addressOf(stateIndices) % 4 != 0))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "flydsl_causal_conv1d: cu_seqlens and state_indices must be 4-byte aligned");
        }

        // An unbound operand's address is never dereferenced (its flag is clear); x's stands
        // in, so every address argument is a valid mapping.
        const auto orX = [x](void* pointer) { return addressOf(pointer != nullptr ? pointer : x); };
        const int64_t xAddress = addressOf(x);
        const int64_t weightAddress = addressOf(weight);
        const int64_t biasAddress = orX(bias);
        const int64_t initialAddress = orX(initialState);
        const int64_t finalAddress = orX(finalState);
        const int64_t cuSeqlensAddress = addressOf(cuSeqlens);
        const int64_t stateIndicesAddress
            = addressOf(stateIndices != nullptr ? stateIndices : cuSeqlens);
        const int64_t yAddress = addressOf(y);
        const int64_t xStride = problem.xStride;
        const int64_t yStride = problem.yStride;
        const int64_t weightStrideC = problem.weightStrideC;
        const int64_t weightStrideK = problem.weightStrideK;
        const int64_t biasStride = problem.biasStride;
        const auto& initialStrides = problem.initialStrides;
        const auto& finalStrides = problem.finalStrides;
        const auto tokens = static_cast<int32_t>(problem.tokens);
        const auto sequences = static_cast<int32_t>(problem.sequences);
        const auto channels = static_cast<int32_t>(problem.channels);
        const auto stateRows = static_cast<int32_t>(problem.stateRows);
        const int32_t cuSeqlensStep = problem.cuSeqlensStep;
        const int32_t stateIndicesStep = problem.stateIndicesStep;
        const int32_t flags = problem.flags;

        // Changing this argument list means changing flydslCausalConv1dKernelSignature()
        // with it.
        preparedConv.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    xAddress,
                    weightAddress,
                    biasAddress,
                    initialAddress,
                    finalAddress,
                    cuSeqlensAddress,
                    stateIndicesAddress,
                    yAddress,
                    xStride,
                    yStride,
                    weightStrideC,
                    weightStrideK,
                    biasStride,
                    initialStrides[0],
                    initialStrides[1],
                    initialStrides[2],
                    finalStrides[0],
                    finalStrides[1],
                    finalStrides[2],
                    tokens,
                    sequences,
                    channels,
                    stateRows,
                    cuSeqlensStep,
                    stateIndicesStep,
                    flags);
    }

private:
    /// Refuses the aliasing the kernel does not compute as the CPU reference does. y may not
    /// share bytes with x (a token tile reads rows another tile writes); final_state may not
    /// share bytes with x or y, nor with initial_state unless it is the same buffer in the
    /// same layout -- the in-place update, where each row is read and then written by the one
    /// block that owns it.
    static void checkAliasing(const CausalConv1dProblem& problem,
                              const void* x,
                              const void* y,
                              const void* initialState,
                              const void* finalState)
    {
        const auto bytes = flydsl_norm::elementBytes(problem.dataType);
        const std::vector<int64_t> rowDims{problem.tokens, problem.channels};
        const auto xRange = rangeOf(x, rowDims, {problem.xStride, 1}, bytes);
        const auto yRange = rangeOf(y, rowDims, {problem.yStride, 1}, bytes);
        bool refused = xRange.overlaps(yRange);
        if(finalState != nullptr)
        {
            const std::vector<int64_t> stateDims{
                problem.stateRows, problem.channels, problem.width - 1};
            const auto& fs = problem.finalStrides;
            const auto finalRange = rangeOf(finalState, stateDims, {fs[0], fs[1], fs[2]}, bytes);
            refused = refused || finalRange.overlaps(xRange) || finalRange.overlaps(yRange);
            if(initialState != nullptr)
            {
                const auto& is = problem.initialStrides;
                const auto initialRange
                    = rangeOf(initialState, stateDims, {is[0], is[1], is[2]}, bytes);
                const bool inPlace = initialState == finalState && is == fs;
                refused = refused || (!inPlace && finalRange.overlaps(initialRange));
            }
        }
        if(refused)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "flydsl_causal_conv1d: y must not alias x, and final_state may alias only "
                "initial_state, as the same buffer in the same layout");
        }
    }

    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslCausalConv1dKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetFlydslCausalConv1dModuleCache()
{
    flydslCausalConv1dKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer.
const FlydslCausalConv1dDispatchHandler& flydslCausalConv1dDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslCausalConv1dKpackModuleCache());
    static const FlydslCausalConv1dDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                     s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslCausalConv1dSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslCausalConv1dGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslCausalConv1dKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslCausalConv1dScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslCausalConv1dDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
