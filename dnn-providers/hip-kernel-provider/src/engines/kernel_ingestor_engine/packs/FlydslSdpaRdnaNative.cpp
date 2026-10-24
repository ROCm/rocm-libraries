// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaCommon.hpp"

/**
 * @file FlydslSdpaRdnaNative.cpp
 * @brief The flyDSL RDNA SDPA prefill pack's native half: kernel matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * Every kernel this pack selects is a pre-built HSACO, checked in as an `hsaco` UKD of the
 * production descriptor root and packed into the per-arch `hip_kernel_provider_<arch>.kpack`.
 * It is compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py`. That kernel bakes
 * only dtype, head_dim, the causal flag and its tile; batch, both sequence lengths, both
 * head counts, every stride, the softmax scale and the causal offset are kernel
 * arguments. So one object serves every shape and layout of its (dtype, head_dim,
 * causal) class, and the matcher's job is to decline exactly what the kernel cannot
 * compute -- see `flydsl/COVERAGE.md` §5 for the list, each with its reason.
 *
 * One kernel family per engine: the split-KV decode kernel is another engine's
 * (`hipkernel:flydsl_sdpa_rdna_decode`, FlydslSdpaRdnaDecodeNative.cpp). The two share the
 * graph matcher and the checks in FlydslSdpaRdnaCommon.hpp; which serves a graph both can
 * is the heuristic's to decide, not a rule here.
 *
 * Symbol names are restated here rather than shared via a header, since a descriptor can't
 * reference a C++ constant; the loader pre-flights every symbol a descriptor names, so a
 * mismatched string is caught without becoming a compile error.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_rdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_rdna.dispatch";

/// A candidate's launch tile, or nullopt when its metadata describes none the kernel can
/// have been built with.
struct SdpaTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
    int64_t dvSplit = 1;
    /// Waves per 16 query rows: 2 for an hd_pair object, else 1.
    int64_t wavesPerRowTile = 1;
};

std::optional<SdpaTile> candidateTile(const KernelDefinition& kernel)
{
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    const auto dvSplit = integerMetadata(kernel, DV_SPLIT_FIELD);
    const auto hdPair = integerMetadata(kernel, HD_PAIR_FIELD);
    if(!blockM.has_value() || !blockN.has_value() || !dvSplit.has_value() || *blockM <= 0
       || *blockM % WAVE_ROWS != 0 || *blockN <= 0 || *blockN % KV_SUB_TILE != 0 || *dvSplit <= 0
       || !hdPair.has_value() || (*hdPair != 0 && *hdPair != 1))
    {
        return std::nullopt;
    }
    return SdpaTile{*blockM, *blockN, *dvSplit, *hdPair == 1 ? 2 : 1};
}

/// The workgroup count `batch_x_qtiles_x_heads_x_dv_split`, or nullopt past int32.
std::optional<int64_t> candidateGrid(const SdpaProblem& problem, const SdpaTile& tile)
{
    const auto queryTiles = (problem.seqLenQ + tile.blockM - 1) / tile.blockM;
    const auto grid = checkedProduct({problem.batch, queryTiles, problem.numHeads, tile.dvSplit});
    if(!grid.has_value() || *grid > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return grid;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
 *
 * dtype, head_dim (or, for a generic object, its largest), the causal variant, the bias and
 * the sink are baked; everything else is a runtime argument, so they are the whole
 * comparison.
 */
bool flydslSdpaRdnaKernelMatches(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel)
{
    SdpaProblem problem;
    if(!familyMatches(context, bound, kernel, problem))
    {
        return false;
    }
    const auto tile = candidateTile(kernel);
    if(!tile.has_value())
    {
        return false;
    }
    // The column split must divide the object's head into whole tiles, and the split grid --
    // larger than the graph-scoped bound assumed -- must still fit the launch.
    const auto headDimMax = integerMetadata(kernel, HEAD_DIM_MAX_FIELD);
    return headDimMax.has_value() && *headDimMax % tile->dvSplit == 0
           && candidateGrid(problem, *tile).has_value();
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * The descriptor `priority`, unchanged. Several candidates can survive a match: a generic
 * object serves every multiple-of-8 head_dim up to its head_dim_max, so it matches beside
 * the specialized object for that head_dim, and the generic tiers' lower priorities put the
 * specialized one first. Autotune measures every survivor; this order is the fallback.
 */
double flydslSdpaRdnaScore(const MatchContext& /*context*/,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The argument list this pack launches with, mirroring launch() one for one, and
 *        the `signature` block every one of this pack's UKDs carries.
 *
 * Q, K, V, O, LSE pointers; seq_len_q, seq_len_kv, num_heads, kv_group, right_bound,
 * left_bound, align_bottom_right, flags (i32: bit 0 the LSE output, bit 1 per-batch
 * lengths, bit 2 padded rows filled, bit 3 packed sequences); scale (f32); the (batch, sequence, head)
 * element strides of Q, K, V, O and LSE (i64); then the BIAS pointer and its (batch, head,
 * query, key) element strides (i64), the runtime head_dim (i32, read only by a generic
 * object), the query heads per V head (i32), the SINK pointer and its head stride (i64,
 * read only by a has_sink object), the per-batch query and key length pointers (null for a
 * dense graph) and the packed-sequence offset tables (null unless packed), and the dropout
 * seed and offset1 pointers, offset2 (i64), threshold (i32) and scale (f32), read by a
 * has_dropout object only -- each appended so the arguments before them kept their offsets.
 *
 * Names are empty and offsets zero because neither is compared for this producer -- FlyDSL
 * emits no argument names; see requireSignatureMatch. The semantic names are recorded per
 * object in the `manifest.json` beside it, under `descriptors/FlyDSL/sdpa_rdna/<arch>/`.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaKernelSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_i32,     s_i32,     s_i32,
        s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_f32,     s_i64,     s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_pointer, s_i64,     s_i64,
        s_i64,     s_i64,     s_i32,     s_i32,     s_pointer, s_i64,     s_pointer, s_pointer,
        s_pointer, s_pointer, s_pointer, s_pointer, s_i64,     s_i32,     s_f32};
    return s_signature;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedFlydslSdpaRdna : public PreparedDispatch
{
public:
    PreparedFlydslSdpaRdna(IngestorKernelCode code,
                           FlydslSdpaRdnaBinding binding,
                           SdpaProblem problem,
                           std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                           std::array<OperandStrides, 5> strides,
                           BiasStrides biasStrides,
                           PhiloxSources philox,
                           std::optional<DropoutMaskLaunch> dropoutMask)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
        , _biasStrides(biasStrides)
        , _philox(philox)
        , _dropoutMask(std::move(dropoutMask))
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaRdnaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

    /// Q, K, V, O and the LSE output (zeros when there is none), in that order.
    const std::array<OperandStrides, 5>& strides() const
    {
        return _strides;
    }

    /// The bias's (batch, head, query, key) strides; zeros when there is none.
    const BiasStrides& biasStrides() const
    {
        return _biasStrides;
    }

    /// Where the dropout seed and offset come from.
    const PhiloxSources& philox() const
    {
        return _philox;
    }

    /// The RNG dump's writer, launched after the attention launch; empty without a dump.
    const std::optional<DropoutMaskLaunch>& dropoutMask() const
    {
        return _dropoutMask;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaRdnaBinding _binding;
    SdpaProblem _problem;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    std::array<OperandStrides, 5> _strides;
    BiasStrides _biasStrides;
    PhiloxSources _philox;
    std::optional<DropoutMaskLaunch> _dropoutMask;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslSdpaRdnaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime. Unused in
    /// practice -- every kernel this pack selects comes from a kpack archive -- but
    /// buildIngestorKernelCode decides that from the kernel's source kind, not from here.
    /// @param kpackLoader Same must-outlive contract.
    FlydslSdpaRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernel's only scratch is LDS and registers; a dropout seed given by value needs 8
    /// bytes to be read from.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& /*kernel*/) const override
    {
        return philoxSources(context, flydslSdpaRdnaBinding(bound)).seedValue.has_value()
                   ? SEED_WORKSPACE_BYTES
                   : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaRdnaBinding(bound);

        // kernel_match declines a candidate without a usable tile, so this is defence in
        // depth: the grid and block below come from block_m, and there is no tile to
        // substitute that would not launch this object with another's geometry.
        const auto tile = candidateTile(kernel);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m/block_n tile");
        }

        const auto& q = requireTensor(context, binding.q);
        const auto& k = requireTensor(context, binding.k);
        const auto& v = requireTensor(context, binding.v);
        const auto& o = requireTensor(context, binding.o);
        auto problem = problemFor(q, k);
        problem.numVHeads = v.dims()->Get(HEAD_AXIS);
        problem.headDimV = v.dims()->Get(HEAD_DIM_AXIS);

        // Build defines are ignored on the kpack path, but the signature still wants an
        // options object. Built from the arch alone: the tensor overload classifies a 4-D
        // stride order as NCHW or NHWC and throws for anything else, which attention
        // layouts are.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);

        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            flydslSdpaRdnaKernelSignature());

        // `batch_x_qtiles_x_heads_x_dv_split`: one workgroup per (batch, query tile,
        // head, output-column tile), each block_m / 16 waves of 32. kernel_match
        // bounded it.
        const auto gridX = candidateGrid(problem, *tile);
        if(!gridX.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: launch grid exceeds int32 for kernel '"
                    + toString(kernel.kernelId) + "'");
        }
        code.setBlockSize(
            static_cast<unsigned int>(tile->blockM / WAVE_ROWS * tile->wavesPerRowTile * WAVE_SIZE),
            1,
            1);
        code.setGridSize(static_cast<unsigned int>(*gridX), 1, 1);

        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }

        BiasStrides biasStrides{0, 0, 0, 0};
        if(binding.bias != NO_BIAS)
        {
            const auto strides = servableBiasStrides(requireTensor(context, binding.bias),
                                                     problem.batch,
                                                     problem.numHeads,
                                                     problem.seqLenQ,
                                                     problem.seqLenKv,
                                                     problem.dataType);
            if(!strides.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_rdna: the bound bias is not one the kernel can read");
            }
            biasStrides = *strides;
        }

        // Packed (THD) sequences (kernel modification 29): a batch's base is its first
        // token's row times the batch stride, so each operand's batch stride is its sequence
        // stride (the LSE is packed through Q's table alike, as the graph match requires).
        std::array<OperandStrides, 5> strides{
            stridesOf(q),
            stridesOf(k),
            stridesOf(v),
            stridesOf(o),
            binding.stats == NO_STATS ? OperandStrides{}
                                      : stridesOf(requireTensor(context, binding.stats))};
        if(binding.packed())
        {
            for(auto& operand : strides)
            {
                operand.batch = operand.sequence;
            }
        }

        // The RNG dump's writer (FlydslSdpaRngDump.hpp), when the graph names a dump.
        auto dropoutMask = prepareDropoutMask(_kpackLoader,
                                              context,
                                              kernel,
                                              binding.rngDump,
                                              problem.batch,
                                              problem.numHeads,
                                              problem.seqLenQ,
                                              problem.seqLenKv);

        return std::make_unique<PreparedFlydslSdpaRdna>(std::move(code),
                                                        binding,
                                                        problem,
                                                        scaleOperand,
                                                        strides,
                                                        biasStrides,
                                                        philoxSources(context, binding),
                                                        std::move(dropoutMask));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaRdna&>(prepared);
        const auto& binding = preparedSdpa.binding();
        const auto& problem = preparedSdpa.problem();
        const float scale
            = launchScale(binding, preparedSdpa.scaleOperand(), deviceBuffers, numDeviceBuffers);

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        // The kernel writes the LSE output only when flags bit 0 is set, so with no stats tensor
        // the pointer is never dereferenced. The bias and the sink are read only by the
        // has_bias and has_sink objects, which kernel_match pairs with a bound one.
        void* stats = optionalBuffer(binding.stats, deviceBuffers, numDeviceBuffers);
        void* bias = optionalBuffer(binding.bias, deviceBuffers, numDeviceBuffers);
        void* sink = optionalBuffer(binding.sink, deviceBuffers, numDeviceBuffers);
        const int64_t sinkStride = binding.sinkStride;
        // Each batch's lengths under padding; null (the launch's lengths) for a dense graph.
        void* seqLenQ = optionalBuffer(binding.seqLenQ, deviceBuffers, numDeviceBuffers);
        void* seqLenKv = optionalBuffer(binding.seqLenKv, deviceBuffers, numDeviceBuffers);
        // Each packed sequence's token range (kernel modification 29); null when not packed.
        void* rowOffQ = optionalBuffer(binding.raggedOffsetQ, deviceBuffers, numDeviceBuffers);
        void* rowOffKv = optionalBuffer(binding.raggedOffsetKv, deviceBuffers, numDeviceBuffers);
        // Read by a has_dropout object only (kernel modification 27).
        const auto dropout = philoxLaunchArgs(binding,
                                              preparedSdpa.philox(),
                                              deviceBuffers,
                                              numDeviceBuffers,
                                              workspace,
                                              handle.getStream());

        const auto& s = preparedSdpa.strides();
        const auto& b = preparedSdpa.biasStrides();

        // Changing this argument list means changing flydslSdpaRdnaKernelSignature() with it.
        preparedSdpa.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    stats,
                    static_cast<int32_t>(problem.seqLenQ),
                    static_cast<int32_t>(problem.seqLenKv),
                    static_cast<int32_t>(problem.numHeads),
                    static_cast<int32_t>(problem.numHeads / problem.numKvHeads),
                    static_cast<int32_t>(binding.rightBound),
                    static_cast<int32_t>(binding.leftBound),
                    static_cast<int32_t>(binding.alignBottomRight),
                    rdnaLaunchFlags(binding),
                    scale,
                    s[0].batch,
                    s[0].sequence,
                    s[0].head,
                    s[1].batch,
                    s[1].sequence,
                    s[1].head,
                    s[2].batch,
                    s[2].sequence,
                    s[2].head,
                    s[3].batch,
                    s[3].sequence,
                    s[3].head,
                    s[4].batch,
                    s[4].sequence,
                    s[4].head,
                    bias,
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    static_cast<int32_t>(problem.headDim),
                    static_cast<int32_t>(problem.numHeads / problem.numVHeads),
                    sink,
                    sinkStride,
                    seqLenQ,
                    seqLenKv,
                    rowOffQ,
                    rowOffKv,
                    dropout.seed,
                    dropout.offset1,
                    dropout.offset2,
                    dropout.threshold,
                    dropout.scale);
        // The keep mask the launch above applied, on its stream after it.
        if(preparedSdpa.dropoutMask().has_value())
        {
            preparedSdpa.dropoutMask()->launch(handle.getStream(),
                                               deviceBuffers,
                                               numDeviceBuffers,
                                               dropout.seed,
                                               dropout.offset1,
                                               dropout.offset2,
                                               dropout.threshold);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaRdnaKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too -- including the
    // RMS-norm pack, which reads the same archive.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaRdnaModuleCache()
{
    flydslSdpaRdnaKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle.
const FlydslSdpaRdnaDispatchHandler& flydslSdpaRdnaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaRdnaKpackModuleCache());
    static const FlydslSdpaRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaRdnaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydsl_sdpa_rdna::flydslSdpaRdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaRdnaKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaRdnaScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaRdnaDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
