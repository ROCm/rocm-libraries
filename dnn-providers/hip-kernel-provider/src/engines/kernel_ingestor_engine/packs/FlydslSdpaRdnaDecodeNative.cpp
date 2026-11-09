// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
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
 * @file FlydslSdpaRdnaDecodeNative.cpp
 * @brief The flyDSL RDNA SDPA decode pack's native half: kernel matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * Its objects are compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention_rdna/flash_attn_decode_rdna.py`: the GQA group's
 * query heads and the query positions packed into 16-row tiles, the keys split across
 * workgroups, and a second kernel in the same object that merges the splits. A different
 * algorithm from the prefill kernel's, with its own argument lists, launch and workspace,
 * so it is an engine of its own (`hipkernel:flydsl_sdpa_rdna_decode`) rather than a knob of
 * the prefill one's. The two share the graph matcher and the checks in
 * FlydslSdpaRdnaCommon.hpp. This matcher declines only what the kernel cannot compute;
 * which of the two engines serves a graph both can is the heuristic's to decide.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_rdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna_decode.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna_decode.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_rdna_decode.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_rdna_decode.dispatch";

/// The decode object's second kernel, which combines its KV splits. Same code object, so
/// it is loaded from the same archive entry by this symbol.
constexpr std::string_view MERGE_SYMBOL_FIELD = "merge_symbol";

/// The decode kernel packs the GQA group's query heads and the query positions into 16-row
/// tiles, one workgroup per tile along the grid's second dimension, as many tiles as fit it.
constexpr int64_t DECODE_ROWS = 16;
constexpr int64_t DECODE_MAX_ROW_TILES = 65535;
/// Waves per decode workgroup, keys per wave per step, and threads per merge workgroup:
/// the geometry the decode kernel and its merge were built with.
constexpr int64_t DECODE_WAVES = 4;
constexpr int64_t DECODE_TILE = 32;
constexpr int64_t DECODE_MERGE_THREADS = 64;
/// f32 statistics per split row of the workspace: the row's max (natural-log units) and its
/// sum, kept apart so the merge weighs a split by its sum even where max + ln(sum) would
/// round to the max (a lowest-finite bias on every key).
constexpr int64_t DECODE_SPLIT_STATS = 2;
/// Split-count bounds: a split should give each wave at least this many tiles, the splits
/// together should roughly fill the device, and never more than this many.
constexpr int64_t DECODE_MIN_TILES_PER_WAVE = 8;
constexpr int64_t DECODE_MAX_SPLITS = 64;
/// Compute units assumed when the device does not report its count.
constexpr int64_t DECODE_FALLBACK_COMPUTE_UNITS = 32;
/// Decode workgroups to aim for per reported unit. RDNA reports workgroup processors --
/// two CUs each -- in multiProcessorCount, so one workgroup per unit leaves half the
/// device idle at small batch; two fill it, and more splits than that cost more in the
/// merge and the shorter KV runs than they recover (measured on gfx1151).
constexpr int64_t DECODE_WORKGROUPS_PER_UNIT = 2;

/// A decode object's launch tile: its column split, and the 16-row tile it was built for.
std::optional<int64_t> decodeColumnSplit(const KernelDefinition& kernel)
{
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto dvSplit = integerMetadata(kernel, DV_SPLIT_FIELD);
    if(blockM != DECODE_ROWS || !dvSplit.has_value() || *dvSplit <= 0)
    {
        return std::nullopt;
    }
    return dvSplit;
}

/// Whether a decode object takes several row tiles. The bias objects are built for one:
/// they already use the whole register file at head_dim 128 and up, and the row index a
/// tiled launch holds through the KV loop spills there.
bool decodeRowTiled(const KernelDefinition& kernel)
{
    return integerMetadata(kernel, HAS_BIAS_FIELD) == 0;
}

/// The decode launch's row tiles: (Hq / Hk) * Sq rows, 16 to a tile, or one.
int64_t decodeRowTiles(const SdpaProblem& problem, bool rowTiled)
{
    return rowTiled ? ((problem.numHeads / problem.numKvHeads) * problem.seqLenQ + DECODE_ROWS - 1)
                          / DECODE_ROWS
                    : 1;
}

/// How many KV splits a decode launch uses: enough workgroups to roughly fill the device
/// (DECODE_WORKGROUPS_PER_UNIT per reported unit), but no split so short that its waves have little to stream, and at most
/// DECODE_MAX_SPLITS. Deterministic in the problem and the device, so workspaceBytes and
/// prepare agree.
int64_t decodeSplits(const SdpaProblem& problem, int64_t rowTiles, int computeUnits)
{
    const int64_t units = (computeUnits > 0 ? computeUnits : DECODE_FALLBACK_COMPUTE_UNITS)
                          * DECODE_WORKGROUPS_PER_UNIT;
    const int64_t groups = std::max<int64_t>(1, problem.batch * problem.numKvHeads * rowTiles);
    const int64_t tiles = (problem.seqLenKv + DECODE_TILE - 1) / DECODE_TILE;
    const int64_t byWork = std::max<int64_t>(1, tiles / (DECODE_WAVES * DECODE_MIN_TILES_PER_WAVE));
    const int64_t byDevice = std::max<int64_t>(1, units / groups);
    return std::min({byWork, byDevice, DECODE_MAX_SPLITS});
}

/// The split workspace: a normalized f32 partial O and its row's max and sum per (batch,
/// query head, query position, split), or nothing when one split writes O directly.
std::optional<size_t> decodeWorkspaceBytes(const SdpaProblem& problem, int64_t splits)
{
    if(splits <= 1)
    {
        return size_t{0};
    }
    const auto elements = checkedProduct({problem.batch,
                                          problem.numHeads,
                                          problem.seqLenQ,
                                          splits,
                                          problem.headDimV + DECODE_SPLIT_STATS});
    if(!elements.has_value())
    {
        return std::nullopt;
    }
    return static_cast<size_t>(*elements) * sizeof(float);
}

/// Whether a decode object's 32-bit bias offsets reach every row of its tile: they span the
/// GQA group's heads as well as the query positions and keys of one (batch, head) slice.
bool decodeBiasFits(const SdpaProblem& problem, const BiasStrides& strides)
{
    const auto lastHead = checkedProduct({problem.numHeads / problem.numKvHeads - 1, strides[1]});
    const auto lastQuery = checkedProduct({problem.seqLenQ - 1, strides[2]});
    const auto lastKey = checkedProduct({problem.seqLenKv - 1 + KV_TILE_OVERREAD, strides[3]});
    return lastHead.has_value() && lastQuery.has_value() && lastKey.has_value()
           && *lastHead <= BIAS_SLICE_ELEMENT_LIMIT - *lastKey
           && *lastQuery <= BIAS_SLICE_ELEMENT_LIMIT - *lastKey - *lastHead;
}

/// One past the last element @p tensor's dims and strides reach, or nullopt past int64.
std::optional<int64_t> wholeTensorElements(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size())
    {
        return std::nullopt;
    }
    int64_t last = 1;
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        const auto reach = checkedProduct({dims->Get(axis) - 1, strides->Get(axis)});
        if(!reach.has_value() || *reach > INT32_LIMIT)
        {
            return std::nullopt;
        }
        last += *reach;
    }
    return last;
}

/// Whether the merge kernel's 32-bit element offsets reach everything it addresses: it
/// indexes the split workspace by (row, split, column) and O and the LSE through their
/// strides over the whole tensor, not one (batch, head) slice.
bool decodeMergeReaches(const MatchContext& context,
                        const BoundTokens& bound,
                        const SdpaProblem& problem,
                        int64_t splits)
{
    const auto workspace = checkedProduct({problem.batch,
                                           problem.numHeads,
                                           problem.seqLenQ,
                                           splits,
                                           problem.headDimV + DECODE_SPLIT_STATS});
    if(!workspace.has_value() || *workspace > INT32_LIMIT)
    {
        return false;
    }
    for(const auto token : {O_TOKEN, STATS_TOKEN})
    {
        const auto uid = tryGetBoundInt(bound, token);
        if(!uid.has_value() || (token == STATS_TOKEN && *uid == NO_STATS))
        {
            continue;
        }
        const auto* tensor = findTensor(context, *uid);
        const auto elements = tensor == nullptr ? std::nullopt : wholeTensorElements(*tensor);
        if(!elements.has_value() || *elements > INT32_LIMIT)
        {
            return false;
        }
    }
    return true;
}

/// Whether a decode object can compute the graph: one V head goes with each K head (the
/// rows of a tile share one KV head), and its rows fit -- any number of 16-row tiles the
/// grid's second dimension holds for a row-tiled object, one tile's for an object built
/// for one.
bool decodeServes(const SdpaProblem& problem, bool rowTiled)
{
    if(problem.numKvHeads != problem.numVHeads)
    {
        return false;
    }
    // The merge kernel runs one workgroup per (batch, query head, query position): a grid
    // past int32 would be refused at launch, so it is declined here instead.
    const auto mergeGrid = checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
    if(!mergeGrid.has_value() || *mergeGrid > INT32_LIMIT)
    {
        return false;
    }
    return rowTiled ? decodeRowTiles(problem, rowTiled) <= DECODE_MAX_ROW_TILES
                    : (problem.numHeads / problem.numKvHeads) * problem.seqLenQ <= DECODE_ROWS;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
 *
 * The family's shared checks (FlydslSdpaRdnaCommon.hpp), then the decode object's own: its
 * rows fit its launch, it names the merge kernel it launches, its column split divides its
 * head, and a bias's 32-bit offsets reach every row of its tile.
 */
bool flydslSdpaRdnaDecodeKernelMatches(const MatchContext& context,
                                       const BoundTokens& bound,
                                       const KernelDefinition& kernel)
{
    SdpaProblem problem;
    if(!familyMatches(context, bound, kernel, problem))
    {
        return false;
    }
    const auto dvSplit = decodeColumnSplit(kernel);
    const auto headDimMax = integerMetadata(kernel, HEAD_DIM_MAX_FIELD);
    if(!dvSplit.has_value() || !headDimMax.has_value() || *headDimMax % *dvSplit != 0
       || !decodeServes(problem, decodeRowTiled(kernel))
       || stringMetadata(kernel, MERGE_SYMBOL_FIELD).value_or("").empty())
    {
        return false;
    }
    const auto splits = decodeSplits(problem,
                                     decodeRowTiles(problem, decodeRowTiled(kernel)),
                                     context.deviceProperties.multiProcessorCount);
    if(!decodeMergeReaches(context, bound, problem, splits))
    {
        return false;
    }
    const auto bias = tryGetBoundInt(bound, BIAS_TOKEN);
    if(bias.has_value() && *bias != NO_BIAS)
    {
        const auto* biasTensor = findTensor(context, *bias);
        const auto strides = biasTensor == nullptr ? std::nullopt
                                                   : servableBiasStrides(*biasTensor,
                                                                         problem.batch,
                                                                         problem.numHeads,
                                                                         problem.seqLenQ,
                                                                         problem.seqLenKv,
                                                                         problem.dataType);
        if(!strides.has_value() || !decodeBiasFits(problem, *strides))
        {
            return false;
        }
    }
    return true;
}

/// Ranks candidates that survived kernelMatches by their descriptor `priority`: the
/// specialized objects above the generic ones, the narrower generic one above the wider.
double flydslSdpaRdnaDecodeScore(const MatchContext& /*context*/,
                                 const BoundTokens& /*bound*/,
                                 const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The decode object's main kernel's argument list (flash_attn_decode_rdna.py).
 *
 * Q, K, V, O, LSE and the split workspace's O and (max, sum) pairs (pointers);
 * seq_len_q, seq_len_kv, num_heads, kv_group, right_bound, left_bound,
 * align_bottom_right, lse_on, num_splits (i32); scale (f32); the (batch, sequence, head) strides of Q, K, V, O and LSE (i64);
 * the bias (pointer) and its (batch, head, query, key) strides (i64), read only by a
 * has_bias object; the head_dim (i32), read only by a generic object; the sink (pointer)
 * and its head stride (i64), read only by a has_sink object.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaDecodeSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_i32,
        s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,
        s_f32,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,
        s_pointer, s_i64,     s_i64,     s_i64,     s_i64,     s_i32,     s_pointer, s_i64};
    return s_signature;
}

/**
 * @brief The decode object's merge kernel's argument list.
 *
 * O, LSE and the split workspace's O and (max, sum) pairs (pointers); seq_len_q, num_heads, lse_on,
 * num_splits (i32); the (batch, sequence, head) strides of O and LSE (i64); the
 * head_dim (i32), the width of a generic object's workspace and O rows; the sink (pointer)
 * and its head stride (i64), read only by a has_sink object.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaMergeSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_pointer,
                                                         s_pointer,
                                                         s_pointer,
                                                         s_pointer,
                                                         s_i32,
                                                         s_i32,
                                                         s_i32,
                                                         s_i32,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i32,
                                                         s_pointer,
                                                         s_i64};
    return s_signature;
}

/// The compiled main and merge kernels plus everything launch() needs, owning nothing that
/// points back into the MatchContext or BoundTokens it came from.
class PreparedFlydslSdpaRdnaDecode : public PreparedDispatch
{
public:
    PreparedFlydslSdpaRdnaDecode(IngestorKernelCode code,
                                 IngestorKernelCode mergeCode,
                                 FlydslSdpaRdnaBinding binding,
                                 SdpaProblem problem,
                                 std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                                 std::array<OperandStrides, 5> strides,
                                 BiasStrides biasStrides,
                                 int64_t splits)
        : _code(std::move(code))
        , _mergeCode(std::move(mergeCode))
        , _binding(binding)
        , _problem(problem)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
        , _biasStrides(biasStrides)
        , _splits(splits)
    {
    }

    compilation::IRunnableKernel& mergeForStream(hipStream_t stream) const
    {
        return _mergeCode.kernelForStream(stream);
    }

    /// The decode launch's KV splits; 1 writes O directly and skips the merge.
    int64_t splits() const
    {
        return _splits;
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

private:
    IngestorKernelCode _code;
    IngestorKernelCode _mergeCode;
    FlydslSdpaRdnaBinding _binding;
    SdpaProblem _problem;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    std::array<OperandStrides, 5> _strides;
    BiasStrides _biasStrides;
    int64_t _splits = 1;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslSdpaRdnaDecodeDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime. Unused in
    /// practice -- every kernel this pack selects comes from a kpack archive -- but
    /// buildIngestorKernelCode decides that from the kernel's source kind, not from here.
    /// @param kpackLoader Same must-outlive contract.
    FlydslSdpaRdnaDecodeDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                        const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// A launch with more than one KV split needs a partial O and LSE per split.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaRdnaBinding(bound);
        auto problem
            = problemFor(requireTensor(context, binding.q), requireTensor(context, binding.k));
        problem.headDimV = requireTensor(context, binding.v).dims()->Get(HEAD_DIM_AXIS);
        const auto bytes
            = decodeWorkspaceBytes(problem,
                                   decodeSplits(problem,
                                                decodeRowTiles(problem, decodeRowTiled(kernel)),
                                                context.deviceProperties.multiProcessorCount));
        if(!bytes.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "flydsl_sdpa_rdna_decode: workspace overflows");
        }
        return *bytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaRdnaBinding(bound);

        // kernel_match declines a candidate without a usable column split, so this is
        // defence in depth.
        const auto dvSplit = decodeColumnSplit(kernel);
        if(!dvSplit.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna_decode: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m/dv_split");
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
                                            flydslSdpaRdnaDecodeSignature());

        // `batch_x_kvheads_x_splits_x_dv_split_by_row_tiles`: one workgroup of
        // DECODE_WAVES waves per (batch, KV head, split, output-column tile), by 16-row
        // tile along y; the merge, one workgroup per (batch, query head, position).
        const auto rowTiles = decodeRowTiles(problem, decodeRowTiled(kernel));
        const int64_t splits
            = decodeSplits(problem, rowTiles, context.deviceProperties.multiProcessorCount);
        const auto mergeSymbol = stringMetadata(kernel, MERGE_SYMBOL_FIELD);
        const auto mainGrid = checkedProduct({problem.batch, problem.numKvHeads, splits, *dvSplit});
        const auto mergeGrid = checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
        if(!mergeSymbol.has_value() || mergeSymbol->empty() || !mainGrid.has_value()
           || *mainGrid > INT32_LIMIT || !mergeGrid.has_value() || *mergeGrid > INT32_LIMIT)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna_decode: kernel '" + toString(kernel.kernelId)
                    + "' names no merge kernel, or its launch overflows");
        }
        // The merge kernel is the same code object under another symbol, so the same
        // archive entry with that symbol and its own argument list.
        KernelDefinition mergeKernel = kernel;
        mergeKernel.source.symbol = *mergeSymbol;
        mergeKernel.source.signature = flydslSdpaRdnaMergeSignature();
        auto mergeCode = buildIngestorKernelCode(_kernelCompiler,
                                                 _kpackLoader,
                                                 context,
                                                 mergeKernel,
                                                 options,
                                                 flydslSdpaRdnaMergeSignature());
        code.setBlockSize(static_cast<unsigned int>(DECODE_WAVES * WAVE_SIZE), 1, 1);
        code.setGridSize(
            static_cast<unsigned int>(*mainGrid), static_cast<unsigned int>(rowTiles), 1);
        mergeCode.setBlockSize(static_cast<unsigned int>(DECODE_MERGE_THREADS), 1, 1);
        mergeCode.setGridSize(static_cast<unsigned int>(*mergeGrid), 1, 1);

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
                    "flydsl_sdpa_rdna_decode: the bound bias is not one the kernel can read");
            }
            biasStrides = *strides;
        }

        return std::make_unique<PreparedFlydslSdpaRdnaDecode>(
            std::move(code),
            std::move(mergeCode),
            binding,
            problem,
            scaleOperand,
            std::array<OperandStrides, 5>{stridesOf(q),
                                          stridesOf(k),
                                          stridesOf(v),
                                          stridesOf(o),
                                          binding.stats == NO_STATS
                                              ? OperandStrides{}
                                              : stridesOf(requireTensor(context, binding.stats))},
            biasStrides,
            splits);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaRdnaDecode&>(prepared);
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
        // Each optional buffer is read only by the objects kernel_match pairs with a bound
        // one: the LSE when lse_on is set, the bias by has_bias, the sink by has_sink.
        void* stats = optionalBuffer(binding.stats, deviceBuffers, numDeviceBuffers);
        void* bias = optionalBuffer(binding.bias, deviceBuffers, numDeviceBuffers);
        void* sink = optionalBuffer(binding.sink, deviceBuffers, numDeviceBuffers);
        const int64_t sinkStride = binding.sinkStride;

        const auto& s = preparedSdpa.strides();
        const auto& b = preparedSdpa.biasStrides();

        const auto splits = preparedSdpa.splits();
        // The workspace is the partial O (f32, head_dim per row and split) followed by
        // each split row's (max, sum) pair, DECODE_SPLIT_STATS f32s; unused with one split.
        void* wsO = splits > 1 ? workspace : nullptr;
        void* wsMaxSum
            = splits > 1 ? static_cast<void*>(static_cast<char*>(workspace)
                                              + static_cast<size_t>(problem.batch * problem.numHeads
                                                                    * problem.seqLenQ * splits
                                                                    * problem.headDimV)
                                                    * sizeof(float))
                         : nullptr;
        const auto lseOn = static_cast<int32_t>(binding.stats == NO_STATS ? 0 : 1);
        // Changing these argument lists means changing flydslSdpaRdnaDecodeSignature() and
        // flydslSdpaRdnaMergeSignature() with them.
        preparedSdpa.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    stats,
                    wsO,
                    wsMaxSum,
                    static_cast<int32_t>(problem.seqLenQ),
                    static_cast<int32_t>(problem.seqLenKv),
                    static_cast<int32_t>(problem.numHeads),
                    static_cast<int32_t>(problem.numHeads / problem.numKvHeads),
                    static_cast<int32_t>(binding.rightBound),
                    static_cast<int32_t>(binding.leftBound),
                    static_cast<int32_t>(binding.alignBottomRight),
                    lseOn,
                    static_cast<int32_t>(splits),
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
                    sink,
                    sinkStride);
        if(splits > 1)
        {
            preparedSdpa.mergeForStream(handle.getStream())
                .launch(handle.getStream(),
                        o.ptr,
                        stats,
                        wsO,
                        wsMaxSum,
                        static_cast<int32_t>(problem.seqLenQ),
                        static_cast<int32_t>(problem.numHeads),
                        lseOn,
                        static_cast<int32_t>(splits),
                        s[3].batch,
                        s[3].sequence,
                        s[3].head,
                        s[4].batch,
                        s[4].sequence,
                        s[4].head,
                        static_cast<int32_t>(problem.headDim),
                        sink,
                        sinkStride);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaRdnaDecodeKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaRdnaDecodeModuleCache()
{
    flydslSdpaRdnaDecodeKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle.
const FlydslSdpaRdnaDecodeDispatchHandler& flydslSdpaRdnaDecodeDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        flydslSdpaRdnaDecodeKpackModuleCache());
    static const FlydslSdpaRdnaDecodeDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaRdnaDecodeSymbols(SymbolScope<Handle>& scope)
{
    // The graph matcher is the prefill engine's too: both serve the same graphs.
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydsl_sdpa_rdna::flydslSdpaRdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaRdnaDecodeKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaRdnaDecodeScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaRdnaDecodeDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
