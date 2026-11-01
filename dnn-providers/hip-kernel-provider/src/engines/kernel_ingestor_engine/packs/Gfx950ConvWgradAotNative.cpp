// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950ConvWgradGeometry.hpp"

/**
 * @file Gfx950ConvWgradAotNative.cpp
 * @brief The hipkernel:Gfx950ConvWgradAot engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * Shape-generic AOT instances of rocKE's backward-weights (wgrad) convolution kernels:
 * the single-launch implicit-GEMM family (kernel_family 0, split_k 1 or atomic) and the
 * direct family (kernel_family 1). Every two-stage (split-K workspace-reduce) instance is
 * the sibling hipkernel:Gfx950ConvWgradTwoStageAot instead (Gfx950ConvWgradTwoStageAotNative.cpp):
 * the descriptor loader requires one consistent launch set per engine, and two-stage
 * instances are two launches. Mirrors rocke/library/builders/common/convolution_backward_filter.py.
 *
 * Direct-family dispatch (directPlan/prepareDirect/launchDirect): the two non-folded
 * variants (direct_wgrad, direct_wgrad_tiled) atomic-add into dW directly, like implicit-
 * GEMM's split-K path. The other five write one partial slot per partition into a
 * hipDNN-workspace scratch and need their matching direct_wgrad_fold kernel (kernel_family
 * 2) to sum it into dW -- found via foldRegistry(), filed as a side effect of kernelMatches
 * seeing each one (registerFoldIfSeen; this SDK has no "runs once per kernel at load time"
 * prepare hook the way hipkernel:Gfx950ConvDgradAot's weight-transform helpers use on their
 * branch), which is always complete well before prepareDirect ever needs it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace wgrad = gfx950_conv_wgrad;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_conv_wgrad_aot.dispatch";
constexpr std::string_view X_TOKEN = "gfx950_conv_wgrad.x.uid";
constexpr std::string_view DY_TOKEN = "gfx950_conv_wgrad.dy.uid";
constexpr std::string_view DW_TOKEN = "gfx950_conv_wgrad.dw.uid";
/// The whole matched problem, so a kernel matcher reads it from what graph_match bound
/// instead of walking the graph again for every kernel.
constexpr std::string_view PROBLEM_TOKEN = "_gfx950_conv_wgrad.problem";
constexpr size_t PROBLEM_TOKEN_SIZE = 18;

/// KMD fields this engine reads. Every field convolution_backward_filter.py's
/// Gfx950ConvWgradAotSpec carries; a direct/helper instance holds the implicit-GEMM
/// fields at their KMD default, and vice versa (same convention as the Python adapter).
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view KERNEL_FAMILY_FIELD = "kernel_family";
constexpr std::string_view TILE_M_FIELD = "tile_m";
constexpr std::string_view TILE_N_FIELD = "tile_n";
constexpr std::string_view TILE_K_FIELD = "tile_k";
constexpr std::string_view SPLIT_K_FIELD = "split_k";
constexpr std::string_view TWO_STAGE_FLAG_FIELD = "two_stage_flag";
constexpr std::string_view WARP_M_FIELD = "warp_m";
constexpr std::string_view WARP_N_FIELD = "warp_n";
constexpr std::string_view WAVE_SIZE_FIELD = "wave_size";
constexpr std::string_view VECTOR_SIZE_A_FIELD = "vector_size_a";
constexpr std::string_view VECTOR_SIZE_B_FIELD = "vector_size_b";

/// Direct/helper-only KMD fields (convolution_backward_filter.py's
/// _DIRECT_CAPABILITY_FIELDS plus the per-variant AOT_WGRAD_DIRECT_KNOBS this native pack
/// actually needs for launch geometry -- knobs that tune the compiled binary's internal
/// scheduling without changing grid/block/kernarg shape, e.g. row_unroll, pingpong, are not
/// read here).
constexpr std::string_view DIRECT_VARIANT_FIELD = "direct_variant";
constexpr std::string_view FILTER_H_FIELD = "filter_h";
constexpr std::string_view FILTER_W_FIELD = "filter_w";
constexpr std::string_view STRIDE_H_FIELD = "stride_h";
constexpr std::string_view STRIDE_W_FIELD = "stride_w";
constexpr std::string_view PAD_H_FIELD = "pad_h";
constexpr std::string_view PAD_W_FIELD = "pad_w";
constexpr std::string_view DILATION_H_FIELD = "dilation_h";
constexpr std::string_view DILATION_W_FIELD = "dilation_w";
constexpr std::string_view CPG_FIELD = "cpg";
constexpr std::string_view KPG_FIELD = "kpg";
constexpr std::string_view TAPS_FIELD = "taps";
constexpr std::string_view WAVES_C_FIELD = "waves_c";
constexpr std::string_view WAVES_K_FIELD = "waves_k";
constexpr std::string_view WAVES_Q_FIELD = "waves_q";
constexpr std::string_view HO_PER_BLOCK_FIELD = "ho_per_block";
constexpr std::string_view MFMA_K_FIELD = "mfma_k";
constexpr std::string_view WAVE_K16_FIELD = "wave_k16";
constexpr std::string_view WAVE_C16_FIELD = "wave_c16";
constexpr std::string_view WAVE_Q4_FIELD = "wave_q4";
constexpr std::string_view Q_GROUPS_FIELD = "q_groups";
constexpr std::string_view UNFOLD_N_FIELD = "unfold_n";
constexpr std::string_view ROWS_PER_SEG_FIELD = "rows_per_seg";
constexpr std::string_view ROWS_PER_CHUNK_FIELD = "rows_per_chunk";
constexpr std::string_view TARGET_BLOCKS_FIELD = "target_blocks";
constexpr std::string_view LANES_C_FIELD = "lanes_c";
constexpr std::string_view LANES_W_FIELD = "lanes_w";
constexpr std::string_view COLS_PER_LANE_FIELD = "cols_per_lane";
constexpr std::string_view PREFETCH_FIELD = "prefetch";
constexpr std::string_view STRIP_FIELD = "strip";
constexpr std::string_view THREADS_FIELD = "threads";
constexpr std::string_view Q_TILES_FIELD = "q_tiles";

constexpr std::string_view DIRECT_VARIANT_WGRAD = "direct_wgrad";
constexpr std::string_view DIRECT_VARIANT_WGRAD_TILED = "direct_wgrad_tiled";
constexpr std::string_view DIRECT_VARIANT_DW = "direct_wgrad_dw";
constexpr std::string_view DIRECT_VARIANT_DW_COLS = "direct_wgrad_dw_cols";
constexpr std::string_view DIRECT_VARIANT_DW_MFMA = "direct_wgrad_dw_mfma";
constexpr std::string_view DIRECT_VARIANT_GROUPED_MFMA = "direct_wgrad_grouped_mfma";
constexpr std::string_view DIRECT_VARIANT_GATHER = "direct_wgrad_gather";
constexpr std::string_view DIRECT_VARIANT_FOLD = "direct_wgrad_fold";
constexpr std::string_view DIRECT_VARIANT_CAST = "direct_wgrad_cast";

constexpr int64_t KERNEL_FAMILY_IMPLICIT_GEMM = 0;
constexpr int64_t KERNEL_FAMILY_DIRECT = 1;
constexpr int64_t KERNEL_FAMILY_WGRAD_HELPER = 2;

struct Binding
{
    int64_t x;
    int64_t dy;
    int64_t dw;
};

struct MatchedProblem
{
    Binding binding;
    wgrad::Problem problem;
    wgrad::Geometry geometry;
    data_objects::DataType dtype;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

bool isDeviceTensor(const data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr && !tensor->virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           && !tensor->ragged_offset_tensor_uid().has_value() && tensor->alignment() >= 16
           && tensor->alignment() % 16 == 0
           && (tensor->data_type() == data_objects::DataType::HALF
               || tensor->data_type() == data_objects::DataType::BFLOAT16);
}

/// hipDNN [N, C, H, W] dims stored NHWC: X [N, C, Hi, Wi] as NHWC, dY [N, K, Ho, Wo] as
/// NHWK, dW [K, C/G, Y, X] as KYXC.
bool hasPackedNhwcStrides(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != 4 || tensor->strides()->size() != 4)
    {
        return false;
    }
    const auto* dims = tensor->dims();
    const std::array<int64_t, 4> extents{dims->Get(0), dims->Get(1), dims->Get(2), dims->Get(3)};
    if(std::any_of(extents.begin(), extents.end(), [](int64_t e) { return e <= 0; })
       || !wgrad::wideProduct({extents[0], extents[1], extents[2], extents[3], 2}))
    {
        return false;
    }
    const std::array<int64_t, 4> expected{
        extents[1] * extents[2] * extents[3], 1, extents[1] * extents[3], extents[1]};
    for(flatbuffers::uoffset_t axis = 0; axis < 4; ++axis)
    {
        if(extents[axis] != 1 && tensor->strides()->Get(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

bool hasSpatialRankTwo(const flatbuffers::Vector<int64_t>* values)
{
    return values != nullptr && values->size() == 2;
}

/// The graph contract: one ConvolutionWrwAttributes node on gfx950 (wave64) with FLOAT
/// accumulation, cross-correlation, 2-D symmetric padding, three distinct device tensors of
/// one 16-bit dtype in packed NHWC, groups = X.C / dW.dims[1] in [1, 65535] dividing K, and
/// dY's dims equal to the forward output of X (whose dims and dW's Y/X define the problem).
std::optional<MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!context.graph.isValid() || context.graph.getGraph().is_override_shape_enabled()
       || !hipdnn_plugin_sdk::archMatches(
           context.deviceProperties.gcnArchName, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX)
       || context.deviceProperties.warpSize != 64 || context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionWrwAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto& attributes = node.attributesAs<data_objects::ConvolutionWrwAttributes>();
    if(attributes.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION
       || !hasSpatialRankTwo(attributes.pre_padding())
       || !hasSpatialRankTwo(attributes.post_padding()) || !hasSpatialRankTwo(attributes.stride())
       || !hasSpatialRankTwo(attributes.dilation()))
    {
        return std::nullopt;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < 2; ++axis)
    {
        if(attributes.pre_padding()->Get(axis) != attributes.post_padding()->Get(axis))
        {
            return std::nullopt;
        }
    }

    const Binding binding{
        attributes.x_tensor_uid(), attributes.dy_tensor_uid(), attributes.dw_tensor_uid()};
    const auto* x = findTensor(context, binding.x);
    const auto* dy = findTensor(context, binding.dy);
    const auto* dw = findTensor(context, binding.dw);
    if(binding.x == binding.dy || binding.x == binding.dw || binding.dy == binding.dw
       || !isDeviceTensor(x) || !isDeviceTensor(dy) || !isDeviceTensor(dw)
       || !hasPackedNhwcStrides(x) || !hasPackedNhwcStrides(dy) || !hasPackedNhwcStrides(dw)
       || x->data_type() != dy->data_type() || x->data_type() != dw->data_type())
    {
        return std::nullopt;
    }
    const auto* xDims = x->dims();
    const auto* dwDims = dw->dims();
    const auto* dyDims = dy->dims();
    // hipDNN carries no groups attribute: dW holding C / groups channels encodes it. The
    // layout checks above already made every extent positive.
    const auto channelsPerGroup = dwDims->Get(1);
    if(xDims->Get(1) % channelsPerGroup != 0)
    {
        return std::nullopt;
    }
    const auto groups = xDims->Get(1) / channelsPerGroup;
    const wgrad::Problem problem{xDims->Get(0),
                                 xDims->Get(1),
                                 dwDims->Get(0),
                                 xDims->Get(2),
                                 xDims->Get(3),
                                 dwDims->Get(2),
                                 dwDims->Get(3),
                                 attributes.stride()->Get(0),
                                 attributes.stride()->Get(1),
                                 attributes.pre_padding()->Get(0),
                                 attributes.pre_padding()->Get(1),
                                 attributes.dilation()->Get(0),
                                 attributes.dilation()->Get(1),
                                 groups};
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry || dyDims->Get(0) != problem.n || dyDims->Get(1) != problem.k
       || dyDims->Get(2) != geometry->ho || dyDims->Get(3) != geometry->wo)
    {
        return std::nullopt;
    }
    return MatchedProblem{binding, problem, *geometry, x->data_type()};
}

std::vector<int64_t> problemTokenValues(const MatchedProblem& matched)
{
    const auto& p = matched.problem;
    return {matched.binding.x,
            matched.binding.dy,
            matched.binding.dw,
            p.n,
            p.c,
            p.k,
            p.hi,
            p.wi,
            p.y,
            p.x,
            p.strideH,
            p.strideW,
            p.padH,
            p.padW,
            p.dilationH,
            p.dilationW,
            p.groups,
            static_cast<int64_t>(matched.dtype)};
}

std::optional<MatchedProblem> matchedFromBound(const MatchContext& context,
                                               const BoundTokens& bound)
{
    static const std::string s_problemToken(PROBLEM_TOKEN);
    const auto it = bound.find(s_problemToken);
    const auto* values
        = it == bound.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
    if(values == nullptr || values->size() != PROBLEM_TOKEN_SIZE)
    {
        return matchProblem(context);
    }
    const auto& v = *values;
    const wgrad::Problem problem{
        v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16]};
    const auto geometry = wgrad::deriveGeometry(problem);
    if(!geometry)
    {
        return std::nullopt;
    }
    return MatchedProblem{
        Binding{v[0], v[1], v[2]}, problem, *geometry, static_cast<data_objects::DataType>(v[17])};
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchProblem(context);
    if(!matched)
    {
        return std::nullopt;
    }
    return BoundTokens{{std::string(X_TOKEN), matched->binding.x},
                       {std::string(DY_TOKEN), matched->binding.dy},
                       {std::string(DW_TOKEN), matched->binding.dw},
                       {std::string(PROBLEM_TOKEN), problemTokenValues(*matched)}};
}

/// Each pack's "operation is <shard>" criterion: every shard (dtype x pipeline/direct)
/// serves the same graph shape, and pipeline/direct is a kernel-level choice kernelMatches
/// and scoreKernel already make -- so the only thing distinguishing a shard at the GRAPH
/// level is its dtype.
template <data_objects::DataType Dtype>
bool shardDtypeCriterion(const MatchContext& context, const BoundTokens& bound)
{
    const auto matched = matchedFromBound(context, bound);
    return matched.has_value() && matched->dtype == Dtype;
}

/// A kernel's metadata, as the fields above read it.
struct KernelMetadata
{
    const KernelDefinition& kernel;

    std::optional<int64_t> intField(std::string_view field) const
    {
        const auto it = kernel.metadata.find(std::string(field));
        const auto* value
            = it == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&it->second);
        return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
    }

    const std::string* stringField(std::string_view field) const
    {
        const auto it = kernel.metadata.find(std::string(field));
        return it == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&it->second);
    }
};

const char* dtypeName(data_objects::DataType dtype)
{
    return dtype == data_objects::DataType::HALF ? "fp16" : "bf16";
}

/// This kernel's implicit-GEMM single-stage kernargs on @p matched, or nullopt when its
/// baked knobs (tile sizes, group_merge, split_k) do not admit the problem. A kernel
/// serves a graph exactly when this (or the direct-family check, currently always
/// declined) succeeds -- matching is tied to a successful launch-plan derivation, not a
/// restated copy of it, the same discipline hipkernel:Gfx950ConvDgradAot follows.
std::optional<wgrad::ImplicitGemmKernargs> implicitGemmPlan(const MatchedProblem& matched,
                                                            const KernelDefinition& kernel)
{
    const KernelMetadata md{kernel};
    const auto* dtype = md.stringField(DTYPE_FIELD);
    if(dtype == nullptr || *dtype != dtypeName(matched.dtype))
    {
        return std::nullopt;
    }
    const auto family = md.intField(KERNEL_FAMILY_FIELD).value_or(KERNEL_FAMILY_IMPLICIT_GEMM);
    if(family != KERNEL_FAMILY_IMPLICIT_GEMM)
    {
        return std::nullopt;
    }
    if(md.intField(TWO_STAGE_FLAG_FIELD).value_or(0) != 0)
    {
        // A two-stage instance belongs to the sibling engine's catalog, never this one's.
        return std::nullopt;
    }
    const auto tileM = md.intField(TILE_M_FIELD);
    const auto tileN = md.intField(TILE_N_FIELD);
    const auto tileK = md.intField(TILE_K_FIELD);
    if(!tileM || !tileN || !tileK)
    {
        return std::nullopt;
    }
    // conv_implicit_gemm_wgrad.py picks the dY/X free-axis global-load vector width at
    // BUILD time (CoalescedTileLoader.choose_vec against the *probe's* kpg/cpg, capped at
    // this KMD knob) -- it is baked instruction selection, not a kernarg, so it is only
    // safe for a runtime problem whose per-group kpg/cpg are themselves multiples of it.
    // 0 means "derived by the builder from the group_merge run" (the Gm-merged depthwise
    // family), which is safe by construction and skips this check. Without this guard a
    // v8-wide grouped/split-K instance built against a probe cpg=32 was dispatched onto a
    // real groups=1024 (cpg=1) graph and the 8-wide vector load ran past the 1-wide real
    // channel run: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, not a wrong-answer failure.
    const auto vectorSizeA = md.intField(VECTOR_SIZE_A_FIELD).value_or(0);
    const auto vectorSizeB = md.intField(VECTOR_SIZE_B_FIELD).value_or(0);
    if((vectorSizeA > 0 && matched.geometry.kpg % vectorSizeA != 0)
       || (vectorSizeB > 0 && matched.geometry.cpg % vectorSizeB != 0))
    {
        return std::nullopt;
    }
    // SPLIT_K_FIELD is a BUILD-time capability flag here, not the launch degree: KMD
    // split_k > 1 says this binary was built with the atomic cshuffle epilogue at all
    // (conv_implicit_gemm_wgrad.py only emits _emit_wgrad_split_k_cshuffle_epilogue when
    // the builder's own spec.split_k > 1). The actual split-K DEGREE is a runtime launch
    // parameter (ks/ks_count are always-present kernargs, conv_abi.py), computed below from
    // the real problem shape -- rocke.helpers.split_k.select_split_k_wgrad plus
    // conv_split_k.py's _resolve_wgrad/_rule -- never read back out of this KMD field. A
    // capability-flagged instance still launches with splitK == 1 whenever the real problem
    // does not benefit (small GEMM), cannot reach the atomic epilogue (odd dW row/store
    // vector), or the degree collapses once clamped to the gridDim.z budget; a non-capable
    // instance (this KMD field <= 1, no atomic epilogue in the binary) always launches with
    // splitK == 1, exactly as before this change -- no path here ever launches a degree > 1
    // into a binary that cannot execute it.
    const auto splitKCapable = md.intField(SPLIT_K_FIELD).value_or(1) > 1;
    auto splitK = int64_t{1};
    if(splitKCapable)
    {
        const auto warpM = md.intField(WARP_M_FIELD).value_or(0);
        const auto warpN = md.intField(WARP_N_FIELD).value_or(0);
        const auto waveSize = md.intField(WAVE_SIZE_FIELD).value_or(0);
        const auto wgN = wgrad::wgradGemmN(matched.problem, matched.geometry);
        const auto wgK = wgrad::wgradGemmK(matched.problem, matched.geometry);
        if(warpM > 0 && warpN > 0 && waveSize > 0 && wgN && wgK
           && wgrad::wgradAtomicEpilogueAvailable(matched.problem, matched.geometry))
        {
            const auto degree = wgrad::selectSplitKDegreeWgrad(wgrad::wgradGemmM(matched.geometry),
                                                                *wgN,
                                                                *wgK,
                                                                *tileM,
                                                                *tileN,
                                                                matched.problem.groups,
                                                                warpM * warpN * waveSize);
            // conv_split_k.py's _rule: clamp to the gridDim.z budget (z = groups *
            // ks_count) before deciding whether there is anything left to split;
            // group_merge is always 1 in this engine's catalog, so grid_groups ==
            // problem.groups exactly (no merged-group divisor to apply).
            const auto zCap = wgrad::MAX_GRID_DIM_Z / matched.problem.groups;
            splitK = std::max<int64_t>(1, std::min(degree, zCap));
        }
    }
    return wgrad::implicitGemmKernargs(
        matched.problem, matched.geometry, *tileM, *tileN, *tileK, splitK);
}

// ---------------------------------------------------------------------------
// Direct family (kernel_family 1) + its direct_wgrad_fold helper (kernel_family 2)
// ---------------------------------------------------------------------------

/// Where a direct_wgrad_fold kernel is looked up: the pack it was loaded from, its dtype
/// and its baked taps count -- the only things build_dw_wgrad_fold's binary depends on
/// (kernels/common/conv_direct_dw_wgrad.py's DwWgradFoldSpec). Mirrors Dgrad's HelperKey.
struct FoldKey
{
    std::string archive;
    DescriptorId packId;
    std::string dtype;
    int64_t taps;

    bool operator<(const FoldKey& other) const
    {
        return std::tie(archive, packId, dtype, taps)
               < std::tie(other.archive, other.packId, other.dtype, other.taps);
    }
};

/// The archive @p kernel's code object is read from, as buildIngestorKernelCode resolves
/// it; empty for a definition built outside a descriptor tree. Verbatim port of Dgrad's
/// archiveOf (Gfx950ConvDgradNative.cpp).
std::string archiveOf(const KernelDefinition& kernel)
{
    if(kernel.source.library.empty())
    {
        return {};
    }
    std::error_code ignored;
    const auto origin = std::filesystem::weakly_canonical(kernel.originDirectory, ignored);
    return std::filesystem::weakly_canonical(origin / kernel.source.library, ignored)
        .generic_string();
}

/// Every direct_wgrad_fold kernel this engine has seen, filed by FoldKey. This SDK has no
/// "runs once per kernel at load time" prepare hook, so this is filed as a side effect of
/// kernelMatches (registerFoldIfSeen, below) -- called by the framework for every catalog
/// kernel on every graph's matching pass, fold kernels included, regardless of what
/// kernelMatches itself returns for them (always false: a helper never serves a graph
/// directly). A key holding two different kernels is ambiguous and serves neither.
class FoldRegistry
{
public:
    void add(FoldKey key, const KernelDefinition& kernel)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        auto& entries = _entries[std::move(key)];
        for(const auto& entry : entries)
        {
            if(entry.kernelId == kernel.kernelId)
            {
                return;
            }
        }
        entries.push_back(kernel);
    }

    std::optional<KernelDefinition> find(const FoldKey& key) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto it = _entries.find(key);
        if(it == _entries.end() || it->second.size() != 1)
        {
            return std::nullopt;
        }
        return it->second.front();
    }

private:
    mutable std::mutex _mutex;
    std::map<FoldKey, std::vector<KernelDefinition>> _entries;
};

FoldRegistry& foldRegistry()
{
    static FoldRegistry s_registry;
    return s_registry;
}

/// Files @p kernel in foldRegistry() if it is a direct_wgrad_fold kernel; a no-op
/// otherwise. Called from kernelMatches's KERNEL_FAMILY_WGRAD_HELPER branch, which every
/// graph's matching pass reaches for every helper kernel in the catalog regardless of
/// match outcome -- see FoldRegistry's doc comment for why this is enough to have every
/// fold kernel on file well before the one producer instance that needs it ever reaches
/// prepare() (prepareDirect runs only after a full matching pass has scored every
/// candidate, fold kernels included).
void registerFoldIfSeen(const KernelDefinition& kernel, const KernelMetadata& md)
{
    const auto* variant = md.stringField(DIRECT_VARIANT_FIELD);
    const auto* dtype = md.stringField(DTYPE_FIELD);
    const auto taps = md.intField(TAPS_FIELD);
    if(variant != nullptr && *variant == DIRECT_VARIANT_FOLD && dtype != nullptr
       && (*dtype == "fp16" || *dtype == "bf16") && taps && *taps > 0)
    {
        foldRegistry().add(FoldKey{archiveOf(kernel), kernel.packId, *dtype, *taps}, kernel);
    }
}

/// Where a direct_wgrad_cast kernel is looked up: the pack it was loaded from and its
/// dtype -- the only things build_direct_wgrad_cast's binary depends on
/// (DirectWgradCastSpec has no geometry knob, just dtype_d). No taps axis, unlike
/// FoldKey: exactly one cast binary exists per dtype in the whole catalog.
struct CastKey
{
    std::string archive;
    DescriptorId packId;
    std::string dtype;

    bool operator<(const CastKey& other) const
    {
        return std::tie(archive, packId, dtype) < std::tie(other.archive, other.packId, other.dtype);
    }
};

/// Mirrors FoldRegistry exactly, for the one-per-dtype cast kernel.
class CastRegistry
{
public:
    void add(CastKey key, const KernelDefinition& kernel)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        auto& entries = _entries[std::move(key)];
        for(const auto& entry : entries)
        {
            if(entry.kernelId == kernel.kernelId)
            {
                return;
            }
        }
        entries.push_back(kernel);
    }

    std::optional<KernelDefinition> find(const CastKey& key) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        const auto it = _entries.find(key);
        if(it == _entries.end() || it->second.size() != 1)
        {
            return std::nullopt;
        }
        return it->second.front();
    }

private:
    mutable std::mutex _mutex;
    std::map<CastKey, std::vector<KernelDefinition>> _entries;
};

CastRegistry& castRegistry()
{
    static CastRegistry s_registry;
    return s_registry;
}

/// Files @p kernel in castRegistry() if it is a direct_wgrad_cast kernel; a no-op
/// otherwise. Same discipline as registerFoldIfSeen: called unconditionally for every
/// KERNEL_FAMILY_WGRAD_HELPER kernel kernelMatches sees, regardless of match outcome.
void registerCastIfSeen(const KernelDefinition& kernel, const KernelMetadata& md)
{
    const auto* variant = md.stringField(DIRECT_VARIANT_FIELD);
    const auto* dtype = md.stringField(DTYPE_FIELD);
    if(variant != nullptr && *variant == DIRECT_VARIANT_CAST && dtype != nullptr
       && (*dtype == "fp16" || *dtype == "bf16"))
    {
        castRegistry().add(CastKey{archiveOf(kernel), kernel.packId, *dtype}, kernel);
    }
}

/// Which second-stage kernel (if any) finishes a direct instance's atomic/partial-slot
/// output into the real dW: the five folded variants reduce a tap-major partial-slot
/// scratch (direct_wgrad_fold); the two non-folded ones (direct_wgrad, direct_wgrad_tiled)
/// atomic-add their fp32 partial sums into an fp32 scratch laid out byte-identically to
/// the real dW (same offset formula, no permutation) and need only a narrowing cast
/// (direct_wgrad_cast) to reach the graph's real fp16/bf16 dW tensor.
enum class SecondStage
{
    None,
    Fold,
    Cast
};

/// A direct instance's prepared launch: its own grid/block/kernargs, and, for a variant
/// with a second stage, what that stage needs. workspaceBytes/foldParts/foldChannels/taps
/// are the fold's own parameters (direct_wgrad_fold); castCount is the cast's (the total
/// dW element count the flat pass casts, one thread per element at CAST_THREADS).
struct DirectPlan
{
    wgrad::LaunchGeometry launch;
    wgrad::DirectKernargs values;
    SecondStage secondStage = SecondStage::None;
    int64_t workspaceBytes = 0;
    int64_t foldParts = 0;
    int64_t foldChannels = 0;
    int64_t taps = 0;
    int64_t castCount = 0;
};

/// DirectWgradCastSpec.threads's only baked value across the whole catalog (never a
/// per-instance knob -- see convolution_backward_filter.py's HELPER_VARIANT_CAST).
constexpr int64_t CAST_THREADS = 256;

/// Every direct/helper instance bakes its exact filter/stride/pad/dilation into the IR
/// (unlike implicit-GEMM's shape-generic tile): the compiled kernel computes with these as
/// constants, so a runtime problem must match every one of them exactly, not merely fit.
bool exactFilterMatch(const wgrad::Problem& problem, const KernelMetadata& md)
{
    const auto filterH = md.intField(FILTER_H_FIELD);
    const auto filterW = md.intField(FILTER_W_FIELD);
    const auto strideH = md.intField(STRIDE_H_FIELD);
    const auto strideW = md.intField(STRIDE_W_FIELD);
    const auto padH = md.intField(PAD_H_FIELD);
    const auto padW = md.intField(PAD_W_FIELD);
    const auto dilationH = md.intField(DILATION_H_FIELD);
    const auto dilationW = md.intField(DILATION_W_FIELD);
    if(!filterH || !filterW || !strideH || !strideW || !padH || !padW || !dilationH || !dilationW)
    {
        return false;
    }
    return *filterH == problem.y && *filterW == problem.x && *strideH == problem.strideH
           && *strideW == problem.strideW && *padH == problem.padH && *padW == problem.padW
           && *dilationH == problem.dilationH && *dilationW == problem.dilationW;
}

/// 0 on the KMD means "runtime_channels" (conv_implicit_gemm_wgrad.py's convention this
/// engine already uses for vector_size_a/b): the instance reads the real channel count
/// from its kernarg rather than baking one, so any value matches here -- the variant's own
/// divisibility/width checks (encoded in its LaunchGeometry function) are the real gate.
/// Nonzero means the binary bakes an exact channel count (e.g. grouped_mfma's cpg/kpg in
/// {4, 8}, or every depthwise variant's cpg == kpg == 1): only that exact runtime value
/// may dispatch to it.
bool channelMatches(int64_t kmdValue, int64_t runtimeValue)
{
    return kmdValue == 0 || kmdValue == runtimeValue;
}

/// This kernel's direct-family plan on @p matched, or nullopt when its baked knobs or
/// filter/stride/pad/dilation/channel-count capabilities do not admit the problem, or (for
/// a variant with a second stage) its matching direct_wgrad_fold / direct_wgrad_cast
/// kernel is not in this pack. A kernel serves a direct-family graph exactly when this
/// succeeds, same discipline as implicitGemmPlan.
///
/// History: an earlier session wired, GPU-benchmarked and `--validate pytorch`-checked all
/// seven direct variants and found every one produced wrong output (100% element mismatch,
/// max_abs_diff/max_rel_diff inf) on real graphs. That session's grid/block/kernarg math
/// cross-check against rocKE's own spec classes found nothing wrong here, which was
/// correct -- the bug was not in this dispatch layer. The real cause (see
/// WGRAD_AOT_PROGRESS.md): every direct-family kernel's "D" kernarg is documented fp32
/// (conv_abi.py's `conv_direct_args_signature`), but `matchProblem`'s shared
/// `isDeviceTensor` check requires the graph's real dW tensor to be the same dtype as
/// X/dY (fp16/bf16, same as implicit-GEMM's dW) -- so every direct kernel was writing
/// genuine 4-byte fp32 values into a tensor sized for 2-byte elements. The fix is
/// two-part, both in rocKE: `direct_wgrad_fold` (the five folded variants' shared reducer)
/// grew a `dtype_d` field so its final store casts the f32 reduction down to the real
/// dtype; `direct_wgrad` and `direct_wgrad_tiled` (whose atomic-add offset formula already
/// matches dW's final layout exactly, no permutation needed) now point "D" at an fp32
/// scratch instead of the real dW and run a new, separate `direct_wgrad_cast` kernel (a
/// flat narrowing cast, no reduction) to reach it. See SecondStage.
std::optional<DirectPlan> directPlan(const MatchedProblem& matched, const KernelDefinition& kernel)
{
    const KernelMetadata md{kernel};
    const auto* dtype = md.stringField(DTYPE_FIELD);
    if(dtype == nullptr || *dtype != dtypeName(matched.dtype) || !exactFilterMatch(matched.problem, md))
    {
        return std::nullopt;
    }
    const auto cpgField = md.intField(CPG_FIELD).value_or(0);
    const auto kpgField = md.intField(KPG_FIELD).value_or(0);
    if(!channelMatches(cpgField, matched.geometry.cpg) || !channelMatches(kpgField, matched.geometry.kpg))
    {
        return std::nullopt;
    }
    const auto* variant = md.stringField(DIRECT_VARIANT_FIELD);
    if(variant == nullptr)
    {
        return std::nullopt;
    }
    const auto& problem = matched.problem;
    const auto& geometry = matched.geometry;

    if(*variant == DIRECT_VARIANT_WGRAD)
    {
        const auto wavesC = md.intField(WAVES_C_FIELD);
        const auto wavesK = md.intField(WAVES_K_FIELD);
        const auto wavesQ = md.intField(WAVES_Q_FIELD);
        const auto hoPerBlock = md.intField(HO_PER_BLOCK_FIELD);
        const auto mfmaK = md.intField(MFMA_K_FIELD);
        if(!wavesC || !wavesK || !wavesQ || !hoPerBlock || !mfmaK)
        {
            return std::nullopt;
        }
        const auto launch = wgrad::directWgradLaunchGeometry(
            problem, geometry, *wavesC, *wavesK, *wavesQ, *hoPerBlock, *mfmaK);
        if(!launch)
        {
            return std::nullopt;
        }
        // direct_wgrad atomic-adds its fp32 partial sums into "D" with an offset formula
        // that already matches dW's final [K, Y, X, C/groups] layout exactly; "D" is
        // pointed at an fp32 scratch of the same element count, and direct_wgrad_cast
        // narrows it into the graph's real fp16/bf16 dW afterward.
        const auto castCount = wgrad::boundedProduct({problem.k, geometry.cpg, problem.y, problem.x});
        if(!castCount)
        {
            return std::nullopt;
        }
        const auto wsBytes = wgrad::boundedProduct({*castCount, 4});
        if(!wsBytes)
        {
            return std::nullopt;
        }
        return DirectPlan{
            *launch, wgrad::directKernargs(problem, geometry), SecondStage::Cast, *wsBytes, 0, 0, 0, *castCount};
    }
    if(*variant == DIRECT_VARIANT_WGRAD_TILED)
    {
        const auto wavesK = md.intField(WAVES_K_FIELD);
        const auto wavesC = md.intField(WAVES_C_FIELD);
        const auto waveK16 = md.intField(WAVE_K16_FIELD).value_or(0);
        const auto waveC16 = md.intField(WAVE_C16_FIELD).value_or(0);
        const auto qGroups = md.intField(Q_GROUPS_FIELD);
        const auto unfoldN = md.intField(UNFOLD_N_FIELD);
        const auto rowsPerSeg = md.intField(ROWS_PER_SEG_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        if(!wavesK || !wavesC || !qGroups || !unfoldN || !rowsPerSeg || !targetBlocks)
        {
            return std::nullopt;
        }
        const auto launch = wgrad::directWgradTiledLaunchGeometry(problem,
                                                                   geometry,
                                                                   *wavesK,
                                                                   *wavesC,
                                                                   waveK16,
                                                                   waveC16,
                                                                   *qGroups,
                                                                   *unfoldN,
                                                                   *rowsPerSeg,
                                                                   *targetBlocks);
        if(!launch)
        {
            return std::nullopt;
        }
        // Same cast-second-stage reasoning as direct_wgrad above.
        const auto castCount = wgrad::boundedProduct({problem.k, geometry.cpg, problem.y, problem.x});
        if(!castCount)
        {
            return std::nullopt;
        }
        const auto wsBytes = wgrad::boundedProduct({*castCount, 4});
        if(!wsBytes)
        {
            return std::nullopt;
        }
        return DirectPlan{
            *launch, wgrad::directKernargs(problem, geometry), SecondStage::Cast, *wsBytes, 0, 0, 0, *castCount};
    }

    // Every remaining variant writes a partial-slot scratch and needs a matching
    // direct_wgrad_fold kernel in this same pack: without it the plan is incomplete, so
    // decline rather than dispatch half a pipeline.
    std::optional<wgrad::FoldedDirectLaunch> folded;
    int64_t taps = 0;
    if(*variant == DIRECT_VARIANT_DW)
    {
        if(geometry.cpg != 1 || geometry.kpg != 1)
        {
            return std::nullopt;
        }
        const auto lanesC = md.intField(LANES_C_FIELD);
        const auto lanesW = md.intField(LANES_W_FIELD);
        const auto colsPerLane = md.intField(COLS_PER_LANE_FIELD).value_or(1);
        const auto prefetch = md.intField(PREFETCH_FIELD);
        const auto rowsPerSeg = md.intField(ROWS_PER_SEG_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        if(!lanesC || !lanesW || !prefetch || !rowsPerSeg || !targetBlocks)
        {
            return std::nullopt;
        }
        folded = wgrad::directWgradDwLaunchGeometry(
            problem, geometry, *lanesC, *lanesW, colsPerLane, *prefetch, *rowsPerSeg, *targetBlocks);
        taps = problem.y * problem.x;
    }
    else if(*variant == DIRECT_VARIANT_DW_COLS)
    {
        if(geometry.cpg != 1 || geometry.kpg != 1)
        {
            return std::nullopt;
        }
        const auto strip = md.intField(STRIP_FIELD);
        const auto threads = md.intField(THREADS_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        if(!strip || !threads || !targetBlocks)
        {
            return std::nullopt;
        }
        folded = wgrad::directWgradDwColsLaunchGeometry(problem, geometry, *strip, *threads, *targetBlocks);
        taps = problem.y * problem.x;
    }
    else if(*variant == DIRECT_VARIANT_DW_MFMA)
    {
        if(geometry.cpg != 1 || geometry.kpg != 1)
        {
            return std::nullopt;
        }
        const auto qTiles = md.intField(Q_TILES_FIELD);
        const auto rowsPerChunk = md.intField(ROWS_PER_CHUNK_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        const auto unfoldN = md.intField(UNFOLD_N_FIELD).value_or(1);
        if(!qTiles || !rowsPerChunk || !targetBlocks)
        {
            return std::nullopt;
        }
        folded = wgrad::directWgradDwMfmaLaunchGeometry(
            problem, geometry, *qTiles, *rowsPerChunk, *targetBlocks, unfoldN);
        taps = problem.y * problem.x;
    }
    else if(*variant == DIRECT_VARIANT_GROUPED_MFMA)
    {
        const auto wavesQ = md.intField(WAVES_Q_FIELD);
        const auto waveQ4 = md.intField(WAVE_Q4_FIELD).value_or(0);
        const auto rowsPerChunk = md.intField(ROWS_PER_CHUNK_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        if(!wavesQ || !rowsPerChunk || !targetBlocks)
        {
            return std::nullopt;
        }
        folded = wgrad::directWgradGroupedMfmaLaunchGeometry(
            problem, geometry, *wavesQ, waveQ4, *rowsPerChunk, *targetBlocks);
        taps = problem.y * problem.x * geometry.cpg;
    }
    else if(*variant == DIRECT_VARIANT_GATHER)
    {
        if(problem.groups != 1)
        {
            return std::nullopt;
        }
        const auto waveK16 = md.intField(WAVE_K16_FIELD).value_or(0);
        const auto wavesK = md.intField(WAVES_K_FIELD);
        const auto prefetch = md.intField(PREFETCH_FIELD);
        const auto rowsPerSeg = md.intField(ROWS_PER_SEG_FIELD);
        const auto targetBlocks = md.intField(TARGET_BLOCKS_FIELD);
        if(!wavesK || !prefetch || !rowsPerSeg || !targetBlocks)
        {
            return std::nullopt;
        }
        folded = wgrad::directWgradGatherLaunchGeometry(
            problem, geometry, waveK16, *wavesK, *prefetch, *rowsPerSeg, *targetBlocks);
        taps = problem.y * problem.x * geometry.cpg;
    }
    else
    {
        // DIRECT_VARIANT_FOLD / DIRECT_VARIANT_CAST themselves (never matched directly,
        // only ever reached through registerFoldIfSeen/registerCastIfSeen) or an unknown
        // variant.
        return std::nullopt;
    }
    if(!folded || taps <= 0)
    {
        return std::nullopt;
    }
    // Whether a matching direct_wgrad_fold kernel is actually on file is checked in
    // prepareDirect(), not here: kernelMatches/scoreKernel run across the whole catalog
    // once per graph (registerFoldIfSeen below files every fold kernel this same pass
    // reaches, in catalog order, after every direct producer), so declining here on a
    // registry that may not be warm yet would wrongly drop an admissible candidate on
    // whichever graph first touches this engine in the process's lifetime.
    return DirectPlan{folded->direct,
                      wgrad::directKernargs(problem, geometry),
                      SecondStage::Fold,
                      folded->workspaceBytes,
                      folded->foldParts,
                      folded->foldChannels,
                      taps,
                      0};
}

bool kernelMatches(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    if(!matched)
    {
        return false;
    }
    const KernelMetadata md{kernel};
    const auto family = md.intField(KERNEL_FAMILY_FIELD).value_or(KERNEL_FAMILY_IMPLICIT_GEMM);
    if(family == KERNEL_FAMILY_WGRAD_HELPER)
    {
        // Never matched directly: a dispatch that needs one loads it beside its instance
        // (foldRegistry/castRegistry, prepareDirect), same convention as Dgrad's MFMA
        // weight-transform helpers. Filing it here (rather than declining outright) is
        // this engine's catalog-discovery mechanism in the absence of a true prepare hook.
        // Both registries run unconditionally for every helper kernel; each is a no-op
        // for the other helper's direct_variant.
        registerFoldIfSeen(kernel, md);
        registerCastIfSeen(kernel, md);
        return false;
    }
    if(family == KERNEL_FAMILY_DIRECT)
    {
        // Re-enabled: see directPlan's doc comment for the root cause (an fp32-into-
        // fp16/bf16-buffer dtype mismatch, not a dispatch-layer bug) and the two-part fix
        // (the fold's dtype_d cast, the new cast kernel for the two atomic variants).
        return directPlan(*matched, kernel).has_value();
    }
    return implicitGemmPlan(*matched, kernel).has_value();
}

double scoreKernel(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
{
    // Deterministic fallback only; no sort_kernel_catalog ships with this engine (the
    // reference model was trained by an external fleet pipeline this session cannot
    // reproduce -- WGRAD_AOT_PROGRESS.md). Prefer the larger tile (more MACs/thread) among
    // admitted candidates; a kernel kernelMatches declined never reaches this.
    const auto matched = matchedFromBound(context, bound);
    if(!matched)
    {
        return -1.0;
    }
    const KernelMetadata md{kernel};
    const auto family = md.intField(KERNEL_FAMILY_FIELD).value_or(KERNEL_FAMILY_IMPLICIT_GEMM);
    if(family == KERNEL_FAMILY_DIRECT)
    {
        // No sort_kernel_catalog for the direct family either: total launched threads is
        // the same "more hardware in flight" proxy as implicit-GEMM's tile product.
        const auto plan = directPlan(*matched, kernel);
        if(!plan)
        {
            return -1.0;
        }
        return static_cast<double>(plan->launch.gridX) * static_cast<double>(plan->launch.gridY)
               * static_cast<double>(plan->launch.gridZ) * static_cast<double>(plan->launch.blockX);
    }
    if(!implicitGemmPlan(*matched, kernel).has_value())
    {
        return -1.0;
    }
    const auto tileM = md.intField(TILE_M_FIELD).value_or(0);
    const auto tileN = md.intField(TILE_N_FIELD).value_or(0);
    const auto tileK = md.intField(TILE_K_FIELD).value_or(0);
    return static_cast<double>(tileM) * static_cast<double>(tileN) * static_cast<double>(tileK);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::vector<KernelArgument> implicitGemmSignature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("dY");
    addPtr("X");
    addPtr("dW");
    addI32("dY_bytes");
    addI32("X_bytes");
    addI32("dW_bytes");
    for(const auto name : wgrad::IMPLICIT_GEMM_KERNARG_NAMES)
    {
        addI32(name);
    }
    return signature;
}

/// The direct-family 20-argument ABI (conv_abi.py's conv_direct_arg_names(direction="wgrad")):
/// A(dY), B(X), D(dW or, for a folded variant, the partial-slot scratch), the three byte
/// counts, then wgrad::DirectKernargs's 14 scalars in order.
std::vector<KernelArgument> directSignature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("A");
    addPtr("B");
    addPtr("D");
    addI32("A_bytes");
    addI32("B_bytes");
    addI32("D_bytes");
    for(const auto name : {"p_N",
                           "p_Hi",
                           "p_Wi",
                           "p_Ho",
                           "p_Wo",
                           "p_groups",
                           "p_total_c",
                           "p_total_k",
                           "p_dY_stride_n",
                           "p_dY_stride_ho",
                           "p_dY_stride_wo",
                           "p_X_stride_n",
                           "p_X_stride_hi",
                           "p_X_stride_wi"})
    {
        addI32(name);
    }
    return signature;
}

/// direct_wgrad_fold's ABI (kernels/common/conv_direct_dw_wgrad.py's
/// dw_wgrad_fold_signature): A(the partial-slot scratch), D(dW), the two byte counts, then
/// parts/channels.
std::vector<KernelArgument> foldSignature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("A");
    addPtr("D");
    addI32("A_bytes");
    addI32("D_bytes");
    addI32("parts");
    addI32("channels");
    return signature;
}

/// direct_wgrad_cast's ABI (kernels/common/conv_direct_wgrad_cast.py's
/// direct_wgrad_cast_signature): A(the fp32 scratch), D(dW), the two byte counts, then
/// count. A `global_buffer` argument is a dtype-erased 8-byte pointer (see
/// KernelArgument's fields: kind/size/offset/name, no element type) -- unlike the Python
/// signature builder, which threads D's real dtype through for the kpack loader to bind a
/// correctly-sized kernarg slot on that side, this host-side ABI description does not need
/// to vary with D's pointee dtype, only with which fields exist and in what order.
std::vector<KernelArgument> castSignature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto addPtr = [&](std::string_view name) {
        offset = (offset + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*);
        signature.push_back({"global_buffer", static_cast<uint32_t>(sizeof(void*)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(void*));
    };
    const auto addI32 = [&](std::string_view name) {
        offset = (offset + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        signature.push_back({"by_value", static_cast<uint32_t>(sizeof(int32_t)), offset, std::string(name)});
        offset += static_cast<uint32_t>(sizeof(int32_t));
    };
    addPtr("A");
    addPtr("D");
    addI32("A_bytes");
    addI32("D_bytes");
    addI32("count");
    return signature;
}

class PreparedConvWgrad : public PreparedDispatch
{
public:
    PreparedConvWgrad(IngestorKernelCode code,
                      Binding binding,
                      std::array<int64_t, 3> bytes,
                      wgrad::ImplicitGemmKernargs values)
        : _code(std::move(code))
        , _binding(binding)
        , _bytes(bytes)
        , _values(values)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }
    const Binding& binding() const
    {
        return _binding;
    }
    const std::array<int64_t, 3>& bytes() const
    {
        return _bytes;
    }
    const wgrad::ImplicitGemmKernargs& values() const
    {
        return _values;
    }

private:
    IngestorKernelCode _code;
    Binding _binding;
    std::array<int64_t, 3> _bytes;
    wgrad::ImplicitGemmKernargs _values;
};

/// A prepared direct-family instance: its own loaded kernel, and, for a variant with a
/// second stage (DirectPlan::secondStage), the loaded direct_wgrad_fold or
/// direct_wgrad_cast kernel its launch() runs second. The member/method names below say
/// "fold" for historical reasons (it was the first second-stage kernel wired); they hold
/// either kind interchangeably -- which one is in `_foldCode` is `_plan.secondStage`.
class PreparedConvWgradDirect : public PreparedDispatch
{
public:
    PreparedConvWgradDirect(IngestorKernelCode code,
                            std::optional<IngestorKernelCode> foldCode,
                            Binding binding,
                            std::array<int64_t, 3> bytes,
                            wgrad::DirectKernargs values,
                            DirectPlan plan)
        : _code(std::move(code))
        , _foldCode(std::move(foldCode))
        , _binding(binding)
        , _bytes(bytes)
        , _values(values)
        , _plan(std::move(plan))
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }
    compilation::IRunnableKernel& foldKernelForStream(hipStream_t stream) const
    {
        return _foldCode->kernelForStream(stream);
    }
    const Binding& binding() const
    {
        return _binding;
    }
    const std::array<int64_t, 3>& bytes() const
    {
        return _bytes;
    }
    const wgrad::DirectKernargs& values() const
    {
        return _values;
    }
    const DirectPlan& plan() const
    {
        return _plan;
    }

private:
    IngestorKernelCode _code;
    std::optional<IngestorKernelCode> _foldCode;
    Binding _binding;
    std::array<int64_t, 3> _bytes;
    wgrad::DirectKernargs _values;
    DirectPlan _plan;
};

class ConvWgradDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit ConvWgradDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// 0 for implicit-GEMM and the two non-folded direct variants; the partial-slot
    /// scratch size for one of the five folded direct variants.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const KernelMetadata md{kernel};
        if(md.intField(KERNEL_FAMILY_FIELD).value_or(KERNEL_FAMILY_IMPLICIT_GEMM) != KERNEL_FAMILY_DIRECT)
        {
            return 0;
        }
        const auto matched = matchedFromBound(context, bound);
        const auto plan = matched ? directPlan(*matched, kernel) : std::nullopt;
        return plan ? static_cast<size_t>(plan->workspaceBytes) : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const KernelMetadata familyMd{kernel};
        if(familyMd.intField(KERNEL_FAMILY_FIELD).value_or(KERNEL_FAMILY_IMPLICIT_GEMM)
           == KERNEL_FAMILY_DIRECT)
        {
            return prepareDirect(context, bound, kernel);
        }
        const auto matched = matchedFromBound(context, bound);
        const auto values = matched ? implicitGemmPlan(*matched, kernel) : std::nullopt;
        if(!matched || !values)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad graph or packaged kernel does not satisfy the convolution "
                "contract");
        }
        // (*values)[18]/[19] are p_wg_M/p_wg_N (the GEMM problem extents, wg_M == kpg
        // exactly) -- NOT the kernel's baked tile sizes. Passing them as tileM/tileN here
        // made numPidM/numPidN collapse to 1 (ceil(wg_M/wg_M)), so the launch covered only
        // the first tile of dW and left the rest unwritten: a real, silent correctness bug,
        // not a grid-limit refusal. Re-read the same KMD fields implicitGemmPlan already
        // validated as present for this kernel.
        const KernelMetadata md{kernel};
        const auto tileM = md.intField(TILE_M_FIELD).value_or(0);
        const auto tileN = md.intField(TILE_N_FIELD).value_or(0);
        // block_size = warp_m * warp_n * wave_size (conv_implicit_gemm_wgrad.py's
        // WgradConvSpec.block_size), per-instance -- not a fixed 256. A wrong thread
        // count silently mis-shapes every thread's tile offset instead of refusing.
        const auto warpM = md.intField(WARP_M_FIELD).value_or(0);
        const auto warpN = md.intField(WARP_N_FIELD).value_or(0);
        const auto waveSize = md.intField(WAVE_SIZE_FIELD).value_or(0);
        const auto blockSize = warpM * warpN * waveSize;
        const auto geom = wgrad::implicitGemmLaunchGeometry(
            matched->problem, matched->geometry, tileM, tileN, (*values)[41], blockSize);
        if(!geom)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_wgrad launch grid exceeds device limits");
        }
        auto code = buildIngestorKernelCode(_loader, context, kernel, implicitGemmSignature());
        code.setBlockSize(geom->blockX, 1, 1);
        code.setGridSize(geom->gridX, geom->gridY, geom->gridZ);
        return std::make_unique<PreparedConvWgrad>(
            std::move(code), matched->binding, matched->geometry.tensorBytes, *values);
    }

    /// Loads the direct kernel itself and, for a variant with a second stage, its
    /// direct_wgrad_fold or direct_wgrad_cast kernel -- directPlan already confirmed the
    /// shape is admissible; the registry lookup here confirms the helper is actually in
    /// this pack.
    std::unique_ptr<PreparedDispatch> prepareDirect(const MatchContext& context,
                                                    const BoundTokens& bound,
                                                    const KernelDefinition& kernel) const
    {
        const auto matched = matchedFromBound(context, bound);
        const auto plan = matched ? directPlan(*matched, kernel) : std::nullopt;
        if(!matched || !plan)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad graph or packaged direct kernel does not satisfy the "
                "convolution contract");
        }
        auto code = buildIngestorKernelCode(_loader, context, kernel, directSignature());
        code.setBlockSize(plan->launch.blockX, 1, 1);
        code.setGridSize(plan->launch.gridX, plan->launch.gridY, plan->launch.gridZ);
        std::optional<IngestorKernelCode> secondStageCode;
        if(plan->secondStage == SecondStage::Fold)
        {
            const KernelMetadata md{kernel};
            const auto* dtype = md.stringField(DTYPE_FIELD);
            const auto helper = dtype
                ? foldRegistry().find(FoldKey{archiveOf(kernel), kernel.packId, *dtype, plan->taps})
                : std::nullopt;
            const auto foldGeom = wgrad::directWgradFoldLaunchGeometry(plan->foldChannels, plan->taps);
            if(!helper || !foldGeom)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad direct instance has no direct_wgrad_fold helper in its "
                    "pack");
            }
            secondStageCode = buildIngestorKernelCode(_loader, context, *helper, foldSignature());
            secondStageCode->setBlockSize(foldGeom->blockX, 1, 1);
            secondStageCode->setGridSize(foldGeom->gridX, foldGeom->gridY, foldGeom->gridZ);
        }
        else if(plan->secondStage == SecondStage::Cast)
        {
            const KernelMetadata md{kernel};
            const auto* dtype = md.stringField(DTYPE_FIELD);
            const auto helper = dtype
                ? castRegistry().find(CastKey{archiveOf(kernel), kernel.packId, *dtype})
                : std::nullopt;
            const auto castGeom = wgrad::directWgradCastLaunchGeometry(plan->castCount, CAST_THREADS);
            if(!helper || !castGeom)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad direct instance has no direct_wgrad_cast helper in its "
                    "pack");
            }
            secondStageCode = buildIngestorKernelCode(_loader, context, *helper, castSignature());
            secondStageCode->setBlockSize(castGeom->blockX, 1, 1);
            secondStageCode->setGridSize(castGeom->gridX, castGeom->gridY, castGeom->gridZ);
        }
        return std::make_unique<PreparedConvWgradDirect>(std::move(code),
                                                         std::move(secondStageCode),
                                                         matched->binding,
                                                         matched->geometry.tensorBytes,
                                                         wgrad::directKernargs(matched->problem, matched->geometry),
                                                         *plan);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        if(const auto* direct = dynamic_cast<const PreparedConvWgradDirect*>(&prepared))
        {
            launchDirect(handle, *direct, deviceBuffers, numDeviceBuffers, workspace);
            return;
        }
        const auto& wgradPrepared = dynamic_cast<const PreparedConvWgrad&>(prepared);
        const auto& binding = wgradPrepared.binding();
        void* dy
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers).ptr;
        void* x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers).ptr;
        void* dw
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.dw, deviceBuffers, numDeviceBuffers).ptr;
        for(const auto* ptr : {dy, x, dw})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad requires non-null device pointers aligned to 16 bytes");
            }
        }
        const auto& bytes = wgradPrepared.bytes();
        if(!wgrad::wideProduct({bytes[0]}) || !wgrad::wideProduct({bytes[1]})
           || !wgrad::wideProduct({bytes[2]}))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_wgrad: invalid tensor byte sizes");
        }
        const auto& values = wgradPrepared.values();
        // conv_implicit_gemm_wgrad.py's split-K contract (p_num_pid_n .. ks_count is the
        // trailing pair; ks_count is values.back()): "the kernel only issues atomic-adds,
        // never a direct store, so any non-zero initial content accumulates into the
        // result... Caller contract (split_k > 1): zero-initialise the dW buffer before
        // every launch." A split_k == 1 kernel stores directly and needs no zeroing.
        if(values.back() > 1)
        {
            const hipError_t memsetErr
                = hipMemsetAsync(dw, 0, static_cast<size_t>(bytes[2]), handle.getStream());
            if(memsetErr != hipSuccess)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    std::string("gfx950_conv_wgrad: failed to zero dW before a split-K atomic "
                                "launch, error: ")
                        + hipGetErrorString(memsetErr));
            }
        }
        std::array<int32_t, 3> byteArgs{
            static_cast<int32_t>(bytes[0]), static_cast<int32_t>(bytes[1]), static_cast<int32_t>(bytes[2])};
        std::vector<void*> params{&dy, &x, &dw, &byteArgs[0], &byteArgs[1], &byteArgs[2]};
        for(auto& value : values)
        {
            params.push_back(const_cast<int32_t*>(&value));
        }
        wgradPrepared.kernelForStream(handle.getStream())
            .launchWithParams(handle.getStream(), params.data());
    }

private:
    /// The two non-folded variants (direct_wgrad, direct_wgrad_tiled) atomic-add their fp32
    /// partial sums into a fp32 workspace laid out byte-identically to dW (same contract as
    /// implicit-GEMM's split-K path: the caller zero-initialises the atomic target first),
    /// then direct_wgrad_cast narrows the workspace into the graph's real fp16/bf16 dW. The
    /// five folded variants store one partial slot per partition into the workspace with a
    /// plain store (never an atomic), so it needs no zeroing -- only the fold kernel's own
    /// store into dW, once, is the only write dW ever sees.
    void launchDirect(const Handle& handle,
                      const PreparedConvWgradDirect& prepared,
                      const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                      uint32_t numDeviceBuffers,
                      void* workspace) const
    {
        const auto& binding = prepared.binding();
        void* dy = hipdnn_plugin_sdk::findDeviceBuffer(binding.dy, deviceBuffers, numDeviceBuffers).ptr;
        void* x = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers).ptr;
        void* dw = hipdnn_plugin_sdk::findDeviceBuffer(binding.dw, deviceBuffers, numDeviceBuffers).ptr;
        const auto& plan = prepared.plan();
        for(const auto* ptr : {dy, x, dw})
        {
            if(ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx950_conv_wgrad requires non-null device pointers aligned to 16 bytes");
            }
        }
        if(plan.secondStage != SecondStage::None && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950_conv_wgrad direct instance requires a workspace");
        }
        const auto& bytes = prepared.bytes();
        if(!wgrad::wideProduct({bytes[0]}) || !wgrad::wideProduct({bytes[1]})
           || !wgrad::wideProduct({bytes[2]}))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "gfx950_conv_wgrad: invalid tensor byte sizes");
        }
        const bool hasSecondStage = plan.secondStage != SecondStage::None;
        void* d = hasSecondStage ? workspace : dw;
        const int32_t dBytes
            = hasSecondStage ? static_cast<int32_t>(plan.workspaceBytes) : static_cast<int32_t>(bytes[2]);
        // The producer atomic-adds whenever its own stage is None (a hypothetical bare
        // direct-into-dW variant) or Cast (direct_wgrad/direct_wgrad_tiled, atomic-adding
        // into the fp32 workspace cast later narrows): either way the atomic target needs
        // zeroing first. Fold's five variants plain-store one partial slot each -- no
        // atomic, no zeroing; only the fold kernel's own store into dW is unconditional.
        if(plan.secondStage != SecondStage::Fold)
        {
            const hipError_t memsetErr = hipMemsetAsync(d, 0, static_cast<size_t>(dBytes), handle.getStream());
            if(memsetErr != hipSuccess)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    std::string("gfx950_conv_wgrad: failed to zero the direct atomic launch's "
                                "target, error: ")
                        + hipGetErrorString(memsetErr));
            }
        }
        std::array<int32_t, 3> byteArgs{
            static_cast<int32_t>(bytes[0]), static_cast<int32_t>(bytes[1]), dBytes};
        const auto& values = prepared.values();
        std::vector<void*> params{&dy, &x, &d, &byteArgs[0], &byteArgs[1], &byteArgs[2]};
        for(auto& value : values)
        {
            params.push_back(const_cast<int32_t*>(&value));
        }
        prepared.kernelForStream(handle.getStream()).launchWithParams(handle.getStream(), params.data());
        if(plan.secondStage == SecondStage::Fold)
        {
            std::array<int32_t, 2> foldByteArgs{dBytes, static_cast<int32_t>(bytes[2])};
            std::array<int32_t, 2> foldScalars{
                static_cast<int32_t>(plan.foldParts), static_cast<int32_t>(plan.foldChannels)};
            std::vector<void*> foldParams{
                &workspace, &dw, &foldByteArgs[0], &foldByteArgs[1], &foldScalars[0], &foldScalars[1]};
            prepared.foldKernelForStream(handle.getStream())
                .launchWithParams(handle.getStream(), foldParams.data());
        }
        else if(plan.secondStage == SecondStage::Cast)
        {
            std::array<int32_t, 2> castByteArgs{dBytes, static_cast<int32_t>(bytes[2])};
            std::array<int32_t, 1> castScalars{static_cast<int32_t>(plan.castCount)};
            std::vector<void*> castParams{
                &workspace, &dw, &castByteArgs[0], &castByteArgs[1], &castScalars[0]};
            prepared.foldKernelForStream(handle.getStream())
                .launchWithParams(handle.getStream(), castParams.data());
        }
    }

    const compilation::KpackKernelLoader& _loader;
};

const ConvWgradDispatchHandler& dispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx950ConvWgradAotKpackModuleCache());
    static const ConvWgradDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx950ConvWgradAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void registerGfx950ConvWgradAotSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());

    // Each pack's "operation is <shard>" UMD names a per-shard graph-scope criterion (the
    // manifest's per-pack `discriminator`), a GraphCriterionFn -- distinct from graph_match
    // (a GraphMatchFn). See shardDtypeCriterion's comment: only dtype distinguishes a shard
    // at the graph level.
    for(const auto* shard : {"bf16_compv3", "bf16_compv4", "bf16_direct", "bf16_mem"})
    {
        scope.add(std::string("hipkernel.gfx950_conv_wgrad_aot.") + shard + "_match",
                  &shardDtypeCriterion<data_objects::DataType::BFLOAT16>);
    }
    for(const auto* shard : {"fp16_compv3", "fp16_compv4", "fp16_direct", "fp16_mem"})
    {
        scope.add(std::string("hipkernel.gfx950_conv_wgrad_aot.") + shard + "_match",
                  &shardDtypeCriterion<data_objects::DataType::HALF>);
    }
}

void resetGfx950ConvWgradAotModuleCache()
{
    gfx950ConvWgradAotKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
