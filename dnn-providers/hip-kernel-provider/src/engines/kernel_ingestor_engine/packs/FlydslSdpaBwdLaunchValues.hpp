// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaCdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorScalarBindings.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdNativeShared.hpp"

/**
 * @file FlydslSdpaBwdLaunchValues.hpp
 * @brief The launch state of the FlyDSL SDPA backward dispatch handlers, and its saved form:
 *        versioned dispatch names and the named, typed values each launch is computed from.
 *        Every handler writes its values from its prepared dispatch and reads them back into
 *        its launch inputs.
 *
 * Eight contracts, one per dispatch handler: the split engine's `prep`, `dkdv`, `dq`,
 * `dq_reduce` (the dBias reduce kernel is its own launch of the row's `sources`), `dkdv_u2`
 * (the dK-only and the dV-only launch) and `dq_u2` (the V-head dQ and tail-zero launches), the
 * last two for V heads that differ from K's; and the merged engine's `merged` and
 * `merged_reduce` (the reduce kernel is its own launch of the row's `sources`). The plain `dq`
 * and `merged` handlers launch no reduce kernel, and refuse a saved plan whose problem needs
 * one. A dq, dq_reduce, dq_u2, merged or merged_reduce stage whose problem has a padding mask
 * and a dense gradient output also launches the tail-zero kernel, a launch of its own that the
 * saved problem implies.
 *
 * The version in a dispatch name identifies the contract between the saved values and the
 * launch. A saved plan pairs an older code object with this launch. Use a new version for a
 * change to any of these:
 *   - the names, types or meanings of the values.
 *   - the kernel arguments the launch passes: their order, count, kinds, sizes or meanings.
 *   - how the launch derives grid, block, workspace layout or window from the values.
 * Keep the handler of the old version registered while plans saved under it must load.
 *
 * Every contract saves the matched problem the launch reads, the nine tensor uids, the scalar
 * bindings and the fields of the kernel's row the launch reads. Derived values (window,
 * dropout threshold, run-time flags, kernel arguments, workspace layout, grids, strides after
 * the row pitch) are recomputed on restore and never saved. The launch reads only the element
 * width of the problem's data type, so that is what is saved.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_merged.dispatch.v1";
inline constexpr std::string_view FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1
    = "hipkernel.flydsl_sdpa_bwd_merged_reduce.dispatch.v1";

/// The primary launch name of a stage that launches the dBias reduce kernel or the tail-zero
/// kernel beside it: `dq` for a split dq stage and `merged` for the merged handler. The other
/// launches are named `flydsl_sdpa_bwd_native::REDUCE_LAUNCH` and `TAILZERO_LAUNCH`. A stage that
/// launches one kernel uses the empty name.
inline constexpr std::string_view FLYDSL_SDPA_BWD_DQ_LAUNCH = "dq";
inline constexpr std::string_view FLYDSL_SDPA_BWD_MERGED_LAUNCH = "merged";

/// The `grid` values of a split row, one launch order per spelling: x, y, z, with the batch
/// always on z.
inline constexpr std::string_view FLYDSL_SDPA_BWD_GRID_PREP = "qblocks_by_heads_by_seqs";
inline constexpr std::string_view FLYDSL_SDPA_BWD_GRID_DKDV_HEADS_FIRST
    = "kheads_by_kvblocks_by_seqs";
inline constexpr std::string_view FLYDSL_SDPA_BWD_GRID_DKDV_BLOCKS_FIRST
    = "kvblocks_by_kheads_by_seqs";
inline constexpr std::string_view FLYDSL_SDPA_BWD_GRID_DQ = "qheads_by_qblocks_by_seqs";

/// The tensor uids a matched SDPA backward graph binds.
using FlydslSdpaBwdUids = flydsl_sdpa_bwd_native::Uids;

/// Everything a split prep launch is computed from. The launch has 256 threads.
struct FlydslSdpaBwdPrepLaunchInputs
{
    flydsl_sdpa_bwd::SdpaBwdProblem problem;
    FlydslSdpaBwdUids uids;
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
    /// The row's `head_dim`, the tile the object is built at.
    int64_t headDim = 0;
    /// The row's `grid`.
    std::string grid;
};

/// Everything a split dkdv launch is computed from.
struct FlydslSdpaBwdDkdvLaunchInputs
{
    flydsl_sdpa_bwd::SdpaBwdProblem problem;
    FlydslSdpaBwdUids uids;
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
    int64_t headDim = 0;
    int64_t blockThreads = 0;
    int64_t blockKv = 0;
    std::string grid;
};

/// Everything a split dq launch is computed from, whichever of the `dq`, `dq_reduce` and `dq_u2`
/// dispatches launches it. The dBias reduce kernel of a `dq_reduce` stage, when the problem's
/// dBias is broadcast, is the row's `dbias_reduce` launch and adds no value.
struct FlydslSdpaBwdDqLaunchInputs
{
    flydsl_sdpa_bwd::SdpaBwdProblem problem;
    FlydslSdpaBwdUids uids;
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
    int64_t headDim = 0;
    int64_t blockThreads = 0;
    int64_t blockM = 0;
    std::string grid;
};

/// Everything a merged launch is computed from, whichever of the `merged` and `merged_reduce`
/// dispatches launches it. The dBias reduce kernel of a `merged_reduce` stage, when the
/// problem's dBias is broadcast, is the row's `dbias_reduce` launch and adds no value.
struct FlydslSdpaBwdMergedLaunchInputs
{
    flydsl_sdpa_bwd::SdpaBwdProblem problem;
    FlydslSdpaBwdUids uids;
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
    int64_t headDim = 0;
    int64_t blockThreads = 0;
    int64_t blockKv = 0;
    int64_t blockM = 0;
};

/// The arguments the split handler of @p slot launches, as the launch passes them; with
/// @p valueHeads, the arguments of the `dkdv_u2` or `dq_u2` handler (V heads that differ from
/// K's).
const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>&
    flydslSdpaBwdSplitSignature(FlydslSdpaBwdSplitSlot slot, bool valueHeads = false);

/// The arguments the merged handler launches.
const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& flydslSdpaBwdMergedSignature();

namespace flydsl_sdpa_bwd_value
{

inline constexpr std::string_view ELEMENT_BYTES = "element_bytes";
inline constexpr std::string_view BIAS_ELEMENT_BYTES = "bias_element_bytes";
inline constexpr std::string_view BATCH = "batch";
inline constexpr std::string_view NUM_HEADS = "num_heads";
inline constexpr std::string_view NUM_KV_HEADS = "num_kv_heads";
inline constexpr std::string_view NUM_V_HEADS = "num_v_heads";
inline constexpr std::string_view SEQLEN_Q = "seqlen_q";
inline constexpr std::string_view SEQLEN_KV = "seqlen_kv";
inline constexpr std::string_view HEAD_DIM_QK = "head_dim_qk";
inline constexpr std::string_view HEAD_DIM_V = "head_dim_v";
inline constexpr std::string_view SCALE_SOURCE = "scale_source";
inline constexpr std::string_view SCALE_VALUE = "scale_value";
inline constexpr std::string_view MASKED = "masked";
inline constexpr std::string_view ALIGN_BOTTOM_RIGHT = "align_bottom_right";
inline constexpr std::string_view DIAGONAL_OFFSET = "diagonal_offset";
inline constexpr std::string_view WINDOW_LEFT = "window_left";
inline constexpr std::string_view WINDOW_RIGHT = "window_right";
inline constexpr std::string_view PADDING_MASK = "padding_mask";
inline constexpr std::string_view SEQ_LEN_Q_UID = "seq_len_q.uid";
inline constexpr std::string_view SEQ_LEN_KV_UID = "seq_len_kv.uid";
inline constexpr std::string_view BIAS_UID = "bias.uid";
inline constexpr std::string_view DBIAS_UID = "dbias.uid";
inline constexpr std::string_view BIAS_BROADCAST_BATCH = "bias_broadcast_batch";
inline constexpr std::string_view BIAS_BROADCAST_HEAD = "bias_broadcast_head";
inline constexpr std::string_view DROPOUT_P = "dropout_p";
inline constexpr std::string_view SEED_WIDTH = "seed_width";
inline constexpr std::string_view SEED_FORM = "seed_form";
inline constexpr std::string_view SEED_UID = "seed.uid";
inline constexpr std::string_view OFFSET_UID = "offset.uid";
inline constexpr std::string_view HEAD_DIM = "head_dim";
inline constexpr std::string_view BLOCK_THREADS = "block_threads";
inline constexpr std::string_view BLOCK_KV = "block_kv";
inline constexpr std::string_view BLOCK_M = "block_m";
inline constexpr std::string_view GRID = "grid";

/// The binding roles of the pass-by-value scalars (IngestorScalarBindings.hpp).
inline constexpr std::string_view SCALE_ROLE = "scale";
inline constexpr std::string_view SEED_ROLE = "seed";
inline constexpr std::string_view OFFSET_ROLE = "offset";

} // namespace flydsl_sdpa_bwd_value

namespace detail
{

namespace bwd = flydsl_sdpa_bwd;
namespace bwdv = flydsl_sdpa_bwd_value;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

inline constexpr int64_t BWD_INT32_MAX = std::numeric_limits<int32_t>::max();
inline constexpr int64_t BWD_INT64_MAX = std::numeric_limits<int64_t>::max();

inline std::string bwdStrideName(std::string_view tensor, std::string_view axis)
{
    return std::string(tensor) + ".stride_" + std::string(axis);
}

/// One of the nine tensors a contract saves: its name, and where the uid, the strides and the
/// ragged offset of it live. The one list the writer, the reader and the allow-list read, so
/// a name cannot pair with another tensor's member.
struct BwdTensor
{
    std::string_view name;
    int64_t FlydslSdpaBwdUids::*uid;
    bwd::Strides bwd::SdpaBwdProblem::*strides;
    bwd::RaggedOffset bwd::RaggedOffsets::*ragged;
};

inline constexpr std::array<BwdTensor, 9> BWD_TENSORS{{
    {"q", &FlydslSdpaBwdUids::q, &bwd::SdpaBwdProblem::q, &bwd::RaggedOffsets::q},
    {"k", &FlydslSdpaBwdUids::k, &bwd::SdpaBwdProblem::k, &bwd::RaggedOffsets::k},
    {"v", &FlydslSdpaBwdUids::v, &bwd::SdpaBwdProblem::v, &bwd::RaggedOffsets::v},
    {"o", &FlydslSdpaBwdUids::o, &bwd::SdpaBwdProblem::o, &bwd::RaggedOffsets::o},
    {"do", &FlydslSdpaBwdUids::dO, &bwd::SdpaBwdProblem::dO, &bwd::RaggedOffsets::dO},
    {"dq", &FlydslSdpaBwdUids::dQ, &bwd::SdpaBwdProblem::dQ, &bwd::RaggedOffsets::dQ},
    {"dk", &FlydslSdpaBwdUids::dK, &bwd::SdpaBwdProblem::dK, &bwd::RaggedOffsets::dK},
    {"dv", &FlydslSdpaBwdUids::dV, &bwd::SdpaBwdProblem::dV, &bwd::RaggedOffsets::dV},
    {"stats", &FlydslSdpaBwdUids::stats, &bwd::SdpaBwdProblem::stats, &bwd::RaggedOffsets::stats},
}};

inline std::string bwdUidName(std::string_view tensor)
{
    return std::string(tensor) + ".uid";
}

inline std::string bwdRaggedName(std::string_view tensor, std::string_view field)
{
    return std::string(tensor) + ".ragged_offset_" + std::string(field);
}

[[noreturn]] inline void refuseBwdWrite(const std::string& why)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "cannot save the SDPA backward launch: " + why);
}

/// The spelling of each enumeration a contract saves as text, one table for the writer, the
/// reader and the allow-list of value names.
inline constexpr std::array<std::pair<std::string_view, bwd::ScaleSource>, 2> BWD_SCALE_SOURCES{
    {{"value", bwd::ScaleSource::VALUE}, {"tensor", bwd::ScaleSource::TENSOR}}};

inline constexpr std::array<std::pair<std::string_view, bwd::SeedWidth>, 3> BWD_SEED_WIDTHS{
    {{"none", bwd::SeedWidth::NONE},
     {"int32", bwd::SeedWidth::INT32},
     {"int64", bwd::SeedWidth::INT64}}};

inline constexpr std::array<std::pair<std::string_view, bwd::SeedForm>, 3> BWD_SEED_FORMS{
    {{"none", bwd::SeedForm::NONE},
     {"device", bwd::SeedForm::DEVICE},
     {"pass_by_value", bwd::SeedForm::PASS_BY_VALUE}}};

/// The spelling of @p value in @p table.
template <typename TEnum, size_t N>
std::string_view bwdEnumName(const std::array<std::pair<std::string_view, TEnum>, N>& table,
                             TEnum value)
{
    for(const auto& entry : table)
    {
        if(entry.second == value)
        {
            return entry.first;
        }
    }
    refuseBwdWrite("an enumerator has no saved spelling");
}

/// Writes the values every contract shares.
inline void writeBwdCommonValues(MetadataValues& values,
                                 const bwd::SdpaBwdProblem& problem,
                                 const FlydslSdpaBwdUids& uids,
                                 const flydsl_sdpa_bwd_native::ScalarOperands& scalars)
{
    const auto put
        = [&values](std::string_view name, auto value) { values[std::string(name)] = value; };
    const auto putInt = [&put](std::string_view name, int64_t value) { put(name, value); };
    const auto putText = [&values](std::string_view name, std::string_view value) {
        values[std::string(name)] = std::string(value);
    };

    putInt(bwdv::ELEMENT_BYTES, static_cast<int64_t>(problem.elementBytes));
    putInt(bwdv::BIAS_ELEMENT_BYTES, static_cast<int64_t>(problem.biasElementBytes));
    putInt(bwdv::BATCH, problem.batch);
    putInt(bwdv::NUM_HEADS, problem.numHeads);
    putInt(bwdv::NUM_KV_HEADS, problem.numKvHeads);
    putInt(bwdv::SEQLEN_Q, problem.seqLenQ);
    putInt(bwdv::SEQLEN_KV, problem.seqLenKv);
    putInt(bwdv::HEAD_DIM_QK, problem.headDimQk);
    putInt(bwdv::HEAD_DIM_V, problem.headDimV);
    putInt(bwdv::NUM_V_HEADS, problem.numVHeads);

    for(const auto& tensor : BWD_TENSORS)
    {
        putInt(bwdUidName(tensor.name), uids.*tensor.uid);
        const auto& strides = problem.*tensor.strides;
        putInt(bwdStrideName(tensor.name, "batch"), strides.batch);
        putInt(bwdStrideName(tensor.name, "head"), strides.head);
        putInt(bwdStrideName(tensor.name, "seq"), strides.seq);
        const auto& ragged = problem.ragged.*tensor.ragged;
        putInt(bwdRaggedName(tensor.name, "uid"), ragged.uid);
        putInt(bwdRaggedName(tensor.name, "multiplier"), ragged.multiplier);
    }

    putText(bwdv::SCALE_SOURCE, bwdEnumName(BWD_SCALE_SOURCES, problem.scaleSource));
    put(bwdv::SCALE_VALUE, static_cast<double>(problem.scaleValue));
    if(problem.scaleSource == bwd::ScaleSource::TENSOR)
    {
        if(!scalars.scale.has_value())
        {
            refuseBwdWrite("the scale is a tensor and the launch holds no scalar for it");
        }
        putScalarBinding(values, bwdv::SCALE_ROLE, *scalars.scale);
    }

    put(bwdv::MASKED, problem.masked);
    put(bwdv::ALIGN_BOTTOM_RIGHT, problem.alignBottomRight);
    putInt(bwdv::DIAGONAL_OFFSET, problem.diagonalOffset);
    putInt(bwdv::WINDOW_LEFT, problem.windowLeft);
    putInt(bwdv::WINDOW_RIGHT, problem.windowRight);

    put(bwdv::PADDING_MASK, problem.paddingMask);
    putInt(bwdv::SEQ_LEN_Q_UID, problem.seqLenQUid);
    putInt(bwdv::SEQ_LEN_KV_UID, problem.seqLenKvUid);

    putInt(bwdv::BIAS_UID, problem.biasUid);
    putInt(bwdv::DBIAS_UID, problem.dbiasUid);
    for(const auto& [name, strides] :
        {std::pair<std::string_view, const bwd::Strides*>{"bias", &problem.biasStrides},
         std::pair<std::string_view, const bwd::Strides*>{"dbias", &problem.dbiasStrides}})
    {
        putInt(bwdStrideName(name, "batch"), strides->batch);
        putInt(bwdStrideName(name, "head"), strides->head);
        putInt(bwdStrideName(name, "seq"), strides->seq);
    }
    put(bwdv::BIAS_BROADCAST_BATCH, problem.biasBroadcastBatch);
    put(bwdv::BIAS_BROADCAST_HEAD, problem.biasBroadcastHead);

    put(bwdv::DROPOUT_P, static_cast<double>(problem.dropoutP));
    putText(bwdv::SEED_WIDTH, bwdEnumName(BWD_SEED_WIDTHS, problem.seedWidth));
    putText(bwdv::SEED_FORM, bwdEnumName(BWD_SEED_FORMS, problem.seedForm));
    if(problem.seedForm == bwd::SeedForm::PASS_BY_VALUE)
    {
        if(!scalars.seed.has_value() || !scalars.offset.has_value())
        {
            refuseBwdWrite("the seed is pass-by-value and the launch holds no scalar for it");
        }
        putScalarBinding(values, bwdv::SEED_ROLE, *scalars.seed);
        putScalarBinding(values, bwdv::OFFSET_ROLE, *scalars.offset);
    }
    else if(problem.seedForm == bwd::SeedForm::DEVICE)
    {
        putInt(bwdv::SEED_UID, problem.seedUid);
        putInt(bwdv::OFFSET_UID, problem.offsetUid);
    }
}

/// The value names of the shared part, for a plan whose scale is a tensor or not and whose
/// seed has the form @p seedForm: the names writeBwdCommonValues writes for such a problem, with
/// every scalar binding baked so that its `baked` name is written too.
inline std::vector<std::string> bwdCommonValueNames(bool scaleIsTensor, bwd::SeedForm seedForm)
{
    bwd::SdpaBwdProblem problem;
    problem.scaleSource = scaleIsTensor ? bwd::ScaleSource::TENSOR : bwd::ScaleSource::VALUE;
    problem.seedForm = seedForm;

    hipdnn_plugin_sdk::ScalarOperand baked;
    baked.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    baked.bakedDefault = hipdnn_plugin_sdk::ScalarValue{int32_t{0}};
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
    scalars.scale = baked;
    scalars.seed = baked;
    scalars.offset = baked;

    MetadataValues values;
    writeBwdCommonValues(values, problem, FlydslSdpaBwdUids{}, scalars);

    std::vector<std::string> names;
    names.reserve(values.size());
    for(const auto& entry : values)
    {
        names.push_back(entry.first);
    }
    return names;
}

/// The enumerator of the string value @p name among @p table. Refuses the plan for any other
/// text.
template <typename TEnum, size_t N>
TEnum requireBwdEnum(const SavedLaunchInputs& inputs,
                     std::string_view name,
                     const std::array<std::pair<std::string_view, TEnum>, N>& table)
{
    const std::string text = requireLaunchString(inputs, name);
    std::string expected = "one of";
    for(const auto& entry : table)
    {
        if(text == entry.first)
        {
            return entry.second;
        }
        expected += " '" + std::string(entry.first) + "'";
    }
    refuseLaunchValue(inputs, name, expected, "'" + text + "'");
}

/// A double that holds a float exactly, as the problem's float members were written.
inline float requireBwdFloat(const SavedLaunchInputs& inputs,
                             std::string_view name,
                             double lowest,
                             double highest)
{
    const double value = requireLaunchDouble(inputs, name);
    if(!std::isfinite(value) || value < lowest || value > highest
       || static_cast<double>(static_cast<float>(value)) != value)
    {
        refuseLaunchValue(inputs,
                          name,
                          "a double that a float holds exactly, in [" + std::to_string(lowest)
                              + ", " + std::to_string(highest) + "]",
                          std::to_string(value));
    }
    return static_cast<float>(value);
}

/// Reads the ragged offset of every tensor into @p problem and checks it against what the
/// matcher records for a ragged graph: the offset uid is a uid or unset, the multiplier is at
/// least 1 for a ragged tensor and 0 for a dense one, a ragged tensor has no batch stride (the
/// matcher records 0, and the launch never reads it), a ragged graph has a padding mask, and
/// the stats are ragged exactly when Q is. The strides and the padding mask are read first.
inline void readBwdRagged(const SavedLaunchInputs& inputs, bwd::SdpaBwdProblem& problem)
{
    for(const auto& tensor : BWD_TENSORS)
    {
        auto& ragged = problem.ragged.*tensor.ragged;
        const auto uidName = bwdRaggedName(tensor.name, "uid");
        const auto multiplierName = bwdRaggedName(tensor.name, "multiplier");
        ragged.uid = requireLaunchInt(inputs, uidName, bwd::NO_UID, BWD_INT64_MAX);
        ragged.multiplier = requireLaunchInt(inputs, multiplierName, 0, BWD_INT64_MAX);
        if(ragged.ragged() != (ragged.multiplier != 0))
        {
            refuseLaunchValue(inputs,
                              multiplierName,
                              ragged.ragged() ? "at least 1 beside a ragged offset uid"
                                              : "0 without a ragged offset uid",
                              std::to_string(ragged.multiplier));
        }
        if(!ragged.ragged())
        {
            continue;
        }
        if(!problem.paddingMask)
        {
            refuseLaunchValue(inputs,
                              uidName,
                              "unset without " + std::string(bwdv::PADDING_MASK),
                              std::to_string(ragged.uid));
        }
        const auto& strides = problem.*tensor.strides;
        if(strides.batch != 0)
        {
            refuseLaunchValue(inputs,
                              bwdStrideName(tensor.name, "batch"),
                              "0 on a ragged tensor",
                              std::to_string(strides.batch));
        }
    }
    if(problem.ragged.stats.ragged() != problem.ragged.q.ragged())
    {
        refuseLaunchValue(inputs,
                          bwdRaggedName("stats", "uid"),
                          "set exactly when " + bwdRaggedName("q", "uid") + " is",
                          std::to_string(problem.ragged.stats.uid));
    }
}

/// The relation a contract requires between the value heads and the key heads: the standard
/// handlers serve `Hk == Hv`, the `u2` handlers `Hk != Hv`, and prep either.
enum class BwdValueHeads
{
    ANY,
    EQUAL_TO_KV,
    DIFFERENT_FROM_KV
};

/// The problem, uids and scalars of a saved launch, with every value in its range.
struct BwdCommonInputs
{
    bwd::SdpaBwdProblem problem;
    FlydslSdpaBwdUids uids;
    flydsl_sdpa_bwd_native::ScalarOperands scalars;
};

/// Copies the shared part @p common into the inputs of a launch.
template <typename TLaunch>
void takeBwdCommon(TLaunch& launch, const BwdCommonInputs& common)
{
    launch.problem = common.problem;
    launch.uids = common.uids;
    launch.scalars = common.scalars;
}

/// Reads and checks the shared part of a contract. The caller has checked the dispatch name
/// and the set of value names.
inline BwdCommonInputs readBwdCommonInputs(const SavedLaunchInputs& inputs,
                                           BwdValueHeads valueHeads)
{
    BwdCommonInputs common;
    auto& problem = common.problem;
    const auto extent = [&inputs](std::string_view name) {
        return requireLaunchInt(inputs, name, 1, BWD_INT32_MAX);
    };

    const int64_t elementBytes = requireLaunchInt(inputs, bwdv::ELEMENT_BYTES, 2, 4);
    if(elementBytes != 2 && elementBytes != 4)
    {
        refuseLaunchValue(inputs, bwdv::ELEMENT_BYTES, "2 or 4", std::to_string(elementBytes));
    }
    problem.elementBytes = static_cast<uint32_t>(elementBytes);
    problem.batch = extent(bwdv::BATCH);
    problem.numHeads = extent(bwdv::NUM_HEADS);
    problem.numKvHeads = extent(bwdv::NUM_KV_HEADS);
    problem.seqLenQ = extent(bwdv::SEQLEN_Q);
    problem.seqLenKv = extent(bwdv::SEQLEN_KV);
    problem.headDimQk = requireLaunchInt(inputs, bwdv::HEAD_DIM_QK, 1, bwd::detail::MAX_HEAD_DIM);
    problem.headDimV = requireLaunchInt(inputs, bwdv::HEAD_DIM_V, 1, bwd::detail::MAX_HEAD_DIM);
    if(problem.numHeads % problem.numKvHeads != 0)
    {
        refuseLaunchValue(inputs,
                          bwdv::NUM_KV_HEADS,
                          "a divisor of " + std::string(bwdv::NUM_HEADS) + " ("
                              + std::to_string(problem.numHeads) + ")",
                          std::to_string(problem.numKvHeads));
    }
    problem.numVHeads = extent(bwdv::NUM_V_HEADS);
    if(problem.numHeads % problem.numVHeads != 0)
    {
        refuseLaunchValue(inputs,
                          bwdv::NUM_V_HEADS,
                          "a divisor of " + std::string(bwdv::NUM_HEADS) + " ("
                              + std::to_string(problem.numHeads) + ")",
                          std::to_string(problem.numVHeads));
    }

    for(const auto& tensor : BWD_TENSORS)
    {
        common.uids.*tensor.uid = requireLaunchInt(inputs, bwdUidName(tensor.name));
        auto& strides = problem.*tensor.strides;
        const auto stride = [&](std::string_view axis) {
            return requireLaunchInt(inputs, bwdStrideName(tensor.name, axis), 0, BWD_INT64_MAX);
        };
        strides.batch = stride("batch");
        strides.head = stride("head");
        strides.seq = stride("seq");
    }

    problem.scaleSource = requireBwdEnum(inputs, bwdv::SCALE_SOURCE, BWD_SCALE_SOURCES);
    problem.scaleValue = requireBwdFloat(
        inputs, bwdv::SCALE_VALUE, -bwd::detail::MAX_SOFTMAX_SCALE, bwd::detail::MAX_SOFTMAX_SCALE);
    if(problem.scaleSource == bwd::ScaleSource::TENSOR)
    {
        common.scalars.scale = requireScalarBinding(inputs, bwdv::SCALE_ROLE);
        problem.scaleUid = common.scalars.scale->uid;
    }

    problem.masked = requireLaunchBool(inputs, bwdv::MASKED);
    problem.alignBottomRight = requireLaunchBool(inputs, bwdv::ALIGN_BOTTOM_RIGHT);
    problem.diagonalOffset
        = requireLaunchInt(inputs, bwdv::DIAGONAL_OFFSET, -BWD_INT32_MAX, BWD_INT32_MAX);
    problem.windowLeft = requireLaunchInt(inputs, bwdv::WINDOW_LEFT, bwd::NO_WINDOW, BWD_INT64_MAX);
    problem.windowRight
        = requireLaunchInt(inputs, bwdv::WINDOW_RIGHT, bwd::NO_WINDOW, BWD_INT64_MAX);
    if(problem.windowLeft == 0)
    {
        refuseLaunchValue(inputs, bwdv::WINDOW_LEFT, "-1 or at least 1", "0");
    }

    problem.paddingMask = requireLaunchBool(inputs, bwdv::PADDING_MASK);
    problem.seqLenQUid = requireLaunchInt(inputs, bwdv::SEQ_LEN_Q_UID);
    problem.seqLenKvUid = requireLaunchInt(inputs, bwdv::SEQ_LEN_KV_UID);
    const bool hasLengths = problem.seqLenQUid != bwd::NO_UID && problem.seqLenKvUid != bwd::NO_UID;
    const bool hasNoLengths
        = problem.seqLenQUid == bwd::NO_UID && problem.seqLenKvUid == bwd::NO_UID;
    if(problem.paddingMask ? !hasLengths : !hasNoLengths)
    {
        refuseLaunchValue(inputs,
                          bwdv::SEQ_LEN_Q_UID,
                          "both length uids set with " + std::string(bwdv::PADDING_MASK)
                              + " and neither without it",
                          std::to_string(problem.seqLenQUid) + " and "
                              + std::to_string(problem.seqLenKvUid));
    }

    readBwdRagged(inputs, problem);

    problem.biasUid = requireLaunchInt(inputs, bwdv::BIAS_UID);
    problem.dbiasUid = requireLaunchInt(inputs, bwdv::DBIAS_UID);
    if(problem.dbiasUid != bwd::NO_UID && problem.biasUid == bwd::NO_UID)
    {
        refuseLaunchValue(inputs,
                          bwdv::DBIAS_UID,
                          "unset when there is no bias",
                          std::to_string(problem.dbiasUid));
    }
    for(const auto& [name, strides] :
        {std::pair<std::string_view, bwd::Strides*>{"bias", &problem.biasStrides},
         std::pair<std::string_view, bwd::Strides*>{"dbias", &problem.dbiasStrides}})
    {
        strides->batch = requireLaunchInt(inputs, bwdStrideName(name, "batch"), 0, BWD_INT64_MAX);
        strides->head = requireLaunchInt(inputs, bwdStrideName(name, "head"), 0, BWD_INT64_MAX);
        strides->seq = requireLaunchInt(inputs, bwdStrideName(name, "seq"), 0, BWD_INT64_MAX);
    }
    problem.biasBroadcastBatch = requireLaunchBool(inputs, bwdv::BIAS_BROADCAST_BATCH);
    problem.biasBroadcastHead = requireLaunchBool(inputs, bwdv::BIAS_BROADCAST_HEAD);

    // The bias's own width: 2 or 4, the data's when there is no bias, and 4 beside 4-byte data
    // (an f32 bias is accepted only beside a 16-bit Q).
    const int64_t biasElementBytes = requireLaunchInt(inputs, bwdv::BIAS_ELEMENT_BYTES, 2, 4);
    if(biasElementBytes != 2 && biasElementBytes != 4)
    {
        refuseLaunchValue(
            inputs, bwdv::BIAS_ELEMENT_BYTES, "2 or 4", std::to_string(biasElementBytes));
    }
    if(problem.biasUid == bwd::NO_UID
       && biasElementBytes != static_cast<int64_t>(problem.elementBytes))
    {
        refuseLaunchValue(inputs,
                          bwdv::BIAS_ELEMENT_BYTES,
                          "equal to " + std::string(bwdv::ELEMENT_BYTES) + " ("
                              + std::to_string(problem.elementBytes) + ") when there is no bias",
                          std::to_string(biasElementBytes));
    }
    if(problem.biasUid != bwd::NO_UID && problem.elementBytes == 4 && biasElementBytes != 4)
    {
        refuseLaunchValue(inputs,
                          bwdv::BIAS_ELEMENT_BYTES,
                          "4 beside 4-byte data",
                          std::to_string(biasElementBytes));
    }
    problem.biasElementBytes = static_cast<uint32_t>(biasElementBytes);

    problem.dropoutP = requireBwdFloat(inputs, bwdv::DROPOUT_P, 0.0, 1.0);
    if(problem.dropoutP >= 1.0F)
    {
        refuseLaunchValue(inputs, bwdv::DROPOUT_P, "in [0, 1)", std::to_string(problem.dropoutP));
    }
    problem.seedWidth = requireBwdEnum(inputs, bwdv::SEED_WIDTH, BWD_SEED_WIDTHS);
    problem.seedForm = requireBwdEnum(inputs, bwdv::SEED_FORM, BWD_SEED_FORMS);
    if((problem.dropoutP > 0.0F) != (problem.seedForm != bwd::SeedForm::NONE)
       || (problem.seedForm == bwd::SeedForm::NONE) != (problem.seedWidth == bwd::SeedWidth::NONE))
    {
        refuseLaunchValue(
            inputs,
            bwdv::SEED_FORM,
            "set exactly when " + std::string(bwdv::DROPOUT_P) + " is above 0, with a seed width",
            std::string(bwdEnumName(BWD_SEED_FORMS, problem.seedForm)) + " with width "
                + std::string(bwdEnumName(BWD_SEED_WIDTHS, problem.seedWidth)));
    }
    // The kernels read 8 bytes at a device seed or offset, so the matcher declines INT32 there.
    if(problem.seedForm == bwd::SeedForm::DEVICE && problem.seedWidth == bwd::SeedWidth::INT32)
    {
        refuseLaunchValue(inputs,
                          bwdv::SEED_WIDTH,
                          "int64 beside a device seed (the kernels read 8 bytes there)",
                          "int32");
    }
    if(problem.seedForm == bwd::SeedForm::PASS_BY_VALUE)
    {
        common.scalars.seed = requireScalarBinding(inputs, bwdv::SEED_ROLE);
        common.scalars.offset = requireScalarBinding(inputs, bwdv::OFFSET_ROLE);
        problem.seedUid = common.scalars.seed->uid;
        problem.offsetUid = common.scalars.offset->uid;
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        const DataType width
            = problem.seedWidth == bwd::SeedWidth::INT64 ? DataType::INT64 : DataType::INT32;
        for(const auto& [role, operand] :
            {std::pair<std::string_view, const hipdnn_plugin_sdk::ScalarOperand*>{
                 bwdv::SEED_ROLE, &*common.scalars.seed},
             std::pair<std::string_view, const hipdnn_plugin_sdk::ScalarOperand*>{
                 bwdv::OFFSET_ROLE, &*common.scalars.offset}})
        {
            if(operand->dataType != width)
            {
                refuseLaunchValue(
                    inputs,
                    std::string(role) + ".data_type",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(width),
                    hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(operand->dataType));
            }
        }
    }
    else if(problem.seedForm == bwd::SeedForm::DEVICE)
    {
        problem.seedUid = requireLaunchInt(inputs, bwdv::SEED_UID);
        problem.offsetUid = requireLaunchInt(inputs, bwdv::OFFSET_UID);
        if(problem.seedUid == bwd::NO_UID || problem.offsetUid == bwd::NO_UID)
        {
            refuseLaunchValue(inputs, bwdv::SEED_UID, "a tensor uid", "unset");
        }
    }
    // Which V-head count the contract's handler serves, then the rules every one shares.
    if(valueHeads == BwdValueHeads::EQUAL_TO_KV && problem.numVHeads != problem.numKvHeads)
    {
        refuseLaunchValue(inputs,
                          bwdv::NUM_V_HEADS,
                          "equal to " + std::string(bwdv::NUM_KV_HEADS) + " ("
                              + std::to_string(problem.numKvHeads) + ") for this contract",
                          std::to_string(problem.numVHeads));
    }
    if(valueHeads == BwdValueHeads::DIFFERENT_FROM_KV && problem.numVHeads == problem.numKvHeads)
    {
        refuseLaunchValue(inputs,
                          bwdv::NUM_V_HEADS,
                          "different from " + std::string(bwdv::NUM_KV_HEADS)
                              + " for this contract",
                          std::to_string(problem.numVHeads));
    }
    if(problem.numVHeads != problem.numKvHeads
       && (problem.elementBytes == 4 || problem.biasUid != bwd::NO_UID || problem.ragged.any()))
    {
        refuseLaunchValue(inputs,
                          bwdv::NUM_V_HEADS,
                          "equal to " + std::string(bwdv::NUM_KV_HEADS)
                              + " with 4-byte data, a bias or a ragged tensor",
                          std::to_string(problem.numVHeads));
    }
    return common;
}

/// Reads the two enumerations that decide which value names a plan may hold.
inline std::vector<std::string>
    bwdAllowedValueNames(const SavedLaunchInputs& inputs,
                         std::initializer_list<std::string_view> rowNames)
{
    const auto scale = requireBwdEnum(inputs, bwdv::SCALE_SOURCE, BWD_SCALE_SOURCES);
    const auto form = requireBwdEnum(inputs, bwdv::SEED_FORM, BWD_SEED_FORMS);
    auto names = bwdCommonValueNames(scale == bwd::ScaleSource::TENSOR, form);
    for(const auto name : rowNames)
    {
        names.emplace_back(name);
    }
    return names;
}

inline std::string requireBwdGrid(const SavedLaunchInputs& inputs,
                                  std::initializer_list<std::string_view> accepted)
{
    std::string grid = requireLaunchString(inputs, bwdv::GRID);
    std::string expected = "one of";
    for(const auto candidate : accepted)
    {
        if(grid == candidate)
        {
            return grid;
        }
        expected += " '" + std::string(candidate) + "'";
    }
    refuseLaunchValue(inputs, bwdv::GRID, expected, "'" + grid + "'");
}

inline int64_t requireBwdHeadDim(const SavedLaunchInputs& inputs)
{
    return requireLaunchInt(inputs, bwdv::HEAD_DIM, 1, bwd::detail::MAX_HEAD_DIM);
}

inline int64_t requireBwdBlockThreads(const SavedLaunchInputs& inputs)
{
    return requireLaunchInt(inputs, bwdv::BLOCK_THREADS, 1, 1024);
}

inline int64_t requireBwdTileRows(const SavedLaunchInputs& inputs, std::string_view name)
{
    return requireLaunchInt(inputs, name, 1, BWD_INT32_MAX);
}

/// Reads the values of a `dkdv` contract, whose dispatch name is @p contract.
inline FlydslSdpaBwdDkdvLaunchInputs readBwdDkdvInputs(const SavedLaunchInputs& inputs,
                                                       std::string_view contract,
                                                       BwdValueHeads valueHeads)
{
    requireLaunchContract(inputs, contract);
    requireOnlyLaunchValues(
        inputs,
        bwdAllowedValueNames(inputs,
                             {bwdv::HEAD_DIM, bwdv::BLOCK_THREADS, bwdv::BLOCK_KV, bwdv::GRID}));
    const auto common = readBwdCommonInputs(inputs, valueHeads);
    FlydslSdpaBwdDkdvLaunchInputs launch;
    takeBwdCommon(launch, common);
    launch.headDim = requireBwdHeadDim(inputs);
    launch.blockThreads = requireBwdBlockThreads(inputs);
    launch.blockKv = requireBwdTileRows(inputs, bwdv::BLOCK_KV);
    launch.grid = requireBwdGrid(
        inputs, {FLYDSL_SDPA_BWD_GRID_DKDV_HEADS_FIRST, FLYDSL_SDPA_BWD_GRID_DKDV_BLOCKS_FIRST});
    return launch;
}

/// Reads the values of a `dq` contract, whose dispatch name is @p contract.
inline FlydslSdpaBwdDqLaunchInputs readBwdDqInputs(const SavedLaunchInputs& inputs,
                                                   std::string_view contract,
                                                   BwdValueHeads valueHeads)
{
    requireLaunchContract(inputs, contract);
    requireOnlyLaunchValues(
        inputs,
        bwdAllowedValueNames(inputs,
                             {bwdv::HEAD_DIM, bwdv::BLOCK_THREADS, bwdv::BLOCK_M, bwdv::GRID}));
    const auto common = readBwdCommonInputs(inputs, valueHeads);
    FlydslSdpaBwdDqLaunchInputs launch;
    takeBwdCommon(launch, common);
    launch.headDim = requireBwdHeadDim(inputs);
    launch.blockThreads = requireBwdBlockThreads(inputs);
    launch.blockM = requireBwdTileRows(inputs, bwdv::BLOCK_M);
    launch.grid = requireBwdGrid(inputs, {FLYDSL_SDPA_BWD_GRID_DQ});
    return launch;
}

} // namespace detail

/// The values of the `hipkernel.flydsl_sdpa_bwd_split.prep.dispatch.v1` contract: the shared
/// problem, uid, stride and scalar values, plus the row's `head_dim` and `grid`. Doubles hold
/// floats exactly. Enumerations are lower-case words (`scale_source` `value` or `tensor`,
/// `seed_width` `none`, `int32` or `int64`, `seed_form` `none`, `device` or `pass_by_value`).
/// Each tensor saves its ragged offset as a uid (-1 when dense) and a multiplier (0 when dense),
/// and the stats save their three element strides. A pass-by-value scalar saves the bindings of
/// IngestorScalarBindings.hpp under the roles `scale`, `seed` and `offset`, and never a value
/// the user supplies at execute.
inline hipdnn_plugin_sdk::ingestor::MetadataValues
    flydslSdpaBwdPrepLaunchValues(const FlydslSdpaBwdPrepLaunchInputs& launch)
{
    namespace v = flydsl_sdpa_bwd_value;
    hipdnn_plugin_sdk::ingestor::MetadataValues values;
    detail::writeBwdCommonValues(values, launch.problem, launch.uids, launch.scalars);
    values[std::string(v::HEAD_DIM)] = launch.headDim;
    values[std::string(v::GRID)] = launch.grid;
    return values;
}

/// Reads the values of the `...prep.dispatch.v1` contract. Extents must be in [1, INT32_MAX],
/// head dims in [1, 512], the element width 2 or 4, strides non-negative, each ragged offset a
/// uid or -1 with a multiplier of at least 1 exactly when it is a uid, only with a padding mask,
/// on a tensor whose batch stride is 0, and on the stats exactly when on Q, the window bounds in
/// [-1, INT64_MAX], the dropout probability in [0, 1) with a seed exactly when it is above 0,
/// and `grid` the prep launch order.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
inline FlydslSdpaBwdPrepLaunchInputs
    readFlydslSdpaBwdPrepLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    namespace v = flydsl_sdpa_bwd_value;
    requireLaunchContract(inputs, FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1);
    requireOnlyLaunchValues(inputs, detail::bwdAllowedValueNames(inputs, {v::HEAD_DIM, v::GRID}));
    const auto common = detail::readBwdCommonInputs(inputs, detail::BwdValueHeads::ANY);
    FlydslSdpaBwdPrepLaunchInputs launch;
    detail::takeBwdCommon(launch, common);
    launch.headDim = detail::requireBwdHeadDim(inputs);
    launch.grid = detail::requireBwdGrid(inputs, {FLYDSL_SDPA_BWD_GRID_PREP});
    return launch;
}

/// The values of the `hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch.v1` contract: the shared
/// values, plus the row's `head_dim`, `block_threads`, `block_kv` and `grid`.
inline hipdnn_plugin_sdk::ingestor::MetadataValues
    flydslSdpaBwdDkdvLaunchValues(const FlydslSdpaBwdDkdvLaunchInputs& launch)
{
    namespace v = flydsl_sdpa_bwd_value;
    hipdnn_plugin_sdk::ingestor::MetadataValues values;
    detail::writeBwdCommonValues(values, launch.problem, launch.uids, launch.scalars);
    values[std::string(v::HEAD_DIM)] = launch.headDim;
    values[std::string(v::BLOCK_THREADS)] = launch.blockThreads;
    values[std::string(v::BLOCK_KV)] = launch.blockKv;
    values[std::string(v::GRID)] = launch.grid;
    return values;
}

/// Reads the values of the `...dkdv.dispatch.v1` contract. The shared checks of the prep
/// reader apply, with V heads equal to K's; `block_threads` must be in [1, 1024], `block_kv` in
/// [1, INT32_MAX], and `grid` a dkdv launch order.
inline FlydslSdpaBwdDkdvLaunchInputs
    readFlydslSdpaBwdDkdvLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdDkdvInputs(
        inputs, FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1, detail::BwdValueHeads::EQUAL_TO_KV);
}

/// Reads the values of the `...dkdv_u2.dispatch.v1` contract: the values of `dkdv`, with V
/// heads that differ from K's. The dV-only launch has the dK-only launch's block and `grid`
/// spelling over the V heads, so it adds no value.
inline FlydslSdpaBwdDkdvLaunchInputs readFlydslSdpaBwdDkdvU2LaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdDkdvInputs(inputs,
                                     FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1,
                                     detail::BwdValueHeads::DIFFERENT_FROM_KV);
}

/// The values of the `hipkernel.flydsl_sdpa_bwd_split.dq.dispatch.v1` contract: the shared
/// values, plus the row's `head_dim`, `block_threads`, `block_m` and `grid`. The `dq_reduce` and
/// `dq_u2` contracts have the same values.
inline hipdnn_plugin_sdk::ingestor::MetadataValues
    flydslSdpaBwdDqLaunchValues(const FlydslSdpaBwdDqLaunchInputs& launch)
{
    namespace v = flydsl_sdpa_bwd_value;
    hipdnn_plugin_sdk::ingestor::MetadataValues values;
    detail::writeBwdCommonValues(values, launch.problem, launch.uids, launch.scalars);
    values[std::string(v::HEAD_DIM)] = launch.headDim;
    values[std::string(v::BLOCK_THREADS)] = launch.blockThreads;
    values[std::string(v::BLOCK_M)] = launch.blockM;
    values[std::string(v::GRID)] = launch.grid;
    return values;
}

/// Reads the values of the `...dq.dispatch.v1` contract. The shared checks of the prep reader
/// apply, with V heads equal to K's; `block_threads` must be in [1, 1024], `block_m` in
/// [1, INT32_MAX], and `grid` the dq launch order.
inline FlydslSdpaBwdDqLaunchInputs
    readFlydslSdpaBwdDqLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdDqInputs(
        inputs, FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1, detail::BwdValueHeads::EQUAL_TO_KV);
}

/// Reads the values of the `...dq_reduce.dispatch.v1` contract: the values of `dq`.
inline FlydslSdpaBwdDqLaunchInputs readFlydslSdpaBwdDqReduceLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdDqInputs(inputs,
                                   FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1,
                                   detail::BwdValueHeads::EQUAL_TO_KV);
}

/// Reads the values of the `...dq_u2.dispatch.v1` contract: the values of `dq`, with V heads
/// that differ from K's.
inline FlydslSdpaBwdDqLaunchInputs
    readFlydslSdpaBwdDqU2LaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdDqInputs(inputs,
                                   FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1,
                                   detail::BwdValueHeads::DIFFERENT_FROM_KV);
}

/// The values of the `hipkernel.flydsl_sdpa_bwd_merged.dispatch.v1` contract: the shared
/// values, plus the row's `head_dim`, `block_threads`, `block_kv` and `block_m`. The
/// `merged_reduce` contract has the same values.
inline hipdnn_plugin_sdk::ingestor::MetadataValues
    flydslSdpaBwdMergedLaunchValues(const FlydslSdpaBwdMergedLaunchInputs& launch)
{
    namespace v = flydsl_sdpa_bwd_value;
    hipdnn_plugin_sdk::ingestor::MetadataValues values;
    detail::writeBwdCommonValues(values, launch.problem, launch.uids, launch.scalars);
    values[std::string(v::HEAD_DIM)] = launch.headDim;
    values[std::string(v::BLOCK_THREADS)] = launch.blockThreads;
    values[std::string(v::BLOCK_KV)] = launch.blockKv;
    values[std::string(v::BLOCK_M)] = launch.blockM;
    return values;
}

namespace detail
{

/// Reads the values of a `merged` contract, whose dispatch name is @p contract. The shared
/// checks of the prep reader apply, with the element width 2 and a bias of the data's width (the
/// merged engine serves 16-bit data and bias only) and V heads equal to K's; `block_threads`
/// must be in [1, 1024], `block_kv` and `block_m` in [1, INT32_MAX].
inline FlydslSdpaBwdMergedLaunchInputs readBwdMergedInputs(const SavedLaunchInputs& inputs,
                                                           std::string_view contract)
{
    namespace v = flydsl_sdpa_bwd_value;
    requireLaunchContract(inputs, contract);
    requireOnlyLaunchValues(
        inputs,
        bwdAllowedValueNames(inputs, {v::HEAD_DIM, v::BLOCK_THREADS, v::BLOCK_KV, v::BLOCK_M}));
    const auto common = readBwdCommonInputs(inputs, BwdValueHeads::EQUAL_TO_KV);
    const auto& problem = common.problem;
    if(problem.elementBytes != 2)
    {
        refuseLaunchValue(inputs,
                          v::ELEMENT_BYTES,
                          "2 (the merged engine serves 16-bit data only)",
                          std::to_string(problem.elementBytes));
    }
    if(problem.biasElementBytes != problem.elementBytes)
    {
        refuseLaunchValue(inputs,
                          v::BIAS_ELEMENT_BYTES,
                          "2 (the merged engine serves a 16-bit bias only)",
                          std::to_string(problem.biasElementBytes));
    }
    FlydslSdpaBwdMergedLaunchInputs launch;
    takeBwdCommon(launch, common);
    launch.headDim = requireBwdHeadDim(inputs);
    launch.blockThreads = requireBwdBlockThreads(inputs);
    launch.blockKv = requireBwdTileRows(inputs, v::BLOCK_KV);
    launch.blockM = requireBwdTileRows(inputs, v::BLOCK_M);
    return launch;
}

} // namespace detail

/// Reads the values of the `...merged.dispatch.v1` contract.
inline FlydslSdpaBwdMergedLaunchInputs readFlydslSdpaBwdMergedLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdMergedInputs(inputs, FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1);
}

/// Reads the values of the `...merged_reduce.dispatch.v1` contract: the values of `merged`.
inline FlydslSdpaBwdMergedLaunchInputs readFlydslSdpaBwdMergedReduceLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs)
{
    return detail::readBwdMergedInputs(inputs, FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
