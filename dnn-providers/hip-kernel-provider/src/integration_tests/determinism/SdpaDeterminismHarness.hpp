// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "SdpaDeterminismCompare.hpp"
#include "SdpaDeterminismGraphs.hpp"

/**
 * @file SdpaDeterminismHarness.hpp
 * @brief The device half of the SDPA determinism suite: a case row (data), the run function
 *        that holds one engine, pinned through the frontend, to bitwise run-to-run
 *        reproducibility on the device it runs on, and the registration of rows as gtest cases.
 *
 * Every row is run as `Comprehensive/IntegrationGpuSdpaDeterminismFwd.RepeatsBitwise/<name>`
 * (or `...Bwd`), or, for a MEASURE_ONLY row, as
 * `Measure/IntegrationGpuSdpaDeterminismMeasure.DISABLED_RepeatsBitwise/<name>`, which no tier
 * selects and which reports instead of failing. A row's name is `<Engine>_<Case>_<archs>`: the
 * last token lists the devices the row runs on (`mi308x` for a `Sku::MI308` row).
 *
 * Per launch the run function poisons every output and the workspace, launches, requires every
 * logical output element written and finite, and requires every logical element to equal
 * launch 0 byte for byte. A final launch with one Q element changed must change an output, so a
 * row whose outputs ignore its inputs cannot pass.
 */
namespace hip_kernel_provider::sdpa_determinism
{

using Bytes = std::vector<uint8_t>;

/// PROVEN rows are expected to pass and run in the Comprehensive tier. MEASURE_ONLY rows are
/// expected to fail the run-to-run check (a known hazard), run only by hand and report.
enum class Disposition
{
    PROVEN,
    MEASURE_ONLY,
};

/// Which SKUs of the row's architecture run it: ASM forward loads other code objects on an
/// MI308X device (`MI308/<co>`) than on any other gfx942 device (`MI300/<co>`).
enum class Sku
{
    ANY,
    MI300,
    MI308,
};

/// What a knob of the row changes: STRUCTURE changes the launch set or a cross-workgroup
/// combine; TILE_SCHEDULE changes neither.
enum class KnobClass
{
    STRUCTURE,
    TILE_SCHEDULE,
};

enum class Direction
{
    FORWARD,
    BACKWARD,
};

/// The integer knob settings an engine is pinned with, as (name, value). String knobs are
/// pinned by ordinal.
using KnobList = std::vector<std::pair<std::string, int64_t>>;

/// Launches of a PROVEN row (the FlyDSL plugin-level harness uses the same count).
inline constexpr size_t REPEAT_LAUNCHES = 20;
/// Launches of a MEASURE_ONLY row.
inline constexpr size_t MEASURE_LAUNCHES = 50;

/// The fixed part of a row.
struct RowMeta
{
    /// The engine the row pins, by name: `ASM_SDPA_ENGINE`, `hipkernel:Gfx950AttentionDense`.
    std::string engine;
    /// The row's case token: letters and digits only (`Bf16Hd128`).
    std::string caseName;
    /// The device architectures the row runs on (`gfx942`); the harness skips any other.
    std::vector<std::string> archs;
    Sku sku = Sku::ANY;
    Disposition disposition = Disposition::PROVEN;
    KnobClass knobClass = KnobClass::STRUCTURE;
    /// The knob settings the engine is pinned with.
    KnobList knobs;
    /// Whether the LSE may hold infinity (the LSE of a row no key reaches, SEM-008). A row that
    /// sets it says why where it is written.
    bool allowInfinity = false;

    RowMeta() = default;

    /// The parameters are the members in order; every one after archs may be omitted.
    RowMeta(std::string engineName,
            std::string caseToken,
            std::vector<std::string> runArchs,
            Sku skuFilter = Sku::ANY,
            Disposition rowDisposition = Disposition::PROVEN,
            KnobClass rowKnobClass = KnobClass::STRUCTURE,
            KnobList pinKnobs = {},
            bool infinityAllowed = false)
        : engine(std::move(engineName))
        , caseName(std::move(caseToken))
        , archs(std::move(runArchs))
        , sku(skuFilter)
        , disposition(rowDisposition)
        , knobClass(rowKnobClass)
        , knobs(std::move(pinKnobs))
        , allowInfinity(infinityAllowed)
    {
    }
};

/// What a launch writes into one output tensor, and which of its elements are logical.
struct OutputSpec
{
    std::string name;
    ElementType type = ElementType::FLOAT32;
    LiveRuns live;
    bool allowInfinity = false;
};

/// A built graph with what the run function needs to fill, launch and check it.
struct PreparedGraph
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    /// Every non-virtual output of the graph, by uid.
    std::map<int64_t, OutputSpec> outputs;
    /// Inputs written with these exact bytes (lengths, offsets, a Philox seed, descales).
    std::map<int64_t, Bytes> fixedInputs;
    /// Inputs that are no node's operand but belong in the variant pack (the offset tables).
    std::vector<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>> extraInputs;
    /// Every other input is filled with finite values in [-inputBound, inputBound].
    float inputBound = 1.0F;
    /// The input one element of which changes for the sensitivity launch, and that element.
    int64_t sensitivityUid = 0;
    size_t sensitivityElement = 0;
    /// Fills what the bounded fill cannot (the backward graph's O and LSE) after the inputs
    /// are filled; returns an error message, empty on success.
    std::function<std::string(
        hipdnnHandle_t, hipStream_t, hipdnn_test_sdk::utilities::GraphTensorBundle&)>
        deriveInputs;
};

struct DeterminismRow;
using GraphBuilder = std::function<PreparedGraph(const DeterminismRow&)>;

struct DeterminismRow
{
    RowMeta meta;
    /// `<Engine>_<Case>_<archs>`.
    std::string name;
    Direction direction = Direction::FORWARD;
    GraphBuilder build;
};

namespace detail
{

inline std::string joinWith(const std::vector<std::string>& parts, const std::string& separator)
{
    std::string joined;
    for(const auto& part : parts)
    {
        joined += (joined.empty() ? "" : separator) + part;
    }
    return joined;
}

/// The engine's name without the `hipkernel:` namespace of the ingestor engines.
inline std::string engineToken(const std::string& engine)
{
    const std::string prefix = "hipkernel:";
    return engine.rfind(prefix, 0) == 0 ? engine.substr(prefix.size()) : engine;
}

inline std::string rowName(const RowMeta& meta)
{
    const std::string archToken
        = meta.sku == Sku::MI308 ? std::string("mi308x") : joinWith(meta.archs, "_");
    return engineToken(meta.engine) + "_" + meta.caseName + "_" + archToken;
}

template <typename Integer>
Bytes bytesOf(const std::vector<Integer>& values)
{
    Bytes bytes(values.size() * sizeof(Integer));
    if(!bytes.empty())
    {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

template <typename Integer>
Bytes bytesOf(Integer value)
{
    return bytesOf(std::vector<Integer>{value});
}

/// The live runs of a forward O (token of Hq * Dv elements) or LSE (token of Hq elements)
/// under @p spec: dense, padded batches (positions past a batch's length are dead) or packed THD
/// (only a sequence's own tokens are live).
inline LiveRuns forwardLiveRuns(const ingestor::FrontendSdpaSpec& spec,
                                const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides,
                                size_t tokenElements)
{
    using namespace ingestor::frontend_sdpa_detail;
    if(!spec.packedLensQ.empty())
    {
        auto origins = spec.packedOriginsQ;
        if(origins.empty())
        {
            origins = tokenOffsets(spec.packedLensQ);
            origins.pop_back();
        }
        const auto& lengths = spec.seqLensQ.empty() ? spec.packedLensQ : spec.seqLensQ;
        return packedRuns(origins, lengths, tokenElements);
    }
    if(!spec.seqLensQ.empty())
    {
        return logicalRuns(dims, strides, [&](const std::vector<int64_t>& index) {
            return index[2] < spec.seqLensQ[static_cast<size_t>(index[0])];
        });
    }
    return logicalRuns(dims, strides);
}

/// The live runs of a forward RNG dump [B, Hq, Sq, Skv]: the positions a live row and a live
/// key address.
inline LiveRuns rngDumpLiveRuns(const ingestor::FrontendSdpaSpec& spec,
                                const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides)
{
    return logicalRuns(dims, strides, [&](const std::vector<int64_t>& index) {
        const auto [liveQ, liveKv] = ingestor::frontend_sdpa_detail::liveLengths(spec, index[0]);
        return index[2] < liveQ && index[3] < liveKv;
    });
}

/// Whether @p deviceId is an MI308X, read as the ASM forward plan builder reads it.
inline bool isMi308Device(int deviceId)
{
    int chipId = 0;
    if(hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, deviceId) != hipSuccess)
    {
        throw std::runtime_error("hipDeviceGetAttribute(PciChipId) failed");
    }
    return chipId == 0x74a2 || chipId == 0x74a8 || chipId == 0x74b6 || chipId == 0x74bc;
}

} // namespace detail

/// A forward row's graph, with its live elements, fixed inputs and sensitivity element, from
/// a graph the frontend builders made. @p descales are the FP8 graph's scalar descale tensors
/// and their values (null tensors when the graph has none).
inline PreparedGraph prepareForward(
    const ingestor::FrontendSdpaGraph& built,
    hipdnn_frontend::DataType oType,
    const std::array<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>, 3>& descales,
    const std::array<float, 3>& descaleValues,
    const DeterminismRow& row)
{
    using namespace ingestor::frontend_sdpa_detail;
    const auto& spec = built.spec;
    PreparedGraph prepared;
    prepared.graph = built.graph;

    const int64_t valueHeadSize = spec.valueHeadSize.value_or(spec.headSize);
    prepared.outputs[O_UID]
        = {"O",
           elementTypeOf(oType),
           detail::forwardLiveRuns(spec,
                                   built.output->get_dim(),
                                   built.output->get_stride(),
                                   static_cast<size_t>(spec.queryHeads * valueHeadSize)),
           false};
    if(built.stats != nullptr)
    {
        prepared.outputs[STATS_UID]
            = {"LSE",
               ElementType::FLOAT32,
               detail::forwardLiveRuns(spec,
                                       built.stats->get_dim(),
                                       built.stats->get_stride(),
                                       static_cast<size_t>(spec.queryHeads)),
               row.meta.allowInfinity};
    }
    if(built.rngDump != nullptr)
    {
        prepared.outputs[RNG_DUMP_UID]
            = {"RngDump",
               ElementType::FLOAT32,
               detail::rngDumpLiveRuns(spec, built.rngDump->get_dim(), built.rngDump->get_stride()),
               false};
    }

    if(!spec.seqLensQ.empty())
    {
        prepared.fixedInputs[SEQ_LEN_Q_UID] = detail::bytesOf(spec.seqLensQ);
    }
    if(!spec.seqLensKv.empty())
    {
        prepared.fixedInputs[SEQ_LEN_KV_UID] = detail::bytesOf(spec.seqLensKv);
    }
    if(!spec.packedLensQ.empty())
    {
        prepared.fixedInputs[OFFSETS_Q_UID]
            = detail::bytesOf(tokenTable(spec.packedOriginsQ, spec.packedLensQ));
        prepared.fixedInputs[OFFSETS_KV_UID]
            = detail::bytesOf(tokenTable(spec.packedOriginsKv, spec.packedLensKv));
        prepared.extraInputs = {built.offsetsQ, built.offsetsKv};
    }
    if(spec.dropout.has_value())
    {
        prepared.fixedInputs[SEED_UID] = detail::bytesOf(spec.dropout->seed);
        prepared.fixedInputs[OFFSET_UID] = detail::bytesOf(spec.dropout->offset);
    }
    const std::array<int64_t, 3> descaleUids = {DESCALE_Q_UID, DESCALE_K_UID, DESCALE_V_UID};
    for(size_t i = 0; i < descales.size(); ++i)
    {
        if(descales[i] == nullptr)
        {
            continue;
        }
        const auto type = elementTypeOf(descales[i]->get_data_type());
        Bytes value(elementBytes(type));
        encodeElement(type, descaleValues[i], value.data());
        prepared.fixedInputs[descaleUids[i]] = std::move(value);
    }

    // The last live query row of batch 0 sees keys (the first row of a causal graph sees one
    // key, so its output ignores Q), at head 0 and element 0 of the head. A top-left band whose
    // left bound is at most Sq - Skv leaves the last rows without keys; there the probe is row
    // Skv - 1, which sees the bound's keys.
    const auto [liveQ, liveKv] = liveLengths(spec, 0);
    int64_t probeRow = std::max<int64_t>(liveQ, 1) - 1;
    if(spec.alignment == hipdnn_frontend::DiagonalAlignment::TOP_LEFT && spec.leftBound.has_value()
       && liveQ - liveKv >= *spec.leftBound)
    {
        probeRow = liveKv - 1;
    }
    int64_t firstToken = 0;
    if(!spec.packedLensQ.empty())
    {
        firstToken = tokenTable(spec.packedOriginsQ, spec.packedLensQ).front();
    }
    prepared.sensitivityUid = Q_UID;
    prepared.sensitivityElement
        = static_cast<size_t>((firstToken + probeRow) * spec.queryHeads * spec.headSize);
    return prepared;
}

/// A forward row on a 16-bit graph: the frontend builder's own tensors and features.
inline DeterminismRow forwardRow(RowMeta meta, ingestor::FrontendSdpaSpec spec)
{
    DeterminismRow row;
    row.name = detail::rowName(meta);
    row.meta = std::move(meta);
    row.direction = Direction::FORWARD;
    row.build = [spec = std::move(spec)](const DeterminismRow& self) {
        return prepareForward(ingestor::buildFrontendSdpaGraph(spec), spec.dataType, {}, {}, self);
    };
    return row;
}

/// A forward row on a graph the 16-bit builder cannot build: the ASM FP8 graph and the FLOAT
/// graphs.
inline DeterminismRow forwardRow(RowMeta meta, DeterminismFwdSpec spec)
{
    DeterminismRow row;
    row.name = detail::rowName(meta);
    row.meta = std::move(meta);
    row.direction = Direction::FORWARD;
    row.build = [spec = std::move(spec)](const DeterminismRow& self) {
        const auto built = buildDeterminismFwdGraph(spec);
        return prepareForward(built.sdpa, spec.dtypes.oType, built.descales, spec.descales, self);
    };
    return row;
}

/// One SDPA backward graph: Q, K, V, O, dO of @p dataType (the LSE is FLOAT [B, Hq, Sq, 1]),
/// producing dQ, dK, dV in the same type, all stored as @p layout. Q, K, V and dO are bounded
/// inputs; O and the LSE come from one forward launch of the same device, so they are
/// consistent with Q, K and V.
struct DeterminismBwdSpec
{
    std::string name = "sdpa_bwd";
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::BFLOAT16;
    int64_t batch = 1;
    int64_t queryHeads = 1;
    int64_t kvHeads = 1;
    int64_t seqQ = 256;
    int64_t seqKv = 256;
    int64_t headSize = 128;
    /// The band's bounds; unset is unbounded. A right bound of 0 is causal.
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    hipdnn_frontend::DiagonalAlignment alignment = hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT;
    kernel_ingestor_engine::testing::SdpaLayout layout
        = kernel_ingestor_engine::testing::SdpaLayout::BHSD;
    /// The scale; 1 / sqrt(headSize) when unset.
    std::optional<float> scale;
};

namespace bwd_uid
{
inline constexpr int64_t Q = 1;
inline constexpr int64_t K = 2;
inline constexpr int64_t V = 3;
inline constexpr int64_t O = 4;
inline constexpr int64_t DO = 5;
inline constexpr int64_t STATS = 6;
inline constexpr int64_t DQ = 7;
inline constexpr int64_t DK = 8;
inline constexpr int64_t DV = 9;
} // namespace bwd_uid

namespace detail
{

/// The forward launch that gives a backward graph its O and LSE: the same Q, K and V (in the
/// forward builder's BSHD layout), served by whichever engine ranks first on this device.
inline std::string deriveBackwardOutputs(const DeterminismBwdSpec& spec,
                                         hipdnnHandle_t handle,
                                         hipStream_t stream,
                                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
{
    using Layout = kernel_ingestor_engine::testing::SdpaLayout;
    ingestor::FrontendSdpaSpec forward;
    forward.name = spec.name + "_forward";
    forward.dataType = spec.dataType;
    forward.batch = spec.batch;
    forward.queryHeads = spec.queryHeads;
    forward.kvHeads = spec.kvHeads;
    forward.seqQ = spec.seqQ;
    forward.seqKv = spec.seqKv;
    forward.headSize = spec.headSize;
    forward.leftBound = spec.leftBound;
    forward.rightBound = spec.rightBound;
    forward.alignment = spec.alignment;
    forward.withStats = true;
    forward.scale = spec.scale;

    const auto built = ingestor::buildFrontendSdpaGraph(forward);
    auto result = built.graph->validate();
    if(result.code != hipdnn_frontend::ErrorCode::OK)
    {
        return "the forward graph for O and LSE is invalid: " + result.err_msg;
    }
    result = built.graph->build(handle);
    if(result.code != hipdnn_frontend::ErrorCode::OK)
    {
        return "no engine serves the forward graph that gives O and LSE on this device: "
               + result.err_msg;
    }

    using namespace ingestor::frontend_sdpa_detail;
    const std::vector<int64_t> qDims = {spec.batch, spec.queryHeads, spec.seqQ, spec.headSize};
    const std::vector<int64_t> kDims = {spec.batch, spec.kvHeads, spec.seqKv, spec.headSize};
    const auto forwardStrides = [](int64_t heads, int64_t seq, int64_t headSize) {
        return kernel_ingestor_engine::testing::sdpaStrides(Layout::BSHD, heads, seq, headSize);
    };
    const auto layoutStrides = [&](int64_t heads, int64_t seq, int64_t headSize) {
        return kernel_ingestor_engine::testing::sdpaStrides(spec.layout, heads, seq, headSize);
    };

    hipdnn_test_sdk::utilities::GraphTensorBundle forwardBundle;
    const std::array<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>, 2> outputs
        = {built.output, built.stats};
    for(const auto& attr : outputs)
    {
        forwardBundle.addTensor(*attr,
                                hipdnn_test_sdk::utilities::createTensorFromAttribute(*attr));
    }
    struct Operand
    {
        int64_t forwardUid;
        int64_t backwardUid;
        const std::vector<int64_t>* dims;
        std::vector<int64_t> forwardStride;
        std::vector<int64_t> backwardStride;
    };
    const std::array<Operand, 3> operands
        = {{{Q_UID,
             bwd_uid::Q,
             &qDims,
             forwardStrides(spec.queryHeads, spec.seqQ, spec.headSize),
             layoutStrides(spec.queryHeads, spec.seqQ, spec.headSize)},
            {K_UID,
             bwd_uid::K,
             &kDims,
             forwardStrides(spec.kvHeads, spec.seqKv, spec.headSize),
             layoutStrides(spec.kvHeads, spec.seqKv, spec.headSize)},
            {V_UID,
             bwd_uid::V,
             &kDims,
             forwardStrides(spec.kvHeads, spec.seqKv, spec.headSize),
             layoutStrides(spec.kvHeads, spec.seqKv, spec.headSize)}}};
    // Q, K and V join forwardBundle from the forward graph's own tensor attributes, with the
    // backward inputs' logical values.
    std::map<int64_t, std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>> operandAttributes;
    built.graph->visit([&](const hipdnn_frontend::graph::INode& node) {
        for(const auto& attr : node.getNodeInputTensorAttributes())
        {
            operandAttributes.emplace(attr->get_uid(), attr);
        }
    });
    for(const auto& operand : operands)
    {
        const auto it = operandAttributes.find(operand.forwardUid);
        if(it == operandAttributes.end())
        {
            return "the forward graph has no operand with uid "
                   + std::to_string(operand.forwardUid);
        }
        forwardBundle.addTensor(*it->second,
                                hipdnn_test_sdk::utilities::createTensorFromAttribute(*it->second));
        auto& to = forwardBundle.getTensor(operand.forwardUid);
        auto& from = bundle.getTensor(operand.backwardUid);
        copyLogical(to.elementSize(),
                    from.rawHostData(),
                    operand.backwardStride,
                    to.rawHostData(),
                    operand.forwardStride,
                    *operand.dims);
        to.markHostModified();
    }

    int64_t workspaceSize = 0;
    result = built.graph->get_workspace_size(workspaceSize);
    if(result.code != hipdnn_frontend::ErrorCode::OK || workspaceSize < 0)
    {
        return "the forward graph has no workspace size: " + result.err_msg;
    }
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));
    auto variantPack = forwardBundle.toDeviceVariantPack();
    result = built.graph->execute(handle, variantPack, workspace.get());
    if(result.code != hipdnn_frontend::ErrorCode::OK)
    {
        return "the forward launch that gives O and LSE failed: " + result.err_msg;
    }
    if(hipStreamSynchronize(stream) != hipSuccess)
    {
        return "the forward launch that gives O and LSE did not finish";
    }

    auto& forwardO = forwardBundle.getTensor(O_UID);
    forwardO.markDeviceModified();
    auto& backwardO = bundle.getTensor(bwd_uid::O);
    copyLogical(forwardO.elementSize(),
                forwardO.rawHostData(),
                forwardStrides(spec.queryHeads, spec.seqQ, spec.headSize),
                backwardO.rawHostData(),
                layoutStrides(spec.queryHeads, spec.seqQ, spec.headSize),
                qDims);
    backwardO.markHostModified();

    auto& forwardStats = forwardBundle.getTensor(STATS_UID);
    forwardStats.markDeviceModified();
    auto& backwardStats = bundle.getTensor(bwd_uid::STATS);
    if(forwardStats.elementSpace() != backwardStats.elementSpace())
    {
        return "the forward LSE and the backward LSE differ in size";
    }
    backwardStats.fillWithData(forwardStats.rawHostData(),
                               forwardStats.elementSpace() * forwardStats.elementSize());
    backwardStats.markHostModified();
    return {};
}

} // namespace detail

/// The harness's own backward graph (Q, K, V, O, dO and the LSE in, dQ, dK, dV out) for
/// @p spec. A row that needs attributes it lacks (bias with dBias, sink with dSink, padding)
/// builds its own graph in its own file and passes it through backwardRowWith.
inline PreparedGraph prepareBackward(const DeterminismBwdSpec& spec)
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;
    using kernel_ingestor_engine::testing::sdpaStrides;

    const auto graph = std::make_shared<Graph>();
    graph->set_name(spec.name)
        .set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const auto makeTensor = [](int64_t uid,
                               const std::string& name,
                               DataType dataType,
                               const std::vector<int64_t>& dims,
                               const std::vector<int64_t>& strides) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid).set_name(name).set_dim(dims).set_stride(strides).set_data_type(
            dataType);
        return tensor;
    };
    const std::vector<int64_t> qDims = {spec.batch, spec.queryHeads, spec.seqQ, spec.headSize};
    const std::vector<int64_t> kDims = {spec.batch, spec.kvHeads, spec.seqKv, spec.headSize};
    const auto qStrides = sdpaStrides(spec.layout, spec.queryHeads, spec.seqQ, spec.headSize);
    const auto kStrides = sdpaStrides(spec.layout, spec.kvHeads, spec.seqKv, spec.headSize);
    const std::vector<int64_t> statsDims = {spec.batch, spec.queryHeads, spec.seqQ, 1};
    const std::vector<int64_t> statsStrides = {spec.queryHeads * spec.seqQ, spec.seqQ, 1, 1};

    const auto q = makeTensor(bwd_uid::Q, "Q", spec.dataType, qDims, qStrides);
    const auto k = makeTensor(bwd_uid::K, "K", spec.dataType, kDims, kStrides);
    const auto v = makeTensor(bwd_uid::V, "V", spec.dataType, kDims, kStrides);
    const auto o = makeTensor(bwd_uid::O, "O", spec.dataType, qDims, qStrides);
    const auto dO = makeTensor(bwd_uid::DO, "dO", spec.dataType, qDims, qStrides);
    const auto stats = makeTensor(bwd_uid::STATS, "LSE", DataType::FLOAT, statsDims, statsStrides);

    SdpaBackwardAttributes attributes;
    attributes.set_name(spec.name);
    attributes.set_attn_scale(
        spec.scale.value_or(1.0F / std::sqrt(static_cast<float>(spec.headSize))));
    if(spec.leftBound.has_value())
    {
        attributes.set_diagonal_band_left_bound(*spec.leftBound);
    }
    if(spec.rightBound.has_value())
    {
        attributes.set_diagonal_band_right_bound(*spec.rightBound);
    }
    if(spec.leftBound.has_value() || spec.rightBound.has_value())
    {
        attributes.set_diagonal_alignment(spec.alignment);
    }

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(attributes));
    dq->set_uid(bwd_uid::DQ)
        .set_name("dQ")
        .set_output(true)
        .set_data_type(spec.dataType)
        .set_dim(qDims)
        .set_stride(qStrides);
    dk->set_uid(bwd_uid::DK)
        .set_name("dK")
        .set_output(true)
        .set_data_type(spec.dataType)
        .set_dim(kDims)
        .set_stride(kStrides);
    dv->set_uid(bwd_uid::DV)
        .set_name("dV")
        .set_output(true)
        .set_data_type(spec.dataType)
        .set_dim(kDims)
        .set_stride(kStrides);

    PreparedGraph prepared;
    prepared.graph = graph;
    const auto type = elementTypeOf(spec.dataType);
    prepared.outputs[bwd_uid::DQ] = {"dQ", type, logicalRuns(qDims, qStrides), false};
    prepared.outputs[bwd_uid::DK] = {"dK", type, logicalRuns(kDims, kStrides), false};
    prepared.outputs[bwd_uid::DV] = {"dV", type, logicalRuns(kDims, kStrides), false};
    // The last query row sees keys; the first row of a causal graph sees one.
    prepared.sensitivityUid = bwd_uid::Q;
    prepared.sensitivityElement = static_cast<size_t>((spec.seqQ - 1) * qStrides[2]);
    prepared.deriveInputs = [spec](hipdnnHandle_t handle,
                                   hipStream_t stream,
                                   hipdnn_test_sdk::utilities::GraphTensorBundle& bundle) {
        return detail::deriveBackwardOutputs(spec, handle, stream, bundle);
    };
    return prepared;
}

/// A backward row on the harness's own graph.
inline DeterminismRow backwardRow(RowMeta meta, DeterminismBwdSpec spec)
{
    DeterminismRow row;
    row.name = detail::rowName(meta);
    row.meta = std::move(meta);
    row.direction = Direction::BACKWARD;
    row.build = [spec = std::move(spec)](const DeterminismRow& /*self*/) {
        return prepareBackward(spec);
    };
    return row;
}

/// A backward row on a graph the row's own file builds.
inline DeterminismRow backwardRowWith(RowMeta meta, GraphBuilder build)
{
    DeterminismRow row;
    row.name = detail::rowName(meta);
    row.meta = std::move(meta);
    row.direction = Direction::BACKWARD;
    row.build = std::move(build);
    return row;
}

/// Runs one row on the current device: the per-case contract of the suite.
class DeterminismCase : public ::testing::Test
{
public:
    explicit DeterminismCase(DeterminismRow row)
        : _row(std::move(row))
    {
    }

    void TestBody() override
    {
        SKIP_IF_NO_DEVICES();
        const bool measure = _row.meta.disposition == Disposition::MEASURE_ONLY;
        RecordProperty("sdpa_row", _row.name);
        RecordProperty("sdpa_engine", _row.meta.engine);
        RecordProperty("sdpa_disposition", measure ? "measure_only" : "proven");
        RecordProperty("sdpa_outcome", "not_run");

        ASSERT_NO_FATAL_FAILURE(openDevice());
        RecordProperty("sdpa_arch", _arch);
        const auto skip = skipReason();
        if(skip.has_value())
        {
            RecordProperty("sdpa_outcome", "skipped");
            GTEST_SKIP() << *skip;
        }

        PreparedGraph prepared = _row.build(_row);
        ASSERT_NO_FATAL_FAILURE(pinEngine(prepared));
        const size_t launches = measure ? MEASURE_LAUNCHES : REPEAT_LAUNCHES;
        RecordProperty("sdpa_launches", std::to_string(launches));
        ASSERT_NO_FATAL_FAILURE(repeat(prepared, launches, measure));
    }

    ~DeterminismCase() override
    {
        if(_handle != nullptr)
        {
            static_cast<void>(hipdnnDestroy(_handle));
        }
        if(_stream != nullptr)
        {
            static_cast<void>(hipStreamDestroy(_stream));
        }
    }

    DeterminismCase(const DeterminismCase&) = delete;
    DeterminismCase& operator=(const DeterminismCase&) = delete;

private:
    using Tensors = hipdnn_test_sdk::utilities::GraphTensorBundle;

    void openDevice()
    {
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);
        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathText = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathText.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
        _arch = hip_kernel_provider_common::getDeviceString(_stream);
    }

    /// The reason this device does not run the row, if it does not: the only skips of the suite.
    std::optional<std::string> skipReason() const
    {
        const auto& archs = _row.meta.archs;
        if(std::find(archs.begin(), archs.end(), _arch) == archs.end())
        {
            return _row.name + " is listed for " + detail::joinWith(archs, ", ")
                   + "; this device is " + _arch;
        }
        const bool mi308 = detail::isMi308Device(_deviceId);
        if(_row.meta.sku == Sku::MI300 && mi308)
        {
            return _row.name + " is listed for MI300 devices; this device is an MI308X";
        }
        if(_row.meta.sku == Sku::MI308 && !mi308)
        {
            return _row.name + " is listed for MI308X devices; this device is " + _arch
                   + " and not an MI308X";
        }
        return std::nullopt;
    }

    /// Steps 2 to 4 of the contract: validate, build, require the pinned engine in the ranked
    /// list, pin it with the row's knobs and require that it is the one served.
    void pinEngine(PreparedGraph& prepared)
    {
        using hipdnn_frontend::ErrorCode;
        auto& graph = *prepared.graph;
        auto result = graph.validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const int64_t pinned = hipdnn_data_sdk::utilities::engineNameToId(_row.meta.engine);
        std::vector<int64_t> ranked;
        result = graph.get_ranked_engine_ids(ranked);
        std::string rankedText;
        for(const auto id : ranked)
        {
            rankedText += " " + std::to_string(id);
        }
        if(result.code != ErrorCode::OK
           || std::find(ranked.begin(), ranked.end(), pinned) == ranked.end())
        {
            RecordProperty("sdpa_outcome", "engine_not_ranked");
            FAIL() << _row.meta.engine << " (" << pinned << ") is not in the ranked engine ids of "
                   << _row.name << " on " << _arch << " (" << result.err_msg
                   << "; ranked:" << rankedText
                   << "). The row says this engine serves it on this device.";
        }

        std::vector<hipdnn_frontend::KnobSetting> settings;
        for(const auto& [knob, value] : _row.meta.knobs)
        {
            settings.emplace_back(knob, value);
        }
        result = graph.create_execution_plan_ext(pinned, settings);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t served = 0;
        result = graph.get_execution_plan_engine_id(served);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(served, pinned) << "the plan is served by engine " << served << ", not "
                                  << _row.meta.engine;
    }

    struct PlannedTensor
    {
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr;
        bool output = false;
    };

    /// The tensors of the graph, by uid: its operands and outputs, and the extra inputs.
    static std::map<int64_t, PlannedTensor> plannedTensors(const PreparedGraph& prepared)
    {
        std::map<int64_t, PlannedTensor> planned;
        prepared.graph->visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& attr : node.getNodeOutputTensorAttributes())
            {
                if(!attr->get_is_virtual())
                {
                    planned.emplace(attr->get_uid(), PlannedTensor{attr, true});
                }
            }
            for(const auto& attr : node.getNodeInputTensorAttributes())
            {
                if(!attr->get_is_virtual())
                {
                    planned.emplace(attr->get_uid(), PlannedTensor{attr, false});
                }
            }
        });
        for(const auto& attr : prepared.extraInputs)
        {
            planned.emplace(attr->get_uid(), PlannedTensor{attr, false});
        }
        return planned;
    }

    /// Creates the tensors and fills every input: the row's fixed bytes where it has them,
    /// bounded finite values elsewhere, then what the row derives from them.
    void fillInputs(const PreparedGraph& prepared,
                    const std::map<int64_t, PlannedTensor>& planned,
                    Tensors& tensors)
    {
        for(const auto& [uid, plan] : planned)
        {
            tensors.addTensor(*plan.attr,
                              hipdnn_test_sdk::utilities::createTensorFromAttribute(*plan.attr));
            if(plan.output)
            {
                tensors.outputTensorIds.insert(uid);
                ASSERT_EQ(prepared.outputs.count(uid), 1U)
                    << "output " << plan.attr->get_name() << " (uid " << uid
                    << ") has no live-element description in " << _row.name;
                continue;
            }
            auto& tensor = tensors.getTensor(uid);
            const auto fixed = prepared.fixedInputs.find(uid);
            if(fixed != prepared.fixedInputs.end())
            {
                ASSERT_EQ(fixed->second.size(), tensor.elementSpace() * tensor.elementSize())
                    << "fixed input " << plan.attr->get_name() << " has the wrong size";
                tensor.fillWithData(fixed->second.data(), fixed->second.size());
                continue;
            }
            const auto type = elementTypeOf(plan.attr->get_data_type());
            ASSERT_EQ(tensor.elementSize(), elementBytes(type));
            fillLogical(type,
                        tensor.rawHostData(),
                        plan.attr->get_dim(),
                        plan.attr->get_stride(),
                        static_cast<uint64_t>(uid),
                        prepared.inputBound);
            tensor.markHostModified();
        }
        for(const auto& [uid, spec] : prepared.outputs)
        {
            ASSERT_EQ(tensors.outputTensorIds.count(uid), 1U)
                << "the live-element description of " << spec.name << " names no graph output";
        }
        if(prepared.deriveInputs)
        {
            const auto error = prepared.deriveInputs(_handle, _stream, tensors);
            ASSERT_TRUE(error.empty()) << error;
        }
    }

    /// One launch: poison the outputs and the workspace, execute, and bring the outputs to
    /// the host.
    void launch(PreparedGraph& prepared,
                Tensors& tensors,
                hipdnn_data_sdk::utilities::Workspace<>& workspace,
                size_t workspaceBytes)
    {
        for(const auto uid : tensors.outputTensorIds)
        {
            auto& tensor = tensors.getTensor(uid);
            poison(tensor.rawHostData(), tensor.elementSpace() * tensor.elementSize());
            tensor.markHostModified();
        }
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMemsetAsync(workspace.get(), POISON_BYTE, workspaceBytes, _stream),
                      hipSuccess);
        }
        auto variantPack = tensors.toDeviceVariantPack();
        const auto result = prepared.graph->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        for(const auto uid : tensors.outputTensorIds)
        {
            tensors.getTensor(uid).markDeviceModified();
        }
    }

    /// What measuring one output over all launches found.
    struct OutputReport
    {
        size_t differingLaunches = 0;
        size_t invalidLaunches = 0;
        size_t firstDifferingLaunch = 0;
        size_t firstDifferingElement = 0;
        size_t differingElementsMax = 0;
        size_t differingElementsTotal = 0;
        bool anyDifference = false;
    };

    void repeat(PreparedGraph& prepared, size_t launches, bool measure)
    {
        const auto planned = plannedTensors(prepared);
        Tensors tensors;
        ASSERT_NO_FATAL_FAILURE(fillInputs(prepared, planned, tensors));

        int64_t workspaceSize = 0;
        const auto result = prepared.graph->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0);
        const auto workspaceBytes = static_cast<size_t>(workspaceSize);
        hipdnn_data_sdk::utilities::Workspace workspace(workspaceBytes);

        std::map<int64_t, Bytes> reference;
        std::map<int64_t, OutputReport> reports;
        std::vector<std::string> failures;
        for(size_t launchIndex = 0; launchIndex < launches && (measure || failures.empty());
            ++launchIndex)
        {
            ASSERT_NO_FATAL_FAILURE(launch(prepared, tensors, workspace, workspaceBytes));
            for(const auto& [uid, output] : prepared.outputs)
            {
                auto& tensor = tensors.getTensor(uid);
                const auto* data = static_cast<const uint8_t*>(tensor.rawHostData());
                const size_t storageBytes = tensor.elementSpace() * tensor.elementSize();
                const size_t size = tensor.elementSize();
                auto invalid = findInvalidElement(
                    output.name, output.type, data, output.live, output.allowInfinity);
                if(invalid.has_value())
                {
                    invalid->launch = launchIndex;
                    ++reports[uid].invalidLaunches;
                    failures.push_back(invalid->describe());
                }
                if(launchIndex == 0)
                {
                    reference[uid] = Bytes(data, data + storageBytes);
                    continue;
                }
                auto difference = findFirstDifference(
                    output.name, size, reference[uid].data(), data, output.live);
                if(difference.has_value())
                {
                    difference->launch = launchIndex;
                    auto& report = reports[uid];
                    const size_t differing
                        = countDifferences(size, reference[uid].data(), data, output.live);
                    if(!report.anyDifference)
                    {
                        report.anyDifference = true;
                        report.firstDifferingLaunch = launchIndex;
                        report.firstDifferingElement = difference->element;
                    }
                    ++report.differingLaunches;
                    report.differingElementsTotal += differing;
                    report.differingElementsMax = std::max(report.differingElementsMax, differing);
                    failures.push_back(difference->describe());
                }
            }
        }

        if(measure)
        {
            reportMeasurement(prepared, reports, launches);
        }
        else if(!failures.empty())
        {
            RecordProperty("sdpa_outcome",
                           failures.front().find("unwritten") != std::string::npos ? "unwritten"
                                                                                   : "differs");
            FAIL() << _row.name << " on " << _arch << ": " << failures.front() << " ("
                   << failures.size() << " finding(s) before the run stopped)";
        }

        ASSERT_NO_FATAL_FAILURE(
            checkSensitivity(prepared, tensors, workspace, workspaceBytes, reference));
        if(!measure)
        {
            RecordProperty("sdpa_outcome", "identical");
        }
    }

    /// Step 7: one Q element changed must change an output.
    void checkSensitivity(PreparedGraph& prepared,
                          Tensors& tensors,
                          hipdnn_data_sdk::utilities::Workspace<>& workspace,
                          size_t workspaceBytes,
                          const std::map<int64_t, Bytes>& reference)
    {
        auto& input = tensors.getTensor(prepared.sensitivityUid);
        const auto planned = plannedTensors(prepared);
        const auto type = elementTypeOf(planned.at(prepared.sensitivityUid).attr->get_data_type());
        const size_t size = elementBytes(type);
        ASSERT_LT(prepared.sensitivityElement, input.elementSpace());
        auto* element
            = static_cast<uint8_t*>(input.rawHostData()) + prepared.sensitivityElement * size;
        Bytes changed(size);
        encodeElement(type, prepared.inputBound, changed.data());
        if(std::memcmp(changed.data(), element, size) == 0)
        {
            encodeElement(type, -prepared.inputBound, changed.data());
        }
        std::memcpy(element, changed.data(), size);
        input.markHostModified();

        ASSERT_NO_FATAL_FAILURE(launch(prepared, tensors, workspace, workspaceBytes));
        bool changedOutput = false;
        for(const auto& [uid, output] : prepared.outputs)
        {
            auto& tensor = tensors.getTensor(uid);
            changedOutput = changedOutput
                            || findFirstDifference(output.name,
                                                   tensor.elementSize(),
                                                   reference.at(uid).data(),
                                                   tensor.rawHostData(),
                                                   output.live)
                                   .has_value();
        }
        if(!changedOutput)
        {
            RecordProperty("sdpa_outcome", "insensitive");
        }
        ASSERT_TRUE(changedOutput)
            << _row.name << ": changing element " << prepared.sensitivityElement
            << " of the input with uid " << prepared.sensitivityUid
            << " changed no output, so the row would pass whatever the "
               "engine did with its inputs";
    }

    void reportMeasurement(const PreparedGraph& prepared,
                           const std::map<int64_t, OutputReport>& reports,
                           size_t launches) const
    {
        bool anyDifference = false;
        for(const auto& [uid, output] : prepared.outputs)
        {
            OutputReport report;
            const auto it = reports.find(uid);
            if(it != reports.end())
            {
                report = it->second;
            }
            anyDifference = anyDifference || report.anyDifference;
            const std::string prefix = "sdpa_" + output.name + "_";
            RecordProperty(prefix + "launches_differing", std::to_string(report.differingLaunches));
            RecordProperty(prefix + "launches_invalid", std::to_string(report.invalidLaunches));
            RecordProperty(prefix + "first_differing_element",
                           report.anyDifference ? std::to_string(report.firstDifferingElement)
                                                : "none");
            RecordProperty(prefix + "differing_elements_max",
                           std::to_string(report.differingElementsMax));
            std::cout << "[ MEASURE  ] " << _row.name << " " << output.name << ": "
                      << report.differingLaunches << " of " << launches - 1
                      << " launches differ from launch 0, " << report.invalidLaunches
                      << " launches invalid, first differing element "
                      << (report.anyDifference ? std::to_string(report.firstDifferingElement)
                                               : std::string("none"))
                      << ", most differing elements in one launch " << report.differingElementsMax
                      << "\n";
        }
        RecordProperty("sdpa_outcome", anyDifference ? "differs" : "identical");
    }

    DeterminismRow _row;
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
    std::string _arch;
    /// No winner record another case measured.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "sdpa-determinism-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

/// One registered row's identity, kept for a host-only check across the whole binary: which
/// engine it pins, which architectures it runs on (skipped elsewhere) and whether a passing
/// run is expected (PROVEN) or the row is a known hazard, measured and reported only
/// (MEASURE_ONLY). registerRows appends to this as every determinism translation unit's
/// static initializer runs, before main() applies any --gtest_filter.
struct RegisteredRow
{
    std::string engine;
    std::vector<std::string> archs;
    Disposition disposition;
};

inline std::vector<RegisteredRow>& registeredDeterminismRows()
{
    static std::vector<RegisteredRow> s_rows;
    return s_rows;
}

/// Registers every row of @p rows as a gtest case under the names in the file comment. Called
/// from a static initializer of the row's own file, before main() runs the tests. A row name
/// registered twice is a defect of the tables and terminates the process.
inline bool registerRows(const std::vector<DeterminismRow>& rows)
{
    static std::set<std::string> s_registered;
    for(const auto& row : rows)
    {
        std::string suite;
        std::string test;
        if(row.meta.disposition == Disposition::PROVEN)
        {
            suite = std::string("Comprehensive/IntegrationGpuSdpaDeterminism")
                    + (row.direction == Direction::FORWARD ? "Fwd" : "Bwd");
            test = "RepeatsBitwise/" + row.name;
        }
        else
        {
            suite = "Measure/IntegrationGpuSdpaDeterminismMeasure";
            test = "DISABLED_RepeatsBitwise/" + row.name;
        }
        registeredDeterminismRows().push_back(
            {row.meta.engine, row.meta.archs, row.meta.disposition});
        std::string key = suite;
        key += '.';
        key += test;
        if(!s_registered.insert(key).second)
        {
            std::cerr << "SDPA determinism row registered twice: " << suite << "." << test << "\n";
            std::terminate();
        }
        static_cast<void>(::testing::RegisterTest(
            suite.c_str(),
            test.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [row]() -> ::testing::Test* { return new DeterminismCase(row); }));
    }
    return true;
}

} // namespace hip_kernel_provider::sdpa_determinism

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
