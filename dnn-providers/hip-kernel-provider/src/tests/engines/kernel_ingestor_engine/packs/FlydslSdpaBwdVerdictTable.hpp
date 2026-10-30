// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

#ifndef HKP_FLYDSL_SDPA_BWD_DATA_RELDIR
#error "HKP_FLYDSL_SDPA_BWD_DATA_RELDIR is defined by the test CMakeLists"
#endif

/**
 * @file FlydslSdpaBwdVerdictTable.hpp
 * @brief The gfx11 verdict table (flydsl/tests/data/sdpa_bwd_gfx11_verdicts.csv) as graphs the
 *        split SDPA backward engine can be asked about, and the slot-product uniqueness
 *        predicate both of its consumers apply.
 *
 * The table's rows are written by flydsl/tools/sdpa_bwd_verdicts.py from the bundle's routing
 * and the matcher's documented rules, never from this engine's output, so a disagreement
 * between a row and the engine is a finding. specOfRow() turns one row into the graph it
 * describes: Q/K at `stride_qk`, V/O/dO/dV at `stride_v` (head stride `seq * stride`, batch
 * stride `heads * head stride`), a bias of the broadcast form the `bias` column names with its
 * rows padded to a whole vector, and the mask, windows, dropout seed form and varlen form
 * the other columns name.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// One row of the verdict table, as written; `hash` columns are the 12-hex spec hashes of the
/// gfx11-generic manifests.
struct VerdictRow
{
    std::string id;
    std::string source;
    std::string dtype;
    int64_t dqk = 0;
    int64_t dv = 0;
    std::string mask;
    int64_t windowLeft = -1;
    int64_t windowRight = -1;
    int64_t dropout = 0;
    std::string bias;
    int64_t dbias = 0;
    std::string varlen;
    int64_t hq = 0;
    int64_t hk = 0;
    int64_t batch = 0;
    int64_t sq = 0;
    int64_t sk = 0;
    int64_t strideQk = 0;
    int64_t strideV = 0;
    std::string seedForm;
    bool served = false;
    std::string prep;
    std::string dkdv;
    std::string dq;
    /// The dBias reduce companion's hash; set only for a broadcast dBias.
    std::string reduce;
    /// The matcher rule that declines the row, or `no_row`; empty for a served row.
    std::string rule;
};

/// The directory staged beside the binary's descriptor trees that holds the verdict table
/// (`sdpa_bwd_gfx11_verdicts.csv`) and the manifests (`<op>/gfx11-generic/manifest.json`);
/// resolved from the executable like the descriptor roots, so a relocated tree carries it.
inline std::filesystem::path verdictDataRoot()
{
    return hip_kernel_provider::testing::descriptorSetRoot(HKP_FLYDSL_SDPA_BWD_DATA_RELDIR);
}

inline std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);
    while(std::getline(stream, cell, ','))
    {
        cells.push_back(cell);
    }
    if(!line.empty() && line.back() == ',')
    {
        cells.emplace_back();
    }
    return cells;
}

/// The rows of the table at @p path; throws when the file is missing or malformed, so a
/// consumer fails rather than checking nothing.
inline std::vector<VerdictRow>
    loadVerdictRows(const std::string& path
                    = (verdictDataRoot() / "sdpa_bwd_gfx11_verdicts.csv").string())
{
    std::ifstream in(path);
    if(!in.good())
    {
        throw std::runtime_error("cannot open the verdict table " + path);
    }
    std::string line;
    std::getline(in, line);
    const auto header = splitCsvLine(line);
    std::map<std::string, size_t> column;
    for(size_t index = 0; index < header.size(); ++index)
    {
        column[header[index]] = index;
    }

    std::vector<VerdictRow> rows;
    while(std::getline(in, line))
    {
        if(line.empty())
        {
            continue;
        }
        const auto cells = splitCsvLine(line);
        if(cells.size() != header.size())
        {
            throw std::runtime_error("verdict table row with " + std::to_string(cells.size())
                                     + " cells, expected " + std::to_string(header.size()) + ": "
                                     + line);
        }
        const auto text = [&](const char* name) -> const std::string& {
            const auto found = column.find(name);
            if(found == column.end())
            {
                throw std::runtime_error(std::string("verdict table has no column ") + name);
            }
            return cells[found->second];
        };
        const auto integer = [&](const char* name) { return std::stoll(text(name)); };

        VerdictRow row;
        row.id = text("id");
        row.source = text("source");
        row.dtype = text("dtype");
        row.dqk = integer("dqk");
        row.dv = integer("dv");
        row.mask = text("mask");
        row.windowLeft = integer("window_left");
        row.windowRight = integer("window_right");
        row.dropout = integer("dropout");
        row.bias = text("bias");
        row.dbias = integer("dbias");
        row.varlen = text("varlen");
        row.hq = integer("hq");
        row.hk = integer("hk");
        row.batch = integer("b");
        row.sq = integer("sq");
        row.sk = integer("sk");
        row.strideQk = integer("stride_qk");
        row.strideV = integer("stride_v");
        row.seedForm = text("seed_form");
        row.served = text("verdict") == "served";
        row.prep = text("prep");
        row.dkdv = text("dkdv");
        row.dq = text("dq");
        row.reduce = text("reduce");
        row.rule = text("rule");
        rows.push_back(std::move(row));
    }
    return rows;
}

/// The cause word the engine's `Decline::cause` carries for each matcher rule a declined row
/// names. Mirrors `RULE_CAUSE` in flydsl/tools/sdpa_bwd_verdicts.py, which derives each entry
/// from the matcher's documentation in FlydslSdpaBwdCommon.hpp; a rule the table uses that is
/// missing here is itself a finding.
inline const std::map<std::string, std::string>& ruleCauses()
{
    static const std::map<std::string, std::string> s_causes{{"SH-1", "shape"},
                                                             {"DT-4", "head_size"},
                                                             {"DT-6", "gqa"},
                                                             {"DT-8", "layout"},
                                                             {"DT-9", "layout"},
                                                             {"LY-3", "layout"},
                                                             {"LY-6", "layout"},
                                                             {"MW-3", "mask"},
                                                             {"MW-9b", "mask"},
                                                             {"MW-9c", "mask"},
                                                             {"MW-10", "mask"},
                                                             {"MW-13", "mask"},
                                                             {"PV-3", "varlen"},
                                                             {"PV-10", "varlen"},
                                                             {"BI-4", "dbias"},
                                                             {"BI-6", "dbias"}};
    return s_causes;
}

/// The `rule` token of a row the matcher accepts and no gfx11 object serves.
inline constexpr const char* NO_ROW_RULE = "no_row";

/// The graph @p row describes.
inline SdpaBwdSpec specOfRow(const VerdictRow& row)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    SdpaBwdSpec spec;
    if(row.dtype == "f16")
    {
        spec.dtype = data_objects::DataType::HALF;
    }
    else if(row.dtype == "bf16")
    {
        spec.dtype = data_objects::DataType::BFLOAT16;
    }
    else if(row.dtype == "fp32")
    {
        spec.dtype = data_objects::DataType::FLOAT;
    }
    else
    {
        throw std::invalid_argument("verdict row " + row.id + ": unknown dtype " + row.dtype);
    }
    spec.batch = row.batch;
    spec.heads = row.hq;
    spec.kvHeads = row.hk;
    spec.seqQ = row.sq;
    spec.seqKv = row.sk;
    spec.headDimQk = row.dqk;
    spec.headDimV = row.dv;

    return spec.withEdit([row](SdpaBwdGraph& graph) {
        namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

        const auto setRows = [&](std::initializer_list<const char*> names,
                                 int64_t heads,
                                 int64_t seq,
                                 int64_t rowStride) {
            for(const char* name : names)
            {
                graph.tensor(name).strides = headMajorStrides(heads, seq, rowStride);
            }
        };
        setRows({"q", "dq"}, row.hq, row.sq, row.strideQk);
        setRows({"k", "dk"}, row.hk, row.sk, row.strideQk);
        setRows({"v", "dv"}, row.hk, row.sk, row.strideV);
        setRows({"o", "do"}, row.hq, row.sq, row.strideV);

        auto& a = graph.attributes;
        if(row.mask == "causal_tl")
        {
            a.causalMask = true;
        }
        else if(row.mask == "causal_br")
        {
            a.causalMaskBottomRight = true;
        }
        if(row.windowLeft != -1)
        {
            a.leftBound = row.windowLeft;
        }
        if(row.windowRight != -1)
        {
            a.rightBound = row.windowRight;
        }

        if(row.bias != "none")
        {
            const bool overBatch = row.bias == "b" || row.bias == "bh";
            const bool overHead = row.bias == "h" || row.bias == "bh";
            const bool overRow = row.bias == "q";
            const std::vector<int64_t> dims{overBatch ? int64_t{1} : row.batch,
                                            overHead ? int64_t{1} : row.hq,
                                            overRow ? int64_t{1} : row.sq,
                                            row.sk};
            auto& bias = addBias(graph, dims);
            const auto pitch = ceilToVectorOf(bias.dtype, row.sk);
            bias.strides = {dims[1] * dims[2] * pitch, dims[2] * pitch, pitch, 1};
            if(row.dbias != 0)
            {
                addDbias(graph);
            }
        }
        else if(row.dbias != 0)
        {
            SdpaBwdTensor dbias;
            dbias.uid = sdpa_bwd_uid::DBIAS;
            dbias.name = "dbias";
            dbias.dtype = graph.tensor("q").dtype;
            dbias.dims = {row.batch, row.hq, row.sq, row.sk};
            const auto pitch = ceilToVectorOf(dbias.dtype, row.sk);
            dbias.strides = {row.hq * row.sq * pitch, row.sq * pitch, pitch, 1};
            a.dbiasUid = dbias.uid;
            graph.tensors.push_back(std::move(dbias));
        }

        if(row.varlen == "padded")
        {
            addSeqLens(graph);
        }
        else if(row.varlen == "packed")
        {
            graph.tensor("q").raggedOffsetUid = sdpa_bwd_uid::SEQ_LEN_Q;
        }

        if(row.dropout != 0)
        {
            addDropout(graph, 0.2F, data_objects::DataType::INT64, row.seedForm == "host");
        }
    });
}

/// The split engine's slots, in the order its descriptors declare them.
inline const char* slotName(FlydslSdpaBwdSplitSlot slot)
{
    switch(slot)
    {
    case FlydslSdpaBwdSplitSlot::PREP:
        return "prep";
    case FlydslSdpaBwdSplitSlot::DKDV:
        return "dkdv";
    default:
        return "dq";
    }
}

/// The native symbols of @p slot of the split engine.
inline PackSymbols splitSymbols(FlydslSdpaBwdSplitSlot slot)
{
    static const std::string s_graphMatch = "hipkernel.flydsl_sdpa_bwd_split.graph_match";
    static const std::string s_engine = "hipkernel:flydsl_sdpa_bwd_split";
    static const std::map<std::string, std::pair<std::string, std::string>> s_symbols{
        {"prep",
         {"hipkernel.flydsl_sdpa_bwd_split.prep.kernel_match",
          "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch"}},
        {"dkdv",
         {"hipkernel.flydsl_sdpa_bwd_split.dkdv.kernel_match",
          "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch"}},
        {"dq",
         {"hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match",
          "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch"}}};
    const auto& [kernelMatch, dispatch] = s_symbols.at(slotName(slot));
    return {s_engine, s_graphMatch, "", kernelMatch, "", dispatch, "", "", ""};
}

/// The knobs each slot's UED enrolls (descriptors/FlyDSL/sdpa_bwd_split/*.ued.json). The
/// census compares this list to the engine descriptor it loads.
inline std::vector<std::string> enrolledKnobs(const std::string& slot)
{
    if(slot == "prep")
    {
        return {"padded", "hints"};
    }
    if(slot == "dkdv")
    {
        return {"head_dim",
                "padded",
                "hints",
                "floor",
                "dform",
                "num_waves",
                "mfma_rows",
                "dkv_shards",
                "block_q",
                "waves_per_eu",
                "tight_registers",
                "num_stream_buffers",
                "gqa_pack",
                "block_threads",
                "block_kv"};
    }
    return {"head_dim",
            "padded",
            "hints",
            "floor",
            "dform",
            "num_waves",
            "block_m",
            "block_n",
            "mfma_rows",
            "waves_per_eu",
            "store_db",
            "bias_stage",
            "block_threads"};
}

/// The first pair of @p kernels (a slot's catalog for one graph) whose enrolled knob tuples
/// are equal, as "name and name"; nullopt when every tuple is unique.
///
/// The predicate of GenericPlanBuilder::enumerateSlotted (GenericPlanBuilder.hpp, "Ambiguous
/// enrolled knob tuple in slot"): per slot, two catalog entries with the same value of every
/// enrolled knob cannot be told apart by an EngineVariant's knob settings. candidateKnobs()
/// keys a kernel by the knobs it carries; a non-integer value enters as its ordinal in the
/// slot's domain, an injective map, so comparing the values themselves is the same test.
inline std::optional<std::string> ambiguousEnrolledTuple(
    const std::vector<const hipdnn_plugin_sdk::ingestor::KernelDefinition*>& kernels,
    const std::vector<std::string>& knobs)
{
    using hipdnn_plugin_sdk::ingestor::MetadataValue;
    std::map<std::vector<std::pair<std::string, MetadataValue>>, std::string> identities;
    for(const auto* kernel : kernels)
    {
        std::vector<std::pair<std::string, MetadataValue>> tuple;
        for(const auto& knob : knobs)
        {
            const auto found = kernel->metadata.find(knob);
            if(found != kernel->metadata.end())
            {
                tuple.emplace_back(knob, found->second);
            }
        }
        const auto inserted = identities.emplace(std::move(tuple), kernel->name);
        if(!inserted.second)
        {
            return inserted.first->second + " and " + kernel->name;
        }
    }
    return std::nullopt;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
