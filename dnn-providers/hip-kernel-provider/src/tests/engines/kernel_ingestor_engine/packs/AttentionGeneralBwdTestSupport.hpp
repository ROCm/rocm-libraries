// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdGeometry.hpp"

/**
 * @file AttentionGeneralBwdTestSupport.hpp
 * @brief Graph, device and descriptor fixtures shared by the hipkernel:AttentionGeneralBwd
 *        host suites.
 *
 * Graphs are built as hipDNN graph JSON and read through the SDK's own JSON reader, the
 * form the replay corpus stores them in, so a hand-built case and a corpus entry reach the
 * lowering by the same path.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::general_bwd_test
{

namespace general_bwd = hip_kernel_provider::kernel_ingestor_engine::general_bwd;
namespace generated = general_bwd::generated;
using json = nlohmann::json;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

inline constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.attention_general_bwd.graph_match";
inline constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.attention_general_bwd.kernel_match";
inline constexpr std::string_view PREP_DISPATCH_SYMBOL
    = "hipkernel.attention_general_bwd.prep_dispatch";
inline constexpr std::string_view MAIN_DISPATCH_SYMBOL
    = "hipkernel.attention_general_bwd.main_dispatch";

/// A graph and the device it is matched on. MatchContext refers to both, so they live
/// together and the fixture is neither copied nor moved.
class GraphFixture
{
public:
    GraphFixture(const json& graph, std::string_view arch, int numCus)
    {
        auto root = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _builder, graph);
        _builder.Finish(root);
        _graph = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper>(
            _builder.GetBufferPointer(), _builder.GetSize());
        _properties.gcnArchName = std::string(arch) + ":sramecc+:xnack-";
        _properties.warpSize = 64;
        _properties.multiProcessorCount = numCus;
    }

    GraphFixture(const GraphFixture&) = delete;
    GraphFixture& operator=(const GraphFixture&) = delete;
    GraphFixture(GraphFixture&&) = delete;
    GraphFixture& operator=(GraphFixture&&) = delete;
    ~GraphFixture() = default;

    MatchContext context() const
    {
        return MatchContext{*_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper> _graph;
    DeviceProperties _properties;
};

/// The full CU count of each shipped device, the count the corpus representatives assume.
inline int fullComputeUnits(std::string_view arch)
{
    return arch == "gfx942" ? 304 : 256;
}

// ---------------------------------------------------------------------------
// Hand-built graphs
// ---------------------------------------------------------------------------

/// One SdpaBackward graph, every operand BSHD-contiguous unless overridden.
struct GraphSpec
{
    std::string dtype = "half";
    int64_t b = 2;
    int64_t hQ = 8;
    int64_t hK = 8;
    int64_t hV = 8;
    int64_t sQ = 128;
    int64_t sKv = 128;
    int64_t dQk = 128;
    int64_t dV = 128;
    bool thd = false;
    bool causal = false;
    bool causalBottomRight = false;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    std::string diagonal = "TOP_LEFT";
    std::optional<double> scale;
    std::optional<int64_t> alignment;
};

inline constexpr int64_t Q_UID = 0;
inline constexpr int64_t K_UID = 1;
inline constexpr int64_t V_UID = 2;
inline constexpr int64_t O_UID = 3;
inline constexpr int64_t DO_UID = 4;
inline constexpr int64_t STATS_UID = 5;
inline constexpr int64_t DQ_UID = 6;
inline constexpr int64_t DK_UID = 7;
inline constexpr int64_t DV_UID = 8;
inline constexpr int64_t OFFSETS_Q_UID = 9;
inline constexpr int64_t OFFSETS_KV_UID = 10;

inline json tensorJson(int64_t uid,
                       std::string_view name,
                       std::string_view dtype,
                       std::vector<int64_t> dims,
                       std::vector<int64_t> strides)
{
    return json{{"uid", uid},
                {"name", name},
                {"data_type", dtype},
                {"dims", std::move(dims)},
                {"strides", std::move(strides)},
                {"virtual", false}};
}

/// BSHD strides of (B, H, S, D) logical dims: token-major, head varying fastest.
inline std::vector<int64_t> bshd(int64_t h, int64_t s, int64_t d)
{
    return {s * h * d, d, h * d, 1};
}

/// @p spec as hipDNN graph JSON. Under THD every data operand is ragged by token offsets (the
/// multiplier is its token stride) and the stats are packed `[T, H, 1]` on Q's table.
inline json bwdGraphJson(const GraphSpec& spec)
{
    const auto qDims = std::vector<int64_t>{spec.b, spec.hQ, spec.sQ, spec.dQk};
    const auto oDims = std::vector<int64_t>{spec.b, spec.hQ, spec.sQ, spec.dV};
    const auto kDims = std::vector<int64_t>{spec.b, spec.hK, spec.sKv, spec.dQk};
    const auto vDims = std::vector<int64_t>{spec.b, spec.hV, spec.sKv, spec.dV};

    json tensors = json::array();
    const auto add = [&](int64_t uid,
                         std::string_view name,
                         const std::vector<int64_t>& dims,
                         std::optional<int64_t> offsets) {
        auto tensor = tensorJson(uid, name, spec.dtype, dims, bshd(dims[1], dims[2], dims[3]));
        if(spec.thd && offsets.has_value())
        {
            tensor["ragged_offset_tensor_uid"] = *offsets;
            tensor["ragged_offset_multiplier"] = dims[1] * dims[3];
        }
        if(spec.alignment.has_value())
        {
            tensor["alignment"] = *spec.alignment;
        }
        tensors.push_back(std::move(tensor));
    };
    add(Q_UID, "Q", qDims, OFFSETS_Q_UID);
    add(K_UID, "K", kDims, OFFSETS_KV_UID);
    add(V_UID, "V", vDims, OFFSETS_KV_UID);
    add(O_UID, "O", oDims, OFFSETS_Q_UID);
    add(DO_UID, "dO", oDims, OFFSETS_Q_UID);
    {
        auto stats = tensorJson(STATS_UID,
                                "stats",
                                "float",
                                {spec.b, spec.hQ, spec.sQ, 1},
                                spec.thd ? std::vector<int64_t>{spec.sQ * spec.hQ, 1, spec.hQ, 1}
                                         : std::vector<int64_t>{spec.hQ * spec.sQ, spec.sQ, 1, 1});
        if(spec.thd)
        {
            stats["ragged_offset_tensor_uid"] = OFFSETS_Q_UID;
            stats["ragged_offset_multiplier"] = spec.hQ;
        }
        tensors.push_back(std::move(stats));
    }
    add(DQ_UID, "DQ", qDims, OFFSETS_Q_UID);
    add(DK_UID, "DK", kDims, OFFSETS_KV_UID);
    add(DV_UID, "DV", vDims, OFFSETS_KV_UID);
    if(spec.thd)
    {
        for(const auto& [uid, name] :
            {std::pair<int64_t, const char*>{OFFSETS_Q_UID, "offsets_q"},
             std::pair<int64_t, const char*>{OFFSETS_KV_UID, "offsets_kv"}})
        {
            tensors.push_back(tensorJson(uid, name, "int32", {spec.b + 1, 1, 1, 1}, {1, 1, 1, 1}));
        }
    }

    const json nullUid = nullptr;
    json node{{"name", "SdpaBwd_0"},
              {"type", "SdpaBackwardAttributes"},
              {"compute_data_type", "float"},
              {"inputs",
               {{"q_tensor_uid", Q_UID},
                {"k_tensor_uid", K_UID},
                {"v_tensor_uid", V_UID},
                {"o_tensor_uid", O_UID},
                {"do_tensor_uid", DO_UID},
                {"stats_tensor_uid", STATS_UID},
                {"attn_mask_tensor_uid", nullUid},
                {"dropout_mask_tensor_uid", nullUid},
                {"dropout_scale_inv_tensor_uid", nullUid},
                {"dropout_scale_tensor_uid", nullUid},
                {"offset_tensor_uid", nullUid},
                {"scale_tensor_uid", nullUid},
                {"seed_tensor_uid", nullUid},
                {"seq_len_kv_tensor_uid", nullUid},
                {"seq_len_q_tensor_uid", nullUid}}},
              {"outputs",
               {{"dq_tensor_uid", DQ_UID},
                {"dk_tensor_uid", DK_UID},
                {"dv_tensor_uid", DV_UID},
                {"dbias_tensor_uid", nullUid}}},
              {"parameters",
               {{"alibi_mask", false},
                {"causal_mask", spec.causal},
                {"causal_mask_bottom_right", spec.causalBottomRight},
                {"diagonal_alignment", spec.diagonal},
                {"padding_mask", false},
                {"attn_scale_value", nullUid},
                {"dropout_probability", nullUid},
                {"left_bound", nullUid},
                {"right_bound", nullUid}}}};
    if(spec.scale.has_value())
    {
        node["parameters"]["attn_scale_value"] = *spec.scale;
    }
    if(spec.leftBound.has_value())
    {
        node["parameters"]["left_bound"] = *spec.leftBound;
    }
    if(spec.rightBound.has_value())
    {
        node["parameters"]["right_bound"] = *spec.rightBound;
    }

    return json{{"name", "SdpaBackward"},
                {"compute_data_type", "float"},
                {"intermediate_data_type", "float"},
                {"io_data_type", spec.dtype},
                {"nodes", json::array({std::move(node)})},
                {"tensors", std::move(tensors)}};
}

/// The tensor of @p graph with @p uid, for a test that edits one operand.
inline json& tensorOf(json& graph, int64_t uid)
{
    for(auto& tensor : graph["tensors"])
    {
        if(tensor["uid"].get<int64_t>() == uid)
        {
            return tensor;
        }
    }
    throw std::runtime_error("no tensor with uid " + std::to_string(uid));
}

// ---------------------------------------------------------------------------
// Kernel descriptors as the KMDs record them
// ---------------------------------------------------------------------------

inline std::string_view dtypeSpelling(generated::Dtype dtype)
{
    return dtype == generated::Dtype::FP16 ? "FP16" : "BF16";
}

inline std::string tuningKeyText(const generated::CatalogRow& row)
{
    if(row.tuningKeyId == 0)
    {
        return "";
    }
    for(const auto& entry : generated::TUNED_KEYS)
    {
        if(entry.arch == row.arch && entry.id == row.tuningKeyId)
        {
            return std::string(entry.key);
        }
    }
    throw std::runtime_error("catalog row names an unknown tuning key");
}

/// The main kernel of @p row, carrying the metadata its KMD record carries.
inline KernelDefinition mainKernelOf(const generated::CatalogRow& row)
{
    KernelDefinition kernel;
    kernel.name = std::string(row.mainKernel);
    kernel.metadata = {
        {"dtype", std::string(dtypeSpelling(row.dtype))},
        {"head_size", row.headSize},
        {"seq_mode", std::string(row.seq == generated::SeqMode::THD ? "thd" : "batched")},
        {"stage_vec", row.stageVec},
        {"dkv_mode", std::string(row.dkv == generated::DkvMode::ATOMIC ? "atomic" : "direct")},
        {"mask_class", std::string(row.mask == generated::MaskClass::BAND ? "band" : "none")},
        {"dq_mode", std::string(row.dq == generated::DqMode::SPLIT ? "split" : "atomic")},
        {"head_pack", row.headPack},
        {"lb_mirror", row.lbMirror},
        {"lb_interleave", row.lbInterleave},
        {"tuning_key", tuningKeyText(row)},
        {"main_block_n", row.mainBlockN},
        {"main_worklist", row.mainWorklist},
        {"main_static_scale",
         std::string(generated::STATIC_SCALES.at(static_cast<std::size_t>(row.mainStaticScale)))},
    };
    return kernel;
}

/// The prep kernel @p row launches, carrying the metadata its KMD record carries.
inline KernelDefinition prepKernelOf(const generated::CatalogRow& row)
{
    KernelDefinition kernel;
    kernel.name = std::string(row.prepKernel);
    kernel.metadata = {
        {"dtype", std::string(dtypeSpelling(row.dtype))},
        {"head_size", row.headSize},
        {"seq_mode", std::string(row.seq == generated::SeqMode::THD ? "thd" : "batched")},
        {"stage_vec", row.stageVec},
        {"worklist", row.prepWorklist},
        {"rows_per_cta", row.prepRowsPerCta},
        {"narrow_vec", row.prepNarrowVec},
    };
    return kernel;
}

/// Every distinct main and prep kernel of @p arch's packs, as the KMDs record them.
struct ArchKernels
{
    std::vector<KernelDefinition> main;
    std::vector<KernelDefinition> prep;
};

inline ArchKernels archKernels(std::string_view arch)
{
    ArchKernels kernels;
    for(const auto& row : generated::CATALOG)
    {
        if(row.arch != arch)
        {
            continue;
        }
        kernels.main.push_back(mainKernelOf(row));
        auto prep = prepKernelOf(row);
        bool seen = false;
        for(const auto& known : kernels.prep)
        {
            seen = seen || known.metadata == prep.metadata;
        }
        if(!seen)
        {
            kernels.prep.push_back(std::move(prep));
        }
    }
    return kernels;
}

// ---------------------------------------------------------------------------
// The replay corpus
// ---------------------------------------------------------------------------

/// The corpus of @p arch: `HKP_GENERAL_BWD_CORPUS` (a corpus file, or a directory holding
/// `corpus_<arch>.jsonl`) when set, else the subset staged beside this binary.
inline std::filesystem::path corpusPath(std::string_view arch)
{
    const std::string fileName = "corpus_" + std::string(arch) + ".jsonl";
    const auto overridePath = hipdnn_data_sdk::utilities::getEnv("HKP_GENERAL_BWD_CORPUS");
    if(!overridePath.empty())
    {
        const std::filesystem::path path(overridePath);
        return std::filesystem::is_directory(path) ? path / fileName : path;
    }
#ifdef HIPKERNELPROVIDER_GENERAL_BWD_CORPUS_RELDIR
    return hip_kernel_provider::testing::descriptorSetRoot(
               HIPKERNELPROVIDER_GENERAL_BWD_CORPUS_RELDIR)
           / fileName;
#else
    return {};
#endif
}

/// The entries of the corpus at @p path recorded for @p arch, in file order; the meta line is
/// skipped.
inline std::vector<json> loadCorpus(const std::filesystem::path& path, std::string_view arch)
{
    std::ifstream in(path);
    if(!in)
    {
        throw std::runtime_error("cannot open the corpus '" + path.string() + "'");
    }
    std::vector<json> entries;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty())
        {
            continue;
        }
        auto entry = json::parse(line);
        if(entry.contains("meta") || entry.value("arch", std::string()) != arch)
        {
            continue;
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

/// A kernarg's value as the corpus spells it: a pointer's role or "null", an integer in
/// decimal, an f32 as "0x%08x" of its bits.
inline std::string kernargText(const general_bwd::Kernarg& kernarg)
{
    switch(kernarg.kind)
    {
    case 'Q':
        return kernarg.role.empty() ? "null" : std::string(kernarg.role);
    case 'f':
    {
        std::array<char, 16> text{};
        std::snprintf(text.data(), text.size(), "0x%08x", static_cast<unsigned>(kernarg.bits));
        return text.data();
    }
    case 'i':
    case 'q':
        return std::to_string(kernarg.intValue());
    default:
        return "?";
    }
}

inline std::string corpusKernargText(const json& value)
{
    if(value.is_null())
    {
        return "null";
    }
    if(value.is_string())
    {
        return value.get<std::string>();
    }
    return std::to_string(value.get<int64_t>());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::general_bwd_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
