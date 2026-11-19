// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <hipdnn_data_sdk/types.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>

#include "harness/VariantPack.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"

namespace hipdnn_integration_tests::bundle::fixtures
{

/// One batchnorm-inference bundle on disk, as four harness suites all need it.
///
/// This graph used to be pasted into every suite that drives the harness. The
/// shape is arbitrary — one node, five leaf inputs, one output — and no test
/// asserts anything about batchnorm itself; it exists only so the harness has a
/// loadable bundle with a comparable output.
constexpr int64_t K_OUTPUT_UID = 5;
constexpr size_t K_OUTPUT_ELEMS = 120;
constexpr float K_OUTPUT_VALUE = 3.5f;

/// The ragged variant of the same bundle: x (uid 0) and y (uid 5) are logical
/// [B, H, S, D] = [2, 4, 3, 5] in the BSHD physical layout, ragged on the int32 offset
/// tensor uid 6. Its offsets must end at K_RAGGED_OUTPUT_ELEMS, the count writeRaggedOutput
/// fills.
constexpr int64_t K_RAGGED_OFFSET_UID = 6;
constexpr size_t K_RAGGED_OUTPUT_ELEMS = 60;

/// Writes a valid {name}.meta.json companion. Metadata is mandatory for a golden bundle
/// (one shipping output .bin blobs) and optional for a graph-only one.
inline void writeMetadata(const std::filesystem::path& dir, const std::string& name)
{
    std::ofstream(dir / (name + ".meta.json"))
        << R"({"format_version": 1, "operation": "BatchnormInference"})";
}

/// `outputDataType` is y's JSON data type. A y of any type other than "float" has an
/// all-zero golden, one byte per element, which is 0 in every fp8 format.
inline void writeBundleFiles(const std::filesystem::path& dir,
                             const std::string& name,
                             bool includeGoldenOutput,
                             const std::string& outputDataType = "float")
{
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json"))
        << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
           R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
           R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
           R"("compute_data_type": "float", "name": ""}], "tensors": [)"
           R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
           R"("data_type": ")"
        << outputDataType
        << R"(", "virtual": false}], "io_data_type": "float", )"
           R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";

    writeMetadata(dir, name);

    const auto basePath = (dir / name).string();
    const auto writeFloatBin = [&](int64_t uid, size_t elems, float value) {
        const std::vector<float> data(elems, value);
        std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    };

    writeFloatBin(0, 120, 0.0f); // x
    writeFloatBin(1, 3, 0.0f); // mean
    writeFloatBin(2, 3, 0.0f); // inv_variance
    writeFloatBin(3, 3, 0.0f); // scale
    writeFloatBin(4, 3, 0.0f); // bias

    if(includeGoldenOutput && outputDataType == "float")
    {
        writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE); // y (golden)
    }
    else if(includeGoldenOutput)
    {
        const std::vector<uint8_t> zeros(K_OUTPUT_ELEMS, 0);
        std::ofstream out(basePath + ".tensor" + std::to_string(K_OUTPUT_UID) + ".bin",
                          std::ios::binary);
        out.write(reinterpret_cast<const char*>(zeros.data()),
                  static_cast<std::streamsize>(zeros.size()));
    }
}

/// Writes the ragged bundle files. `goldenOutput` is y's ragged storage, which holds
/// `offsets.back()` elements. The offset tensor is always declared as uid 6, and last, so both
/// load passes must defer their ragged tensors. Ragged tensors reference `referencedOffsetUid`,
/// so any other value leaves their offset tensor missing.
inline void writeRaggedBundleFiles(const std::filesystem::path& dir,
                                   const std::string& name,
                                   const std::vector<int32_t>& offsets,
                                   const std::vector<float>& goldenOutput,
                                   int64_t referencedOffsetUid = K_RAGGED_OFFSET_UID)
{
    const std::string raggedOffsetField
        = R"(, "ragged_offset_tensor_uid": )" + std::to_string(referencedOffsetUid);

    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json"))
        << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
           R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
           R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
           R"("compute_data_type": "float", "name": ""}], "tensors": [)"
           R"({"name": "", "uid": 0, "strides": [60, 5, 20, 1], "dims": [2, 4, 3, 5], )"
           R"("data_type": "float", "virtual": false)"
        << raggedOffsetField
        << R"(}, )"
           R"({"name": "", "uid": 1, "strides": [4, 1, 1, 1], "dims": [1, 4, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 2, "strides": [4, 1, 1, 1], "dims": [1, 4, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 3, "strides": [4, 1, 1, 1], "dims": [1, 4, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 4, "strides": [4, 1, 1, 1], "dims": [1, 4, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 5, "strides": [60, 5, 20, 1], "dims": [2, 4, 3, 5], )"
           R"("data_type": "float", "virtual": false)"
        << raggedOffsetField
        << R"(}, )"
           R"({"name": "", "uid": 6, "strides": [1, 1, 1, 1], "dims": [3, 1, 1, 1], )"
           R"("data_type": "int32", "virtual": false}], "io_data_type": "float", )"
           R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";
    writeMetadata(dir, name);

    const auto basePath = (dir / name).string();
    const auto writeBin = [&](int64_t uid, const void* data, size_t byteCount) {
        std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin", std::ios::binary);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(byteCount));
    };

    const std::vector<float> raggedInput(static_cast<size_t>(offsets.back()), 0.0f);
    const std::vector<float> channelData(4, 0.0f);
    writeBin(0, raggedInput.data(), raggedInput.size() * sizeof(float));
    for(const int64_t uid : {1, 2, 3, 4})
    {
        writeBin(uid, channelData.data(), channelData.size() * sizeof(float));
    }
    writeBin(K_OUTPUT_UID, goldenOutput.data(), goldenOutput.size() * sizeof(float));
    writeBin(K_RAGGED_OFFSET_UID, offsets.data(), offsets.size() * sizeof(int32_t));
}

/// One ragged tensor's layout in a ragged SDPA bundle: logical dims, strides, the
/// int32 ragged_offset values (in units of `offsetMultiplier` elements) and the number of
/// elements its .bin blob holds. A blob that is not the tensor's elementSpace is what
/// the loader must reject.
struct RaggedLayout
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int32_t> offsets;
    int64_t offsetMultiplier = 1;
    size_t blobElems = 0;
};

/// Uids of the ragged SDPA forward bundle written by writeRaggedSdpaBundleFiles().
constexpr int64_t K_SDPA_Q_UID = 0;
constexpr int64_t K_SDPA_K_UID = 1;
constexpr int64_t K_SDPA_V_UID = 2;
constexpr int64_t K_SDPA_O_UID = 3;
constexpr int64_t K_SDPA_STATS_UID = 4;
constexpr int64_t K_SDPA_QKVO_OFFSET_UID = 5;
constexpr int64_t K_SDPA_STATS_OFFSET_UID = 6;
constexpr int64_t K_SDPA_SEQ_LEN_Q_UID = 7;
constexpr int64_t K_SDPA_SEQ_LEN_KV_UID = 8;

/// The graph of a one-node SDPA forward bundle whose Q, K, V and O share the `qkvo`
/// layout and ragged offset tensor, and whose Stats has its own `stats` layout and offset
/// tensor. A non-empty `seqLen` adds the seq_len_q and seq_len_kv tensors (both holding it),
/// which then give every ragged tensor its per-batch lengths. `indexType` is the data type
/// of the offset and seq_len tensors. With `raggedStats` false, Stats is a dense tensor of
/// the `stats` dims and strides and the graph has no Stats offset tensor. With `raggedKv`
/// false, K and V are dense tensors of the `qkvo` dims and strides.
inline nlohmann::json raggedSdpaGraph(const RaggedLayout& qkvo,
                                      const RaggedLayout& stats,
                                      const std::vector<int32_t>& seqLen = {},
                                      const std::string& indexType = "int32",
                                      bool raggedStats = true,
                                      bool raggedKv = true)
{
    using nlohmann::json;
    const bool hasSeqLen = !seqLen.empty();

    json inputs = {{"q_tensor_uid", K_SDPA_Q_UID},
                   {"k_tensor_uid", K_SDPA_K_UID},
                   {"v_tensor_uid", K_SDPA_V_UID}};
    for(const char* key : {"attn_mask_tensor_uid",
                           "scale_tensor_uid",
                           "seq_len_q_tensor_uid",
                           "seq_len_kv_tensor_uid",
                           "seed_tensor_uid",
                           "offset_tensor_uid",
                           "dropout_mask_tensor_uid",
                           "dropout_scale_tensor_uid",
                           "page_table_k_tensor_uid",
                           "page_table_v_tensor_uid",
                           "block_mask_tensor_uid",
                           "sink_token_tensor_uid",
                           "descale_q_tensor_uid",
                           "descale_k_tensor_uid",
                           "descale_v_tensor_uid",
                           "descale_s_tensor_uid",
                           "scale_s_tensor_uid",
                           "scale_o_tensor_uid"})
    {
        inputs[key] = nullptr;
    }
    if(hasSeqLen)
    {
        inputs["seq_len_q_tensor_uid"] = K_SDPA_SEQ_LEN_Q_UID;
        inputs["seq_len_kv_tensor_uid"] = K_SDPA_SEQ_LEN_KV_UID;
    }

    json outputs = {{"o_tensor_uid", K_SDPA_O_UID}, {"stats_tensor_uid", K_SDPA_STATS_UID}};
    for(const char* key : {"max_tensor_uid",
                           "sum_exp_tensor_uid",
                           "rng_dump_tensor_uid",
                           "amax_s_tensor_uid",
                           "amax_o_tensor_uid"})
    {
        outputs[key] = nullptr;
    }

    const json attributes = {{"generate_stats", true},
                             {"alibi_mask", false},
                             {"padding_mask", hasSeqLen},
                             {"causal_mask", false},
                             {"causal_mask_bottom_right", false},
                             {"dropout_probability", nullptr},
                             {"attn_scale_value", 1.0},
                             {"left_bound", nullptr},
                             {"right_bound", nullptr},
                             {"max_seq_len_kv", nullptr},
                             {"diagonal_alignment", "TOP_LEFT"},
                             {"mma_core_mode", "unset"},
                             {"implementation", "AUTO"}};

    json tensors = json::array();
    const auto addRagged = [&](int64_t uid, const RaggedLayout& layout, int64_t offsetUid) {
        tensors.push_back({{"uid", uid},
                           {"name", ""},
                           {"dims", layout.dims},
                           {"strides", layout.strides},
                           {"data_type", "float"},
                           {"virtual", false},
                           {"ragged_offset_tensor_uid", offsetUid},
                           {"ragged_offset_multiplier", layout.offsetMultiplier}});
    };
    const auto addAux = [&](int64_t uid, int64_t count) {
        tensors.push_back({{"uid", uid},
                           {"name", ""},
                           {"dims", std::vector<int64_t>{count, 1, 1, 1}},
                           {"strides", std::vector<int64_t>{1, 1, 1, 1}},
                           {"data_type", indexType},
                           {"virtual", false}});
    };
    for(const int64_t uid : {K_SDPA_Q_UID, K_SDPA_K_UID, K_SDPA_V_UID, K_SDPA_O_UID})
    {
        if(!raggedKv && (uid == K_SDPA_K_UID || uid == K_SDPA_V_UID))
        {
            tensors.push_back({{"uid", uid},
                               {"name", ""},
                               {"dims", qkvo.dims},
                               {"strides", qkvo.strides},
                               {"data_type", "float"},
                               {"virtual", false}});
            continue;
        }
        addRagged(uid, qkvo, K_SDPA_QKVO_OFFSET_UID);
    }
    if(raggedStats)
    {
        addRagged(K_SDPA_STATS_UID, stats, K_SDPA_STATS_OFFSET_UID);
    }
    else
    {
        tensors.push_back({{"uid", K_SDPA_STATS_UID},
                           {"name", ""},
                           {"dims", stats.dims},
                           {"strides", stats.strides},
                           {"data_type", "float"},
                           {"virtual", false}});
    }
    addAux(K_SDPA_QKVO_OFFSET_UID, static_cast<int64_t>(qkvo.offsets.size()));
    if(raggedStats)
    {
        addAux(K_SDPA_STATS_OFFSET_UID, static_cast<int64_t>(stats.offsets.size()));
    }
    if(hasSeqLen)
    {
        addAux(K_SDPA_SEQ_LEN_Q_UID, static_cast<int64_t>(seqLen.size()));
        addAux(K_SDPA_SEQ_LEN_KV_UID, static_cast<int64_t>(seqLen.size()));
    }

    return {{"nodes",
             json::array({{{"type", "SdpaAttributes"},
                           {"compute_data_type", "float"},
                           {"name", ""},
                           {"inputs", inputs},
                           {"outputs", outputs},
                           {"attributes", attributes}}})},
            {"tensors", tensors},
            {"io_data_type", "float"},
            {"compute_data_type", "float"},
            {"intermediate_data_type", "float"},
            {"name", ""}};
}

/// Writes raggedSdpaGraph() with int32 index tensors as a bundle. O and Stats carry golden
/// blobs.
inline void writeRaggedSdpaBundleFiles(const std::filesystem::path& dir,
                                       const std::string& name,
                                       const RaggedLayout& qkvo,
                                       const RaggedLayout& stats,
                                       const std::vector<int32_t>& seqLen = {})
{
    const bool hasSeqLen = !seqLen.empty();
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json")) << raggedSdpaGraph(qkvo, stats, seqLen).dump();
    std::ofstream(dir / (name + ".meta.json"))
        << R"({"format_version": 1, "operation": "SdpaFwd"})";

    const auto basePath = (dir / name).string();
    const auto writeBin = [&](int64_t uid, const void* data, size_t byteCount) {
        std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin", std::ios::binary);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(byteCount));
    };
    const std::vector<float> qkvoData(qkvo.blobElems, 0.0f);
    for(const int64_t uid : {K_SDPA_Q_UID, K_SDPA_K_UID, K_SDPA_V_UID, K_SDPA_O_UID})
    {
        writeBin(uid, qkvoData.data(), qkvoData.size() * sizeof(float));
    }
    const std::vector<float> statsData(stats.blobElems, 0.0f);
    writeBin(K_SDPA_STATS_UID, statsData.data(), statsData.size() * sizeof(float));
    writeBin(K_SDPA_QKVO_OFFSET_UID, qkvo.offsets.data(), qkvo.offsets.size() * sizeof(int32_t));
    writeBin(K_SDPA_STATS_OFFSET_UID, stats.offsets.data(), stats.offsets.size() * sizeof(int32_t));
    if(hasSeqLen)
    {
        writeBin(K_SDPA_SEQ_LEN_Q_UID, seqLen.data(), seqLen.size() * sizeof(int32_t));
        writeBin(K_SDPA_SEQ_LEN_KV_UID, seqLen.data(), seqLen.size() * sizeof(int32_t));
    }
}

/// A ragged SDPA bundle layout: what writeRaggedSdpaBundleFiles() is given, and the
/// elementSpace and per-batch lengths the Data SDK must derive for each of its tensors.
/// The names are the layout ids of the ragged footprint model (see RFC 0014 worked examples).
struct RaggedSdpaCase
{
    std::string name;
    RaggedLayout qkvo;
    RaggedLayout stats;
    std::vector<int32_t> seqLen;
    std::vector<int64_t> lengths;
    size_t qkvoSpace = 0;
    size_t statsSpace = 0;
};

/// Builds the case with both blobs sized to the elementSpace the Data SDK derives.
inline RaggedSdpaCase makeRaggedSdpaCase(std::string name,
                                         RaggedLayout qkvo,
                                         RaggedLayout stats,
                                         std::vector<int32_t> seqLen,
                                         std::vector<int64_t> lengths,
                                         size_t qkvoSpace,
                                         size_t statsSpace)
{
    qkvo.blobElems = qkvoSpace;
    stats.blobElems = statsSpace;
    return {std::move(name),
            std::move(qkvo),
            std::move(stats),
            std::move(seqLen),
            std::move(lengths),
            qkvoSpace,
            statsSpace};
}

/// Head-major primary [B,H,S,D] = [2,2,5,2] (head pitch S_max*D), offsets {0,4,10}.
inline RaggedSdpaCase raggedSdpaHeadMajorCase()
{
    return makeRaggedSdpaCase("L6",
                              {{2, 2, 5, 2}, {20, 10, 2, 1}, {0, 4, 10}},
                              {{2, 2, 5, 1}, {10, 5, 1, 1}, {0, 2, 5}},
                              {},
                              {2, 3},
                              20,
                              10);
}

/// Packed token-major primary with `[H,T]` Stats (head pitch = total tokens).
inline RaggedSdpaCase raggedSdpaHeadTokenStatsCase()
{
    return makeRaggedSdpaCase("L7a",
                              {{2, 4, 3, 2}, {24, 2, 8, 1}, {0, 16, 40}},
                              {{2, 4, 3, 1}, {20, 5, 1, 1}, {0, 2, 5}},
                              {},
                              {2, 3},
                              40,
                              20);
}

/// `[T,H]` Stats over the same primary as L7a: the same logical data, stored token-major.
inline RaggedSdpaCase raggedSdpaTokenHeadStatsCase()
{
    return makeRaggedSdpaCase("L7b",
                              {{2, 4, 3, 2}, {24, 2, 8, 1}, {0, 16, 40}},
                              {{2, 4, 3, 1}, {12, 1, 4, 1}, {0, 8, 20}},
                              {},
                              {2, 3},
                              40,
                              20);
}

/// `[H,T]` Stats whose head pitch (10) exceeds the token count (5).
inline RaggedSdpaCase raggedSdpaHeadPitchStatsCase()
{
    return makeRaggedSdpaCase("L7c",
                              {{2, 4, 3, 2}, {24, 2, 8, 1}, {0, 16, 40}},
                              {{2, 4, 3, 1}, {35, 10, 1, 1}, {0, 2, 5}},
                              {},
                              {2, 3},
                              40,
                              35);
}

/// Gapped `[H,T]` Stats with seq_len {2,3}: batch 1 starts at origin 4, past batch 0's rows.
inline RaggedSdpaCase raggedSdpaGappedStatsCase()
{
    return makeRaggedSdpaCase("L7d",
                              {{2, 2, 3, 2}, {12, 2, 4, 1}, {0, 8, 20}},
                              {{2, 2, 3, 1}, {14, 7, 1, 1}, {0, 4, 7}},
                              {2, 3},
                              {2, 3},
                              20,
                              14);
}

/// Shifted origin: batch 0 starts at element 4, no seq_len.
inline RaggedSdpaCase raggedSdpaShiftedOriginCase()
{
    return makeRaggedSdpaCase("L8",
                              {{2, 1, 4, 2}, {8, 2, 2, 1}, {4, 8, 14}},
                              {{2, 1, 4, 1}, {4, 1, 1, 1}, {2, 4, 7}},
                              {},
                              {2, 3},
                              14,
                              7);
}

/// Non-monotonic origins with seq_len {2,3}: batch 0 sits after batch 1.
inline RaggedSdpaCase raggedSdpaNonMonotonicCase()
{
    return makeRaggedSdpaCase("L9",
                              {{2, 1, 4, 2}, {8, 2, 2, 1}, {10, 0, 0}},
                              {{2, 1, 4, 1}, {4, 1, 1, 1}, {3, 0, 0}},
                              {2, 3},
                              {2, 3},
                              14,
                              5);
}

/// Trailing gap: the last batch's rows end at element 9, but offset[B] is 12, so a blob
/// sized to offset[B] is larger than elementSpace.
inline RaggedSdpaCase raggedSdpaTrailingGapCase()
{
    return makeRaggedSdpaCase("L14",
                              {{2, 1, 4, 2}, {8, 2, 2, 1}, {0, 4, 12}},
                              {{2, 1, 4, 1}, {4, 1, 1, 1}, {0, 2, 5}},
                              {2, 3},
                              {2, 3},
                              10,
                              5);
}

/// Every layout the host loader must read from a blob sized to its elementSpace.
inline std::vector<RaggedSdpaCase> raggedSdpaLoadableCases()
{
    return {raggedSdpaHeadMajorCase(),
            raggedSdpaHeadTokenStatsCase(),
            raggedSdpaTokenHeadStatsCase(),
            raggedSdpaHeadPitchStatsCase(),
            raggedSdpaGappedStatsCase(),
            raggedSdpaShiftedOriginCase(),
            raggedSdpaNonMonotonicCase(),
            raggedSdpaTrailingGapCase()};
}

/// Writes the case's bundle files.
inline void writeRaggedSdpaCase(const std::filesystem::path& dir, const RaggedSdpaCase& c)
{
    writeRaggedSdpaBundleFiles(dir, c.name, c.qkvo, c.stats, c.seqLen);
}

/// Fills the L7d bundle's O and Stats outputs in `variantPack` with `value`, each to the
/// elementSpace of its ragged layout.
inline void writeRaggedSdpaGappedOutputs(VariantPack& variantPack, float value)
{
    const auto c = raggedSdpaGappedStatsCase();
    auto* o = static_cast<float*>(variantPack.at(K_SDPA_O_UID));
    std::fill(o, o + c.qkvoSpace, value);
    auto* stats = static_cast<float*>(variantPack.at(K_SDPA_STATS_UID));
    std::fill(stats, stats + c.statsSpace, value);
}

/// Loads the bundle already written under `dir`.
inline std::shared_ptr<IntegrationTestBundle> loadWrittenBundle(const std::filesystem::path& dir,
                                                                const std::string& name)
{
    auto result = loadIntegrationTestBundle(dir / (name + ".json"));
    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    return std::make_shared<IntegrationTestBundle>(
        std::move(std::get<IntegrationTestBundle>(result)));
}

/// Writes the bundle under `parentDir/name` and loads it.
inline std::shared_ptr<IntegrationTestBundle> loadBundle(const std::filesystem::path& parentDir,
                                                         const std::string& name,
                                                         bool includeGoldenOutput,
                                                         const std::string& outputDataType
                                                         = "float")
{
    const auto dir = parentDir / name;
    writeBundleFiles(dir, name, includeGoldenOutput, outputDataType);
    return loadWrittenBundle(dir, name);
}

/// How writeRaggedSdpaRecipeBundle() writes a case.
struct RaggedRecipeOptions
{
    std::string indexType = "int32";
    bool raggedStats = true;
    bool raggedKv = true;
    /// Index tensors that get no `values` fill spec.
    std::set<int64_t> withoutValues;
    /// Keys merged into the metadata object.
    nlohmann::json extraMetadata = nlohmann::json::object();
};

/// Writes the case as a recipe bundle under `parentDir/name` and loads it: no blobs, and
/// format_version 2 metadata whose `inputs` give each offset and seq_len tensor its exact
/// values.
inline std::shared_ptr<IntegrationTestBundle>
    writeRaggedSdpaRecipeBundle(const std::filesystem::path& parentDir,
                                const std::string& name,
                                const RaggedSdpaCase& c,
                                const RaggedRecipeOptions& options = {})
{
    const auto dir = parentDir / name;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json"))
        << raggedSdpaGraph(
               c.qkvo, c.stats, c.seqLen, options.indexType, options.raggedStats, options.raggedKv)
               .dump();

    nlohmann::json inputs = nlohmann::json::object();
    const auto addValues = [&](int64_t uid, const std::vector<int32_t>& values) {
        if(options.withoutValues.count(uid) == 0)
        {
            inputs[std::to_string(uid)]
                = {{"kind", "values"}, {"values", values}, {"repeat", false}};
        }
    };
    addValues(K_SDPA_QKVO_OFFSET_UID, c.qkvo.offsets);
    if(options.raggedStats)
    {
        addValues(K_SDPA_STATS_OFFSET_UID, c.stats.offsets);
    }
    if(!c.seqLen.empty())
    {
        addValues(K_SDPA_SEQ_LEN_Q_UID, c.seqLen);
        addValues(K_SDPA_SEQ_LEN_KV_UID, c.seqLen);
    }
    nlohmann::json metadata
        = {{"format_version", 2}, {"operation", "SdpaFwd"}, {"seed", 7}, {"inputs", inputs}};
    metadata.update(options.extraMetadata);
    std::ofstream(dir / (name + ".meta.json")) << metadata.dump();
    return loadWrittenBundle(dir, name);
}

/// Writes and loads the golden bundle with mean (uid 1) as a [2,1,1,1] lengths tensor
/// holding `lengths` (INT32, or INT64 with `int64Lengths`). Golden y holds K_OUTPUT_VALUE
/// in the live rows (index s < lengths[b] on axis 2) and `pastValue` past them. With
/// `withLiveRows` the metadata declares those live rows.
inline std::shared_ptr<IntegrationTestBundle>
    writeLiveRowBundle(const std::filesystem::path& parentDir,
                       const std::string& name,
                       const std::vector<int64_t>& lengths,
                       float pastValue,
                       bool withLiveRows = true,
                       bool int64Lengths = false)
{
    const auto dir = parentDir / name;
    writeBundleFiles(dir, name, /*includeGoldenOutput=*/true);
    const auto basePath = (dir / name).string();

    std::ifstream graphFile(dir / (name + ".json"));
    auto graph = nlohmann::json::parse(graphFile);
    graphFile.close();
    for(auto& tensor : graph.at("tensors"))
    {
        if(tensor.at("uid") == 1)
        {
            tensor["data_type"] = int64Lengths ? "int64" : "int32";
            tensor["dims"] = {2, 1, 1, 1};
            tensor["strides"] = {1, 1, 1, 1};
        }
    }
    std::ofstream(dir / (name + ".json")) << graph.dump();

    std::ofstream lengthsFile(basePath + ".tensor1.bin", std::ios::binary);
    for(const int64_t length : lengths)
    {
        if(int64Lengths)
        {
            lengthsFile.write(reinterpret_cast<const char*>(&length), sizeof(length));
        }
        else
        {
            const auto narrow = static_cast<int32_t>(length);
            lengthsFile.write(reinterpret_cast<const char*>(&narrow), sizeof(narrow));
        }
    }
    lengthsFile.close();

    std::vector<float> golden(K_OUTPUT_ELEMS, pastValue);
    for(int64_t b = 0; b < 2; ++b)
    {
        for(int64_t h = 0; h < 3; ++h)
        {
            for(int64_t s = 0; s < 4 && s < lengths[static_cast<size_t>(b)]; ++s)
            {
                for(int64_t d = 0; d < 5; ++d)
                {
                    golden[static_cast<size_t>(b * 60 + h * 20 + s * 5 + d)] = K_OUTPUT_VALUE;
                }
            }
        }
    }
    std::ofstream(basePath + ".tensor5.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(golden.data()),
               static_cast<std::streamsize>(golden.size() * sizeof(float)));

    if(withLiveRows)
    {
        const auto metadata = nlohmann::json::parse(
            R"({"format_version": 2, "operation": "BatchnormInference",
                "live_rows": [{"uid": 5, "axis": 2, "lengths": 1}]})");
        std::ofstream(dir / (name + ".meta.json")) << metadata.dump();
    }
    return loadWrittenBundle(dir, name);
}

/// Fills the bundle's single output uid in `variantPack` with `value`.
inline void writeOutput(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_OUTPUT_ELEMS, value);
}

/// Fills the FP8_E4M3 output of a bundle loaded with outputDataType "fp8_e4m3".
inline void writeFp8E4M3Output(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<hipdnn_data_sdk::types::fp8_e4m3*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_OUTPUT_ELEMS, hipdnn_data_sdk::types::fp8_e4m3(value));
}

/// Fills the ragged bundle's output uid in `variantPack` with `value`.
inline void writeRaggedOutput(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_RAGGED_OUTPUT_ELEMS, value);
}

/// The template of `graph`: each tensor takes its dims, strides and data type from the
/// case.
inline nlohmann::json templateOfGraph(nlohmann::json graph)
{
    for(auto& tensor : graph.at("tensors"))
    {
        tensor["dims"] = "${case.dims}";
        tensor["strides"] = "${case.strides}";
        tensor["data_type"] = "${case.data_type}";
    }
    return graph;
}

/// A sweep case for the template of `graph`, with format_version 2 metadata. The case
/// marks `absentUids` absent, when there are any, and has no values for them.
inline nlohmann::json optionalTensorSweepCase(const std::string& id,
                                              const nlohmann::json& graph,
                                              const std::vector<int64_t>& absentUids = {})
{
    auto tensors = nlohmann::json::array();
    for(const auto& tensor : graph.at("tensors"))
    {
        const auto uid = tensor.at("uid").get<int64_t>();
        if(std::find(absentUids.begin(), absentUids.end(), uid) != absentUids.end())
        {
            continue;
        }
        tensors.push_back(nlohmann::json{{"uid", uid},
                                         {"dims", tensor.at("dims")},
                                         {"strides", tensor.at("strides")},
                                         {"data_type", tensor.at("data_type")}});
    }
    nlohmann::json caseJson
        = {{"id", id},
           {"values", {{"tensors", std::move(tensors)}}},
           {"metadata", {{"format_version", 2}, {"operation", "SdpaFwd"}, {"seed", 42}}}};
    if(!absentUids.empty())
    {
        caseJson["absent_tensors"] = absentUids;
    }
    return caseJson;
}

/// Writes graph.template.json and sweep.json under `dir`. A null `version` leaves the
/// sweep without one.
inline void writeTemplateSweep(const std::filesystem::path& dir,
                               const nlohmann::json& templateJson,
                               const nlohmann::json& cases,
                               const nlohmann::json& version = 2)
{
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "graph.template.json") << templateJson.dump(2);
    nlohmann::json sweep = {{"cases", cases}};
    if(!version.is_null())
    {
        sweep["version"] = version;
    }
    std::ofstream(dir / "sweep.json") << sweep.dump(2);
}

/// The uids of the host variant pack of `bundle`, built with a new tensor for each input
/// (each tensor that is not virtual and not an output) and each output.
inline std::set<int64_t> hostVariantPackUids(const IntegrationTestBundle& bundle)
{
    const auto wrapper = bundle.graphWrapper();
    const auto& attributes = wrapper.getTensorMap();
    const std::set<int64_t> outputUids(bundle.outputTensorUids.begin(),
                                       bundle.outputTensorUids.end());

    TensorMap inputs;
    for(const auto& [uid, tensorAttributes] : attributes)
    {
        if(!tensorAttributes->virtual_() && outputUids.count(uid) == 0)
        {
            inputs.emplace(uid,
                           hipdnn_test_sdk::detail::createTensorFromAttribute(*tensorAttributes));
        }
    }
    auto outputs = detail::allocateSentinelOutputs(
        attributes, bundle.outputTensorUids, inputs, {}, /*onDevice=*/false);
    const auto variantPack = detail::buildVariantPack(
        inputs, outputs, attributes, bundle.outputTensorUids, /*useDevice=*/false);

    std::set<int64_t> uids;
    for(const auto& entry : variantPack)
    {
        uids.insert(entry.first);
    }
    return uids;
}

} // namespace hipdnn_integration_tests::bundle::fixtures
