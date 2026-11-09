// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "harness/VariantPack.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

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

inline void writeBundleFiles(const std::filesystem::path& dir,
                             const std::string& name,
                             bool includeGoldenOutput)
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
           R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
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

    if(includeGoldenOutput)
    {
        writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE); // y (golden)
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

/// Writes a one-node SDPA forward bundle whose Q, K, V and O share the `qkvo` layout and
/// ragged offset tensor, and whose Stats has its own `stats` layout and offset tensor.
/// A non-empty `seqLen` adds the seq_len_q and seq_len_kv tensors (both holding it), which
/// then give every ragged tensor its per-batch lengths. O and Stats carry golden blobs.
inline void writeRaggedSdpaBundleFiles(const std::filesystem::path& dir,
                                       const std::string& name,
                                       const RaggedLayout& qkvo,
                                       const RaggedLayout& stats,
                                       const std::vector<int32_t>& seqLen = {})
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
                           {"data_type", "int32"},
                           {"virtual", false}});
    };
    for(const int64_t uid : {K_SDPA_Q_UID, K_SDPA_K_UID, K_SDPA_V_UID, K_SDPA_O_UID})
    {
        addRagged(uid, qkvo, K_SDPA_QKVO_OFFSET_UID);
    }
    addRagged(K_SDPA_STATS_UID, stats, K_SDPA_STATS_OFFSET_UID);
    addAux(K_SDPA_QKVO_OFFSET_UID, static_cast<int64_t>(qkvo.offsets.size()));
    addAux(K_SDPA_STATS_OFFSET_UID, static_cast<int64_t>(stats.offsets.size()));
    if(hasSeqLen)
    {
        addAux(K_SDPA_SEQ_LEN_Q_UID, static_cast<int64_t>(seqLen.size()));
        addAux(K_SDPA_SEQ_LEN_KV_UID, static_cast<int64_t>(seqLen.size()));
    }

    const json graph = {{"nodes",
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

    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json")) << graph.dump();
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
                                                         bool includeGoldenOutput)
{
    const auto dir = parentDir / name;
    writeBundleFiles(dir, name, includeGoldenOutput);
    return loadWrittenBundle(dir, name);
}

/// Fills the bundle's single output uid in `variantPack` with `value`.
inline void writeOutput(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_OUTPUT_ELEMS, value);
}

/// Fills the ragged bundle's output uid in `variantPack` with `value`.
inline void writeRaggedOutput(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_RAGGED_OUTPUT_ELEMS, value);
}

} // namespace hipdnn_integration_tests::bundle::fixtures
