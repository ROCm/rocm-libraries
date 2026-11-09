/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <gtest/gtest.h>
#include "gtest_common.hpp" // IsTestSupportedByDevice, Gpu
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp> // common::EngineeredConvFeatures
#include <miopen/env_debug.hpp>                     // debug::env::UpdateEnvVariable (table toggle)
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <string>
#include <map>
#include <set>
#include <numeric>
#include <optional>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

// Implemented in ai_heuristics.cpp; forward-declared (like the offline generator) so the key-format
// round-trip test can rebuild the vocabulary the same way generation and the enumeration path do.
namespace miopen {
namespace solver {
namespace conv {
void FillHeuristicKernels(const std::vector<std::string>& valid_kernels,
                          std::vector<int>& indexes,
                          std::vector<std::vector<std::string>>& kernels);
} // namespace conv
} // namespace solver
} // namespace miopen

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
using namespace miopen::ai::tuning::candidate_selection;

namespace {

struct CandidateSelectionParams
{
    std::string arch        = "gfx942";
    std::string solver      = "ConvHipImplicitGemm3DGroupWrwXdlops";
    std::string kernel_name = "DeviceGroupedConvBwdWeight_Xdl_CShuffle";
    int split_k             = 8;
};

void PrintTo(const CandidateSelectionParams& p, std::ostream* os)
{
    *os << p.arch << "_" << p.solver << "_" << p.kernel_name << "_splitk" << p.split_k;
}

// Default validation function: accepts all kernel/split_k combinations
inline constexpr auto accept_all_combinations = [](int, int) { return true; };

// Files required for every test: the input encoder (run live) and the metadata. The config encoder
// is NOT listed here -- it is deliberately dropped from the shipped artifacts (the precomputed
// kernel_config_embeddings table replaces it), so its absence must not fail file-presence checks.
auto GetCandidateSelectionModelFiles(const CandidateSelectionParams& params)
{
    const auto db_path = miopen::GetSystemDbPath();
    return std::vector<miopen::fs::path>{
        db_path / (params.arch + "_" + params.solver + "_input_encoder.tn.model"),
        db_path / (params.arch + "_" + params.solver + "_metadata.tn.model")};
}

// Note: there is no "skip if the config encoder is missing" guard any more. The config encoder is
// never shipped (the precomputed embedding table replaces it) and nothing here needs it -- the
// model constructor does not load it, EncodeKernelConfigs resolves through the table, and the
// input-feature tests use the input encoder. The fixture already skips/fails on missing input
// encoder + metadata.

auto GetMissingModelFiles(const CandidateSelectionParams& params)
{
    std::vector<miopen::fs::path> missing_files;
    for(const auto& file : GetCandidateSelectionModelFiles(params))
    {
        if(!miopen::fs::exists(file))
            missing_files.push_back(file);
    }
    return missing_files;
}

std::string FormatMissingFiles(const std::vector<miopen::fs::path>& missing_files)
{
    std::ostringstream os;
    for(const auto& file : missing_files)
        os << "\n  " << file.string();
    return os.str();
}

std::vector<std::vector<std::string>> GenerateValidKernelParams(
    const CandidateSelectionMetadata& meta, const std::string& kernel_name, int num_candidates = 3)
{
    const auto& kernel_str_mapping = meta.GetKernelStrMapping(kernel_name);
    std::vector<std::vector<std::string>> valid_kernel_params;
    for(int i = 0; i < num_candidates; ++i)
    {
        std::vector<std::string> candidate(meta.output_params().size(), "nan");
        candidate[0] = kernel_name;
        for(const auto& kv : kernel_str_mapping)
        {
            const std::string& param_name = kv.second;
            const std::string& index      = kv.first;
            const int index_int           = std::stoi(index);
            if(param_name.find("kernel_name") != std::string::npos)
                continue;
            auto it = meta.sequence_encodings().find(param_name);
            if(it == meta.sequence_encodings().end())
                candidate[index_int] = "0";
            else
                candidate[index_int] = it->second.begin()->first;
        }
        valid_kernel_params.push_back(candidate);
    }
    return valid_kernel_params;
}

class GPU_CandidateSelection_FP32 : public ::testing::TestWithParam<CandidateSelectionParams>
{
protected:
    miopen::Handle handle;

    void SetUp() override
    {
        const auto& params      = GetParam();
        const auto current_arch = handle.GetDeviceName();
        // Candidate-selection (KTN) models and the CK-IGEMM path ship only for gfx942/gfx950. On
        // any other device (e.g. RDNA gfx110X) there is nothing to exercise, so skip rather than
        // run the foreign-arch models on an unsupported device (they are shipped for every build
        // target).
        if(!IsTestSupportedByDevice(Gpu::gfx94X | Gpu::gfx950))
            GTEST_SKIP() << "Candidate selection ships for gfx942/gfx950 only; current device "
                         << current_arch << " is unsupported.";
        const auto missing = GetMissingModelFiles(params);

        if(missing.empty())
            return;

        const auto message = "Missing AI candidate selection model files for arch " + params.arch +
                             ", solver " + params.solver + ":" + FormatMissingFiles(missing);

        if(params.arch != current_arch)
        {
            GTEST_SKIP() << message << "\nCurrent device arch is " << current_arch
                         << "; skipping foreign-arch model validation.";
        }

        FAIL() << message << "\nCurrent device arch is " << current_arch
               << "; native-arch model files are required.";
    }
};

} // anonymous namespace

// === TESTS ===

TEST_P(GPU_CandidateSelection_FP32, FilesExist_Test)
{
    const auto missing = GetMissingModelFiles(GetParam());
    ASSERT_TRUE(missing.empty()) << "Missing AI candidate selection model files:"
                                 << FormatMissingFiles(missing);
}

TEST_P(GPU_CandidateSelection_FP32, MetadataAndModelInit_Test)
{
    const auto& params = GetParam();
    ASSERT_NO_THROW({
        CandidateSelectionMetadata meta(params.arch, params.solver);
        CandidateSelectionModel model(params.arch, params.solver);
    });
}

TEST_P(GPU_CandidateSelection_FP32, ModelCaching_Test)
{
    const auto& params = GetParam();
    auto& model1       = GetCandidateSelectionModel(params.arch, params.solver);
    auto& model2       = GetCandidateSelectionModel(params.arch, params.solver);
    ASSERT_EQ(&model1, &model2)
        << "GetCandidateSelectionModel did not return the same cached object!";
}

TEST_P(GPU_CandidateSelection_FP32, EncodeInputFeatures_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto encoded = model.EncodeInputFeatures(features);
    ASSERT_FALSE(encoded.empty()) << "EncodeInputFeatures returned empty vector!";
}

// (EncodeKernelConfigs on a synthetic candidate is covered by EncodeKernelConfigsEdgeCases: with
// the fdeep config tower removed, a candidate that is not an exact table key misses and throws, so
// there is no "encode an arbitrary candidate to a non-empty vector" case to assert -- only a hit,
// which requires a real enumerated config and is exercised end to end by the driver.)

TEST_P(GPU_CandidateSelection_FP32, EncodeInputFeaturesEdgeCases_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> empty_features;
    EXPECT_THROW(model.EncodeInputFeatures(empty_features), std::exception);
    std::map<std::string, float> long_features;
    for(const auto& name : meta.input_params())
        long_features[name] = 1.0f;
    long_features["extra_param"] = 2.0f;
    EXPECT_NO_THROW({
        auto encoded = model.EncodeInputFeatures(long_features);
        ASSERT_FALSE(encoded.empty());
    });
    if(!meta.GetConstantInputIndices().empty())
    {
        std::map<std::string, float> features;
        for(const auto& name : meta.input_params())
            features[name] = 1.0f;
        for(auto idx : meta.GetConstantInputIndices())
            if(idx < meta.input_params().size())
                features[meta.input_params()[idx]] = 42.0f;
        EXPECT_NO_THROW({
            auto encoded = model.EncodeInputFeatures(features);
            ASSERT_FALSE(encoded.empty());
        });
    }
}

// EncodeKernelConfigs resolves candidates only through the precomputed table (no fdeep fallback;
// the config encoder is not shipped), so it does NOT need the config encoder and runs without the
// SKIP guard. A miss -- empty batch, a dimension mismatch, or a right-width candidate whose exact
// key is not in the table (an un-enumerated config) -- must throw, which is what lets
// RunAIHeuristics drop to the non-AI heuristic.
TEST_P(GPU_CandidateSelection_FP32, EncodeKernelConfigsEdgeCases_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    size_t feature_size = meta.output_params().size() - meta.GetConstantOutputIndices().size();
    std::vector<std::vector<float>> empty_candidates;
    EXPECT_THROW(model.EncodeKernelConfigs(empty_candidates), std::exception);
    std::vector<std::vector<float>> candidates_short(1, std::vector<float>(feature_size - 1, 2.0f));
    EXPECT_THROW(model.EncodeKernelConfigs(candidates_short), std::exception);
    std::vector<std::vector<float>> candidates_long(1, std::vector<float>(feature_size + 1, 2.0f));
    EXPECT_THROW(model.EncodeKernelConfigs(candidates_long), std::exception);
    // Right width, but this value vector is not a real encoded config -> exact-key miss -> throw.
    std::vector<std::vector<float>> candidates_unknown(1, std::vector<float>(feature_size, 2.0f));
    EXPECT_THROW(model.EncodeKernelConfigs(candidates_unknown), std::exception);
}

TEST_P(GPU_CandidateSelection_FP32, KernelStrMappingUnknownKernelThrows_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    EXPECT_THROW(meta.GetKernelStrMapping("unknown_kernel_name"), std::exception);
}

TEST_P(GPU_CandidateSelection_FP32, OutputConstantRetrieval_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!meta.output_params().empty())
    {
        auto known = meta.GetOutputConstant(meta.output_params()[0]);
        SUCCEED();
    }
    auto unknown = meta.GetOutputConstant("nonexistent_param");
    EXPECT_EQ(unknown, std::nullopt);
}

TEST_P(GPU_CandidateSelection_FP32, InputOutputParamIndexThrows_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    EXPECT_THROW(meta.GetInputParamIndex("nonexistent_param"), std::exception);
    EXPECT_THROW(meta.GetOutputParamIndex("nonexistent_param"), std::exception);
}

TEST_P(GPU_CandidateSelection_FP32, EncodeKernelParamsBadValueThrows_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::vector<std::vector<std::string>> bad_params = {
        {params.kernel_name, "nonexistent_value", "nan"}};

    // The function should not throw, but should return empty result due to invalid mapping
    std::vector<std::vector<float>> result;
    EXPECT_NO_THROW(result = EncodeKernelParams(bad_params, meta, /*use_split_k=*/false));

    // Verify that the invalid candidate was skipped (empty result)
    EXPECT_TRUE(result.empty())
        << "Expected empty result when all candidates have invalid mappings";
}

TEST_P(GPU_CandidateSelection_FP32, SelectBestCandidateValid_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto encoded_features = model.EncodeInputFeatures(features);
    // Rank real config embeddings taken straight from the shipped vocab (exactly what a table hit
    // returns), so the selection is exercised without reconstructing a raw table key:
    // GenerateValidKernelParams builds synthetic configs that are not enumerated, hence not in the
    // table, and would miss.
    const auto& vocab = GetConfigEmbeddingVocab(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty()) << "shipped config-embedding vocab is empty";
    std::vector<std::vector<float>> encoded_configs;
    for(std::size_t i = 0; i < 3 && i < vocab.size(); ++i)
        encoded_configs.push_back(vocab[i].second);
    auto ids = model.SelectBestCandidateIndices(encoded_features, encoded_configs);
    ASSERT_FALSE(ids.empty()) << "No candidates were selected!";
    for(const auto& candidate : ids)
    {
        const int idx = candidate.first;
        ASSERT_GE(idx, 0) << "Candidate index is negative!";
        ASSERT_LT(idx, static_cast<int>(encoded_configs.size()))
            << "Candidate index " << idx << " out of range [0, " << encoded_configs.size() - 1
            << "]";
    }
}

TEST_P(GPU_CandidateSelection_FP32, SelectBestCandidateEmptyInput_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    std::vector<float> encoded_features;
    std::vector<std::vector<float>> encoded_configs;
    EXPECT_THROW(model.SelectBestCandidateIndices(encoded_features, encoded_configs),
                 std::exception);
}

TEST_P(GPU_CandidateSelection_FP32, ModelSelectBestCandidate_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto result              = ModelSelectBestCandidate(params.arch,
                                           params.solver,
                                           features,
                                           valid_kernel_params,
                                           /*use_split_k=*/false,
                                           accept_all_combinations);
    // GenerateValidKernelParams builds synthetic (un-enumerated) configs that are not in the table,
    // so each lookup misses: EncodeKernelConfigs throws, ModelSelectBestCandidate catches it and
    // returns an empty result -- the contract RunAIHeuristics relies on to drop to the non-AI
    // heuristic -- without throwing or crashing.
    EXPECT_TRUE(result.IsEmpty())
        << "un-enumerated candidates must yield an empty result (non-AI fallback)";
}

// End-to-end miss -> non-AI fallback. With the config table disabled, every EncodeKernelConfigs
// lookup misses and throws; ModelSelectBestCandidate must catch it and return an EMPTY result --
// which is what makes RunAIHeuristics return false and the solver drop to its non-AI heuristic --
// rather than throwing or crashing. (The per-call throw is covered by EncodeKernelConfigsEdgeCases;
// this proves the catch/empty contract the fallback relies on.)
TEST_P(GPU_CandidateSelection_FP32, ModelSelectBestCandidateTableMissFallsBack_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);

    constexpr std::string_view kDisable = "MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB";
    miopen::debug::env::UpdateEnvVariable(kDisable, "1");
    CandidateSelectionResult result;
    const auto ran = [&] {
        result = ModelSelectBestCandidate(params.arch,
                                          params.solver,
                                          features,
                                          valid_kernel_params,
                                          /*use_split_k=*/false,
                                          accept_all_combinations);
    };
    EXPECT_NO_THROW(ran());
    miopen::debug::env::ClearEnvVariable(
        kDisable); // restore before asserting, so leaks can't occur
    EXPECT_TRUE(result.IsEmpty())
        << "A config-table miss must yield an empty result (non-AI fallback), not a pick";
}

// Pins the key format the whole precompute design depends on. Rebuild each shipped table's keys the
// way the offline generator and the runtime enumeration path do -- the vocab config strings ->
// FillHeuristicKernels -> [ExpandKernelParamsWithSplitK] -> EncodeKernelParams -- and assert every
// resulting key is in the table and returns the stored embedding. If EncodeKernelParams or
// FillHeuristicKernels ever drift from how a table was generated, every lookup would miss and all
// solvers would silently fall to the non-AI heuristic; this catches that. CPU-only (no GPU
// kernels).
TEST_P(GPU_CandidateSelection_FP32, TableKeysRoundTripThroughEncodeKernelParams_Test)
{
    const auto& p = GetParam();
    CandidateSelectionMetadata meta(p.arch, p.solver);
    const auto& vocab = GetConfigEmbeddingVocab(p.arch, p.solver);
    ASSERT_FALSE(vocab.empty()) << "no v3 vocab for " << p.arch << " " << p.solver;

    const bool uses_split_k = p.solver.find("Fwd") == std::string::npos; // Bwd/Wrw use split_k

    // Recover the config-string vocabulary (strip the "+<split_k>" suffix that follows the final
    // '>').
    std::map<std::string, std::vector<float>> emb_by_id;
    std::vector<std::string> all_kernels;
    std::set<std::string> seen_cfg;
    for(const auto& [kid, emb] : vocab)
    {
        emb_by_id[kid]  = emb;
        std::string cfg = kid;
        const auto gt   = kid.rfind('>');
        const auto plus = kid.find('+', gt == std::string::npos ? 0 : gt);
        if(plus != std::string::npos)
            cfg = kid.substr(0, plus);
        if(seen_cfg.insert(cfg).second)
            all_kernels.push_back(cfg);
    }

    std::vector<int> indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    miopen::solver::conv::FillHeuristicKernels(all_kernels, indexes, heuristic_kernels);

    std::vector<std::vector<std::string>> params;
    std::vector<std::string> param_kernel_ids;
    if(uses_split_k)
    {
        const std::vector<int>& split_ks =
            meta.GetSplitKValues(); // the set the table was built with
        std::vector<int> hidx(heuristic_kernels.size());
        std::iota(hidx.begin(), hidx.end(), 0);
        auto [expanded, mapping] = ExpandKernelParamsWithSplitK(
            heuristic_kernels, hidx, split_ks, accept_all_combinations);
        params = std::move(expanded);
        param_kernel_ids.resize(params.size());
        for(std::size_t i = 0; i < params.size() && i < mapping.size(); ++i)
        {
            const int kidx = mapping[i].first;
            if(kidx >= 0 && static_cast<std::size_t>(kidx) < indexes.size())
                param_kernel_ids[i] =
                    all_kernels[indexes[kidx]] + "+" + std::to_string(mapping[i].second);
        }
    }
    else
    {
        params = heuristic_kernels;
        param_kernel_ids.resize(params.size());
        for(std::size_t j = 0; j < params.size() && j < indexes.size(); ++j)
            param_kernel_ids[j] = all_kernels[indexes[j]];
    }

    const auto encoded = EncodeKernelParams(params, meta, uses_split_k);
    ASSERT_EQ(encoded.size(), param_kernel_ids.size());

    std::size_t checked = 0;
    for(std::size_t i = 0; i < encoded.size(); ++i)
    {
        const auto it = emb_by_id.find(param_kernel_ids[i]);
        if(it == emb_by_id.end())
            continue; // sentinel/unknown, deduped, or not a vocab row
        const auto hit = TryEncodeKernelConfigsFromTable({encoded[i]}, p.arch, p.solver);
        ASSERT_TRUE(hit.has_value()) << "EncodeKernelParams key for vocab id '"
                                     << param_kernel_ids[i] << "' is not in the shipped table";
        ASSERT_EQ(hit->size(), 1u);
        EXPECT_EQ((*hit)[0], it->second)
            << "table returned a different embedding for '" << param_kernel_ids[i] << "'";
        ++checked;
    }
    EXPECT_GT(checked, 0u) << "no vocab keys were round-tripped";
    EXPECT_GE(checked, vocab.size() / 2)
        << "too few vocab keys matched (" << checked << "/" << vocab.size() << ")";
}

TEST_P(GPU_CandidateSelection_FP32, ExpandKernelParamsWithSplitK_Test)
{
    const auto& params                            = GetParam();
    std::vector<std::vector<std::string>> kernels = {{"typeA", "p1"}, {"typeB", "p2"}};
    std::vector<int> indexes                      = {0, 1};
    std::vector<int> split_ks = miopen::solver::conv::GenerateSplitK(params.split_k);
    auto [expanded, mapping] =
        ExpandKernelParamsWithSplitK(kernels, indexes, split_ks, accept_all_combinations);
    ASSERT_EQ(expanded.size(), 8u);
    ASSERT_EQ(mapping.size(), 8u);
    std::vector<std::vector<std::string>> expected_expanded = {
        {"typeA", "p1", "1"},
        {"typeA", "p1", "2"},
        {"typeA", "p1", "4"},
        {"typeA", "p1", "8"},
        {"typeB", "p2", "1"},
        {"typeB", "p2", "2"},
        {"typeB", "p2", "4"},
        {"typeB", "p2", "8"},
    };
    std::vector<std::pair<int, int>> expected_mapping = {
        {0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 1}, {1, 2}, {1, 4}, {1, 8}};
    for(size_t i = 0; i < expanded.size(); ++i)
    {
        ASSERT_EQ(expanded[i], expected_expanded[i]);
        ASSERT_EQ(mapping[i], expected_mapping[i]);
    }
}

TEST_P(GPU_CandidateSelection_FP32, ExpandKernelParamsWithSplitKFunctionality_Test)
{
    const auto& params                            = GetParam();
    std::vector<std::vector<std::string>> kernels = {
        {"DeviceGroupedConvBwdWeight_Xdl_CShuffle", "p1"}};
    std::vector<int> indexes  = {0};
    std::vector<int> split_ks = miopen::solver::conv::GenerateSplitK(params.split_k);
    auto [expanded, mapping] =
        ExpandKernelParamsWithSplitK(kernels, indexes, split_ks, accept_all_combinations);
    ASSERT_EQ(expanded.size(), split_ks.size());
    ASSERT_EQ(mapping.size(), split_ks.size());
    for(size_t i = 0; i < split_ks.size(); ++i)
    {
        ASSERT_EQ(expanded[i][0], "DeviceGroupedConvBwdWeight_Xdl_CShuffle");
        ASSERT_EQ(expanded[i][2], std::to_string(split_ks[i]));
        ASSERT_EQ(mapping[i].first, 0);
        ASSERT_EQ(mapping[i].second, split_ks[i]);
    }
}

// === Wavelet / conditional-kernel decode tests ===
//
// The Wavelet WRW kernel (DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3) has a variable
// type-string length and a packed first token "{load}l+{math}m". These tests exercise the
// special-case decode in EncodeKernelParams (see ai_candidate_selection.cpp). They only run on the
// WRW solvers whose metadata actually contains the Wavelet kernel.
//
// Candidate layout (0-based, kernel_name prepended at [0]):
//   [0] kernel_name
//   [1] "{load}l+{math}m"  (CK type-string index 0)
//   [2..15] 14 remaining base params (CK indices 1..14)  -> 15 base tokens total, base len 16
//   [16] NumGroupsToMerge  (CK index 15, present only when > 1)
//   [last] SplitK          (appended by ExpandKernelParamsWithSplitK when use_split_k)

namespace {
constexpr const char* kWaveletKernel = "DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3";

bool MetadataHasKernel(const CandidateSelectionMetadata& meta, const std::string& kernel_name)
{
    try
    {
        meta.GetKernelStrMapping(kernel_name);
        return true;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

// Find the slot of an output param (by suffix) within the *encoded* vector, i.e. the index after
// constant output params have been dropped. Returns -1 if absent or constant.
int EncodedSlotForSuffix(const CandidateSelectionMetadata& meta, const std::string& suffix)
{
    int slot = 0;
    for(const auto& param_name : meta.output_params())
    {
        if(meta.GetOutputConstant(param_name).has_value())
            continue; // constant params are skipped in the encoded vector
        if(param_name.size() >= suffix.size() &&
           param_name.compare(param_name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return slot;
        ++slot;
    }
    return -1;
}

// Return the metadata constant for an output param matched by suffix (e.g. the value a retrain
// pinned a feature to when it was constant in the training set), or std::nullopt when no output
// param with that suffix is a declared constant.
std::optional<std::string> OutputConstantForSuffix(const CandidateSelectionMetadata& meta,
                                                   const std::string& suffix)
{
    for(const auto& param_name : meta.output_params())
    {
        if(param_name.size() >= suffix.size() &&
           param_name.compare(param_name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return meta.GetOutputConstant(param_name);
    }
    return std::nullopt;
}

// Build a Wavelet candidate. Base form (no NGM, no SplitK) has length 16: kernel_name + packed
// token + 14 numeric base params. include_num_groups appends NumGroupsToMerge at [16] (length 17).
std::vector<std::string> MakeWaveletCandidate(bool include_num_groups, const std::string& ngm = "2")
{
    std::vector<std::string> c;
    c.push_back(kWaveletKernel); // [0]
    c.push_back("128l+256m");    // [1] packed TileLoadMath token
    for(int i = 0; i < 14; ++i)  // [2..15] remaining 14 base params
        c.push_back("1");
    if(include_num_groups)
        c.push_back(ngm); // [16] NumGroupsToMerge (CK emits only when > 1)
    return c;
}

// The CShuffleV3 large-tensor kernel emits BlkGemmPipelineScheduler / -Version
// ("inline_after_optionals") right after the optional NumGroupsToMerge, then the appended SplitK.
// base_param_count is 15, so the candidate is kernel_name + 15 base params, then (optional NGM),
// scheduler, version, then the appended SplitK. CK keeps the "BlkGemmPipelineScheduler:" label on
// these tokens at inference, so that is what the candidate carries here.
constexpr const char* kV3LargeTensorKernel =
    "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3_Large_Tensor";

std::vector<std::string> MakeV3LargeTensorCandidate(bool include_num_groups,
                                                    const std::string& scheduler,
                                                    const std::string& version,
                                                    const std::string& ngm = "2")
{
    std::vector<std::string> c;
    c.push_back(kV3LargeTensorKernel); // [0]
    for(int i = 0; i < 15; ++i)        // [1..15] 15 base params
        c.push_back("1");
    if(include_num_groups)
        c.push_back(ngm);   // NumGroupsToMerge (present only when > 1)
    c.push_back(scheduler); // inline-after-optionals "BlkGemmPipelineScheduler:..."
    c.push_back(version);   // inline-after-optionals "BlkGemmPipelineVersion:..."
    return c;
}

// A complete input-feature map for EngineerCandidateSelectionInputFeatures (every FeatureAt key),
// so a throw can only come from the precision-class check, not a missing feature.
std::map<std::string, float> MakeFullInputFeatureMap()
{
    return {
        {"direction", 0.0f},     {"batchsize", 1.0f},   {"in_channels", 64.0f},
        {"out_channels", 64.0f}, {"in_h", 56.0f},       {"in_w", 56.0f},
        {"out_h", 56.0f},        {"out_w", 56.0f},      {"fil_h", 3.0f},
        {"fil_w", 3.0f},         {"group_count", 1.0f}, {"in_layout", 1.0f},
        {"fil_layout", 1.0f},    {"out_layout", 1.0f},  {"conv_stride_h", 1.0f},
        {"conv_stride_w", 1.0f}, {"dilation_h", 1.0f},  {"dilation_w", 1.0f},
        {"pad_h", 1.0f},         {"pad_w", 1.0f},       {"precision", 0.0f},
    };
}
} // namespace

TEST_P(GPU_CandidateSelection_FP32, WaveletSplitTileLoadMath_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kWaveletKernel))
        GTEST_SKIP() << "Wavelet kernel not present in metadata for " << params.solver;

    auto candidate = MakeWaveletCandidate(/*include_num_groups=*/false); // length 16
    std::vector<std::vector<std::string>> kparams{candidate};

    auto encoded = EncodeKernelParams(kparams, meta, /*use_split_k=*/false);
    ASSERT_EQ(encoded.size(), 1u) << "Wavelet candidate was unexpectedly skipped";

    // Validate that the packed "{load}l+{math}m" token decodes its math part to 256. Whether
    // TileMathThreadGroupSize survives as a live encoded feature depends on the trained model: the
    // 2D Wrw models keep it live (TileLoad is the constant), whereas the retrained 3D Wrw models
    // pin *both* TileLoad and TileMath to a constant. Assert against whichever form the shipped
    // metadata declares so this stays correct across retrains.
    const std::string math_suffix = "TileMathThreadGroupSize";
    const int math_slot           = EncodedSlotForSuffix(meta, math_suffix);
    if(math_slot >= 0)
    {
        EXPECT_FLOAT_EQ(encoded[0][math_slot], 256.0f)
            << "math part of '128l+256m' not decoded to 256";
    }
    else
    {
        // Dropped as a constant: the value is fixed in metadata, not emitted into the vector.
        const auto math_const = OutputConstantForSuffix(meta, math_suffix);
        ASSERT_TRUE(math_const.has_value())
            << "TileMathThreadGroupSize is neither a live encoded feature nor a metadata constant";
        EXPECT_EQ(*math_const, "256") << "constant TileMathThreadGroupSize does not match 256";
    }
}

TEST_P(GPU_CandidateSelection_FP32, WaveletNumGroupsConditional_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kWaveletKernel))
        GTEST_SKIP() << "Wavelet kernel not present in metadata for " << params.solver;

    const int ngm_slot = EncodedSlotForSuffix(meta, "NumGroupsToMerge");
    ASSERT_GE(ngm_slot, 0);
    const float missing = meta.GetMissingValueToken();

    // Present: NumGroupsToMerge="2" (candidate length 17) -> decoded (not missing).
    {
        auto candidate = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
        auto encoded   = EncodeKernelParams({candidate}, meta, /*use_split_k=*/false);
        ASSERT_EQ(encoded.size(), 1u);
        EXPECT_NE(encoded[0][ngm_slot], missing)
            << "NumGroupsToMerge present but decoded as missing";
    }
    // Absent: base candidate (length 16) -> NumGroupsToMerge resolves to the missing token.
    {
        auto candidate = MakeWaveletCandidate(/*include_num_groups=*/false);
        auto encoded   = EncodeKernelParams({candidate}, meta, /*use_split_k=*/false);
        ASSERT_EQ(encoded.size(), 1u);
        EXPECT_FLOAT_EQ(encoded[0][ngm_slot], missing)
            << "NumGroupsToMerge absent but not decoded as missing token";
    }
}

TEST_P(GPU_CandidateSelection_FP32, WaveletSplitKResolution_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kWaveletKernel))
        GTEST_SKIP() << "Wavelet kernel not present in metadata for " << params.solver;

    const int splitk_slot = EncodedSlotForSuffix(meta, "SplitK");
    ASSERT_GE(splitk_slot, 0);
    const float missing = meta.GetMissingValueToken();

    // With use_split_k: SplitK is the appended last token (NGM present + appended splitk -> len
    // 18).
    {
        auto candidate = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
        candidate.push_back("8"); // appended SplitK (mirrors ExpandKernelParamsWithSplitK)
        auto encoded = EncodeKernelParams({candidate}, meta, /*use_split_k=*/true);
        ASSERT_EQ(encoded.size(), 1u);
        EXPECT_FLOAT_EQ(encoded[0][splitk_slot], 8.0f) << "appended SplitK not decoded to 8";
    }
    // Without use_split_k: SplitK must resolve to the missing token.
    {
        auto candidate = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
        auto encoded   = EncodeKernelParams({candidate}, meta, /*use_split_k=*/false);
        ASSERT_EQ(encoded.size(), 1u);
        EXPECT_FLOAT_EQ(encoded[0][splitk_slot], missing)
            << "SplitK decoded as present when use_split_k is false";
    }
}

// The pipeline params (BlkGemmPipelineScheduler/-Version) in the CShuffleV3 large-tensor kernel are
// "inline_after_optionals": their position shifts with the optional NumGroupsToMerge. This verifies
// that the trailing SplitK is decoded from the correct (appended last) token rather than a
// pipeline-param token, and that the pipeline params themselves decode.
TEST_P(GPU_CandidateSelection_FP32, V3LargeTensorPipelineParamDecode_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kV3LargeTensorKernel))
        GTEST_SKIP() << "V3 large-tensor kernel not present in metadata for " << params.solver;
    if(meta.GetConditionalLayout(kV3LargeTensorKernel) == nullptr)
        GTEST_SKIP() << "V3 large-tensor kernel has no conditional layout in metadata for "
                     << params.solver;

    const int splitk_slot = EncodedSlotForSuffix(meta, "SplitK");
    const int sched_slot  = EncodedSlotForSuffix(meta, "BlkGemmPipelineScheduler");
    ASSERT_GE(splitk_slot, 0);
    ASSERT_GE(sched_slot, 0) << "scheduler expected to be a live (encoded) feature";
    const float missing = meta.GetMissingValueToken();

    // NumGroups omitted (the g=1 case): [16]=scheduler, [17]=version, [18]=appended SplitK.
    {
        auto candidate = MakeV3LargeTensorCandidate(/*include_num_groups=*/false,
                                                    "BlkGemmPipelineScheduler:Intrawave",
                                                    "BlkGemmPipelineVersion:v4");
        candidate.push_back("8"); // appended SplitK (mirrors ExpandKernelParamsWithSplitK)
        auto encoded = EncodeKernelParams({candidate}, meta, /*use_split_k=*/true);
        ASSERT_EQ(encoded.size(), 1u) << "V3 large-tensor candidate was unexpectedly skipped";
        EXPECT_FLOAT_EQ(encoded[0][splitk_slot], 8.0f)
            << "SplitK must decode from the appended token, not a pipeline-param token";
        EXPECT_NE(encoded[0][sched_slot], missing) << "scheduler token not decoded";
    }
    // NumGroups present: scheduler shifts right by one but is still resolved by position.
    {
        auto candidate = MakeV3LargeTensorCandidate(/*include_num_groups=*/true,
                                                    "BlkGemmPipelineScheduler:Interwave",
                                                    "BlkGemmPipelineVersion:v1",
                                                    "2");
        candidate.push_back("4"); // appended SplitK
        auto encoded = EncodeKernelParams({candidate}, meta, /*use_split_k=*/true);
        ASSERT_EQ(encoded.size(), 1u);
        EXPECT_FLOAT_EQ(encoded[0][splitk_slot], 4.0f)
            << "SplitK mis-decoded with NumGroups present";
        EXPECT_NE(encoded[0][sched_slot], missing);
    }
    // The scheduler value is actually read: Intrawave vs Interwave must encode differently.
    {
        auto intra = MakeV3LargeTensorCandidate(
            false, "BlkGemmPipelineScheduler:Intrawave", "BlkGemmPipelineVersion:v4");
        auto inter = MakeV3LargeTensorCandidate(
            false, "BlkGemmPipelineScheduler:Interwave", "BlkGemmPipelineVersion:v4");
        auto e_intra = EncodeKernelParams({intra}, meta, /*use_split_k=*/false);
        auto e_inter = EncodeKernelParams({inter}, meta, /*use_split_k=*/false);
        ASSERT_EQ(e_intra.size(), 1u);
        ASSERT_EQ(e_inter.size(), 1u);
        EXPECT_NE(e_intra[0][sched_slot], e_inter[0][sched_slot])
            << "Intrawave vs Interwave must encode to different values";
    }
}

// A candidate carrying a non-numeric value in a numeric output slot must be ranked last via the
// all-NaN sentinel, not abort encoding for every candidate. Here the appended SplitK token is
// non-numeric; the other (valid) candidate must still encode normally.
TEST_P(GPU_CandidateSelection_FP32, EncodeFailureSkipsCandidate_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kWaveletKernel))
        GTEST_SKIP() << "Wavelet kernel not present in metadata for " << params.solver;

    auto good = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
    good.push_back("8"); // valid appended SplitK
    auto bad = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
    bad.push_back("not_a_number"); // non-numeric appended SplitK -> unparseable numeric slot

    std::vector<std::vector<float>> encoded;
    ASSERT_NO_THROW(encoded = EncodeKernelParams({good, bad}, meta, /*use_split_k=*/true));
    ASSERT_EQ(encoded.size(), 2u) << "both candidates must produce an encoding row";

    const auto is_nan = [](float v) { return std::isnan(v); };
    EXPECT_FALSE(std::all_of(encoded[0].begin(), encoded[0].end(), is_nan))
        << "valid candidate must not be sentinel'd";
    ASSERT_FALSE(encoded[1].empty());
    EXPECT_TRUE(std::all_of(encoded[1].begin(), encoded[1].end(), is_nan))
        << "candidate with unparseable numeric value must be the all-NaN sentinel";
    EXPECT_EQ(encoded[0].size(), encoded[1].size()) << "sentinel width must match a normal row";
}

// A precision the model wasn't trained on (e.g. INT8 on a model whose precision encoding lacks it,
// such as the 3-class Wrw/Bwd models) must throw so the caller falls back to the non-AI heuristic,
// rather than encoding an all-zero precision one-hot.
TEST_P(GPU_CandidateSelection_FP32, UnsupportedPrecisionThrows_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    const auto spatial_dim = meta.GetInputConstant("spatial_dim");
    if(!spatial_dim.has_value() || *spatial_dim != "2")
        GTEST_SKIP() << "Engineered input features are 2D-only; " << params.solver << " is not 2D";

    auto feats = MakeFullInputFeatureMap();

    // A supported precision (FP32) never throws (also proves the feature map is complete).
    feats["precision"] = static_cast<float>(miopenFloat);
    EXPECT_NO_THROW(EngineerCandidateSelectionInputFeatures(feats, meta));

    // INT8 throws iff this model's precision encoding cannot represent it.
    feats["precision"]     = static_cast<float>(miopenInt8);
    const bool int8_in_enc = meta.GetInputEncodingIndex("precision", "INT8") <
                             meta.GetInputEncodingClassCount("precision");
    if(int8_in_enc)
        EXPECT_NO_THROW(EngineerCandidateSelectionInputFeatures(feats, meta));
    else
        EXPECT_ANY_THROW(EngineerCandidateSelectionInputFeatures(feats, meta));
}

// The engineered input vector must stay stable across refactors. This pins the assembly order and
// the metadata-driven quantities (layout/precision one-hot widths and indices, and num_cu in the
// derived block) by composing the expected vector from its components; the derived-feature math
// itself is pinned deterministically by CPU_ConvAiEngineeredConvFeatures_NONE.Golden.
TEST_P(GPU_CandidateSelection_FP32, EngineeredInputGolden_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    const auto spatial_dim = meta.GetInputConstant("spatial_dim");
    if(!spatial_dim.has_value() || *spatial_dim != "2")
        GTEST_SKIP() << "Engineered input features are 2D-only; " << params.solver << " is not 2D";

    auto feats         = MakeFullInputFeatureMap();
    feats["precision"] = static_cast<float>(miopenFloat); // FP32

    const auto engineered = EngineerCandidateSelectionInputFeatures(feats, meta);

    std::vector<float> expected;
    // Layout one-hots: in/fil/out, each set to class index 1 (NHWC) by MakeFullInputFeatureMap.
    const std::size_t layout_w = meta.GetInputEncodingClassCount("in_layout");
    for(int l = 0; l < 3; ++l)
    {
        std::vector<float> one_hot(layout_w, 0.0f);
        one_hot.at(1) = 1.0f;
        expected.insert(expected.end(), one_hot.begin(), one_hot.end());
    }
    // Precision one-hot: FP32, width and index from the metadata.
    {
        std::vector<float> one_hot(meta.GetInputEncodingClassCount("precision"), 0.0f);
        one_hot.at(meta.GetInputEncodingIndex("precision", "FP32")) = 1.0f;
        expected.insert(expected.end(), one_hot.begin(), one_hot.end());
    }
    // Raw passthrough: C_in,H_in,W_in,C_out,H_out,W_out,K_h,K_w then
    // pad/stride/dilation/batch/group.
    const std::vector<float> raw = {64.0f,
                                    56.0f,
                                    56.0f,
                                    64.0f,
                                    56.0f,
                                    56.0f,
                                    3.0f,
                                    3.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f};
    expected.insert(expected.end(), raw.begin(), raw.end());
    // Derived block (shared math), normalized with the metadata's num_cu. direction=0 (Forward) and
    // C_in==C_out here, so the direction-dependent GEMM assignment is exercised by the CPU golden.
    const auto derived =
        miopen::ai::common::EngineeredConvFeatures(1,
                                                   64,
                                                   64,
                                                   56,
                                                   56,
                                                   56,
                                                   56,
                                                   3,
                                                   3,
                                                   1,
                                                   meta.GetNumCu(),
                                                   miopen::ai::common::ConvDirection::Forward);
    expected.insert(expected.end(), derived.begin(), derived.end());

    ASSERT_EQ(engineered.size(), expected.size());
    for(std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(engineered[i], expected[i]) << "engineered feature mismatch at index " << i;
}

// Hardcoded counterpart to EngineeredInputGolden_Test: the one-hot widths/indices and num_cu are
// written as literals rather than re-derived from the metadata, so a retrain that silently changes
// the encoding class counts or num_cu is caught here (the golden test above composes its expected
// vector from the same metadata accessors and so cannot catch such drift).
TEST_P(GPU_CandidateSelection_FP32, EngineeredInputHardcoded_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    const auto spatial_dim = meta.GetInputConstant("spatial_dim");
    if(!spatial_dim.has_value() || *spatial_dim != "2")
        GTEST_SKIP() << "Engineered input features are 2D-only; " << params.solver << " is not 2D";

    // Contract of the shipped 2D candidate-selection metadata, as literals. The Fwd input encoder
    // carries an extra INT8 precision class (4 vs 3); FP32 is index 2 in both.
    const bool is_fwd                      = params.solver.find("Fwd") != std::string::npos;
    const std::size_t expected_precision_w = is_fwd ? 4u : 3u;
    const std::size_t expected_num_cu      = (params.arch == "gfx942") ? 304u : 256u;
    ASSERT_EQ(meta.GetInputEncodingClassCount("in_layout"), 2u); // NCHW, NHWC
    ASSERT_EQ(meta.GetInputEncodingClassCount("fil_layout"), 2u);
    ASSERT_EQ(meta.GetInputEncodingClassCount("out_layout"), 2u);
    ASSERT_EQ(meta.GetInputEncodingClassCount("precision"), expected_precision_w);
    ASSERT_EQ(meta.GetInputEncodingIndex("precision", "FP32"), 2u);
    ASSERT_EQ(meta.GetNumCu(), expected_num_cu);

    auto feats            = MakeFullInputFeatureMap();
    feats["precision"]    = static_cast<float>(miopenFloat); // FP32
    const auto engineered = EngineerCandidateSelectionInputFeatures(feats, meta);

    // MakeFullInputFeatureMap sets all layouts to NHWC (index 1 of 2); FP32 is index 2.
    std::vector<float> expected = {
        0.0f,
        1.0f, // in_layout  NHWC
        0.0f,
        1.0f, // fil_layout NHWC
        0.0f,
        1.0f, // out_layout NHWC
    };
    std::vector<float> precision_one_hot(expected_precision_w, 0.0f);
    precision_one_hot.at(2) = 1.0f; // FP32
    expected.insert(expected.end(), precision_one_hot.begin(), precision_one_hot.end());

    const std::vector<float> raw = {64.0f,
                                    56.0f,
                                    56.0f,
                                    64.0f,
                                    56.0f,
                                    56.0f,
                                    3.0f,
                                    3.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f,
                                    1.0f};
    expected.insert(expected.end(), raw.begin(), raw.end());
    const auto derived =
        miopen::ai::common::EngineeredConvFeatures(1,
                                                   64,
                                                   64,
                                                   56,
                                                   56,
                                                   56,
                                                   56,
                                                   3,
                                                   3,
                                                   1,
                                                   expected_num_cu,
                                                   miopen::ai::common::ConvDirection::Forward);
    expected.insert(expected.end(), derived.begin(), derived.end());

    ASSERT_EQ(engineered.size(), expected.size());
    for(std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(engineered[i], expected[i])
            << "hardcoded engineered feature mismatch at index " << i;
}

// The engineered input path became N-D in this PR, but EngineeredInput*_Test above are 2D-only and
// Forward-only -- nothing exercised the 3D path or the Backward in<->out swap. This runs the 3D
// engineered input path in a non-Forward direction, pinning both new swaps: (a) the in_d<->out_d
// swap in CandidateSelectionRawFeatureValue (raw passthrough) and (b) the D_in/D_out swap feeding
// common::EngineeredConvFeatures (derived block). Uses distinct in/out extents so an un-swapped or
// dropped dimension is visible.
TEST_P(GPU_CandidateSelection_FP32, Engineered3dInputSwap_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    const auto spatial_dim = meta.GetInputConstant("spatial_dim");
    if(!spatial_dim.has_value() || *spatial_dim != "3")
        GTEST_SKIP() << params.solver << " is not a 3D solver";

    const float in_c = 32.0f, out_c = 128.0f;
    const float in_d = 10.0f, in_h = 20.0f, in_w = 18.0f;
    const float out_d = 6.0f, out_h = 16.0f, out_w = 14.0f;
    const float k_d = 3.0f, k_h = 3.0f, k_w = 3.0f;
    // direction = 2 (BackwardWeights) matches the 3D Wrw solver; is_fwd == false so the swaps fire.
    std::map<std::string, float> feats = {
        {"direction", 2.0f},     {"batchsize", 1.0f},
        {"group_count", 1.0f},   {"in_channels", in_c},
        {"out_channels", out_c}, {"in_d", in_d},
        {"in_h", in_h},          {"in_w", in_w},
        {"out_d", out_d},        {"out_h", out_h},
        {"out_w", out_w},        {"fil_d", k_d},
        {"fil_h", k_h},          {"fil_w", k_w},
        {"pad_d", 1.0f},         {"pad_h", 1.0f},
        {"pad_w", 1.0f},         {"conv_stride_d", 1.0f},
        {"conv_stride_h", 1.0f}, {"conv_stride_w", 1.0f},
        {"dilation_d", 1.0f},    {"dilation_h", 1.0f},
        {"dilation_w", 1.0f},    {"in_layout", 1.0f},
        {"fil_layout", 1.0f},    {"out_layout", 1.0f},
        {"bias", 0.0f},          {"precision", static_cast<float>(miopenFloat)},
    };

    std::vector<float> engineered;
    ASSERT_NO_THROW(engineered = EngineerCandidateSelectionInputFeatures(feats, meta));

    // (a) Derived tail: Backward swaps in<->out, so the forward-convention dims fed to
    // EngineeredConvFeatures use out_* for the "in" slots (incl. depth) and vice versa. Computed
    // directly here rather than hardcoded, so it pins the swap without duplicating the feature
    // math.
    const auto expected_derived = miopen::ai::common::EngineeredConvFeatures(
        /*N=*/1,
        /*C_in=*/static_cast<std::size_t>(out_c),
        /*C_out=*/static_cast<std::size_t>(in_c),
        /*H_in=*/static_cast<std::size_t>(out_h),
        /*W_in=*/static_cast<std::size_t>(out_w),
        /*H_out=*/static_cast<std::size_t>(in_h),
        /*W_out=*/static_cast<std::size_t>(in_w),
        /*K_h=*/static_cast<std::size_t>(k_h),
        /*K_w=*/static_cast<std::size_t>(k_w),
        /*groups=*/1,
        meta.GetNumCu(),
        miopen::ai::common::ConvDirection::BackwardWeights,
        /*spatial_dim=*/3,
        /*D_in=*/static_cast<std::size_t>(out_d),
        /*D_out=*/static_cast<std::size_t>(in_d),
        /*K_d=*/static_cast<std::size_t>(k_d));
    ASSERT_GE(engineered.size(), expected_derived.size());
    const std::size_t derived_off = engineered.size() - expected_derived.size();
    for(std::size_t i = 0; i < expected_derived.size(); ++i)
        EXPECT_FLOAT_EQ(engineered[derived_off + i], expected_derived[i])
            << "3D Backward derived feature mismatch at index " << i;
    // Trailing 3D feature is log1p(D_in) after the swap: D_in := out_d.
    EXPECT_FLOAT_EQ(engineered.back(), std::log1p(out_d));

    // (b) Raw passthrough swap: reconstruct the one-hot header width and raw ordering from the
    // metadata (mirroring EngineerCandidateSelectionInputFeatures) and assert the in_d slot carries
    // out_d's value and the out_d slot carries in_d's value.
    const auto layout_w = [&](const std::string& f) {
        const auto w = meta.GetInputEncodingClassCount(f);
        return w == 0 ? std::size_t{2} : w;
    };
    const std::size_t header = layout_w("in_layout") + layout_w("fil_layout") +
                               layout_w("out_layout") +
                               meta.GetInputEncodingClassCount("precision");
    const auto is_categorical = [](const std::string& n) {
        return n == "in_layout" || n == "fil_layout" || n == "out_layout" || n == "precision";
    };
    std::size_t raw_idx   = 0;
    std::size_t in_d_idx  = 0;
    std::size_t out_d_idx = 0;
    bool in_d_found       = false;
    bool out_d_found      = false;
    for(const auto& name : meta.input_params())
    {
        if(meta.GetInputConstant(name).has_value() || is_categorical(name) || name == "direction")
            continue;
        if(name == "in_d")
        {
            in_d_idx   = header + raw_idx;
            in_d_found = true;
        }
        else if(name == "out_d")
        {
            out_d_idx   = header + raw_idx;
            out_d_found = true;
        }
        ++raw_idx;
    }
    ASSERT_TRUE(in_d_found);
    ASSERT_TRUE(out_d_found);
    ASSERT_LT(in_d_idx, engineered.size());
    ASSERT_LT(out_d_idx, engineered.size());
    EXPECT_FLOAT_EQ(engineered[in_d_idx], out_d)
        << "in_d slot must carry out_d after Backward swap";
    EXPECT_FLOAT_EQ(engineered[out_d_idx], in_d)
        << "out_d slot must carry in_d after Backward swap";
}

// The conditional-layout descriptor is loaded from metadata: present for the Wavelet kernel,
// absent for a fully static kernel (which then decodes via the plain kernel_str_mapping path).
TEST_P(GPU_CandidateSelection_FP32, ConditionalLayoutDescriptorPresence_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);

    // Static kernel (used as the common test kernel) must have no conditional layout.
    EXPECT_EQ(meta.GetConditionalLayout("DeviceGroupedConvBwdWeight_Xdl_CShuffle"), nullptr);

    if(MetadataHasKernel(meta, kWaveletKernel))
    {
        const auto* layout = meta.GetConditionalLayout(kWaveletKernel);
        ASSERT_NE(layout, nullptr) << "Wavelet metadata present but conditional layout missing";
        EXPECT_EQ(layout->base_param_count, 15u);
        EXPECT_TRUE(layout->conditional_params.count("NumGroupsToMerge") == 1);
        EXPECT_TRUE(layout->conditional_params.count("SplitK") == 1);
        EXPECT_TRUE(layout->packed_params.count("TileLoadMathThreadGroupSize") == 1);
    }
}

// A fully static kernel decodes via the plain path (no conditional descriptor needed).
TEST_P(GPU_CandidateSelection_FP32, StaticKernelDecodes_Test)
{
    const auto& params        = GetParam();
    const std::string& kernel = params.kernel_name; // the per-solver static kernel
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!MetadataHasKernel(meta, kernel))
        GTEST_SKIP() << "Kernel not present in metadata for " << params.solver;

    // The per-solver common kernel is static for most solvers, but not universally: e.g. the Bwd
    // DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1 carries a conditional SplitK suffix. When
    // the chosen kernel has a conditional layout it is not a static kernel for this solver, so skip
    // -- conditional decode is exercised by the Wavelet / V3-large-tensor tests above.
    if(meta.GetConditionalLayout(kernel) != nullptr)
        GTEST_SKIP() << kernel << " has a conditional layout for " << params.solver
                     << " (not a static kernel)";

    // Build a candidate from the static mapping (all numeric "1"s), as GenerateValidKernelParams
    // does.
    auto candidates = GenerateValidKernelParams(meta, kernel, 1);
    auto encoded    = EncodeKernelParams(candidates, meta, /*use_split_k=*/false);
    ASSERT_EQ(encoded.size(), 1u) << "static kernel candidate was unexpectedly skipped";
    EXPECT_EQ(encoded[0].size(),
              meta.output_params().size() - meta.GetConstantOutputIndices().size());
}

// The kernel-config derived block is produced by the metadata-driven registry in
// EngineerCandidateSelectionKernelConfigFeatures. This pins the SplitK encoding this PR added --
// log2_splitk (log2 over the {1,2,...} ladder) plus a separate is_autodeduce flag for the SplitK<1
// (autodeduce / -1) mode -- which replaced feeding SplitK as a raw numeric that systematically
// mis-ranked Wrw SplitK. Runs on the Wrw solvers, whose metadata declares 015_SplitK and the
// Wavelet kernel used here to build a candidate carrying an appended SplitK.
TEST_P(GPU_CandidateSelection_FP32, KernelConfigSplitKDerived_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);

    const auto& derived_names = meta.kernel_config_derived_features();
    const auto name_slot      = [&](const std::string& n) -> int {
        const auto it = std::find(derived_names.begin(), derived_names.end(), n);
        return it == derived_names.end() ? -1 : static_cast<int>(it - derived_names.begin());
    };
    const int log2_idx = name_slot("log2_splitk");
    const int auto_idx = name_slot("is_autodeduce");
    if(log2_idx < 0 || auto_idx < 0)
        GTEST_SKIP() << params.solver
                     << " metadata predates the SplitK derived features (legacy schema)";
    // SplitK must be routed through the derived block, not raw passthrough, else it is
    // double-counted.
    EXPECT_FALSE(meta.kernel_config_raw_exclude().empty())
        << "a SplitK-derived model must exclude its SplitK param from raw passthrough";

    if(!MetadataHasKernel(meta, kWaveletKernel))
        GTEST_SKIP() << "Wavelet kernel (used to build a SplitK candidate) not in " << params.solver
                     << " metadata";

    // Encode a candidate with a known appended SplitK via the production path, then run the derived
    // registry over it. The derived block is the final derived_names.size() entries of the output.
    const auto derived_of = [&](const std::string& split_k) {
        auto candidate = MakeWaveletCandidate(/*include_num_groups=*/true, "2");
        candidate.push_back(split_k); // appended SplitK (mirrors ExpandKernelParamsWithSplitK)
        auto raw = EncodeKernelParams({candidate}, meta, /*use_split_k=*/true);
        EXPECT_EQ(raw.size(), 1u) << "SplitK candidate was unexpectedly skipped";
        auto eng = EngineerCandidateSelectionKernelConfigFeatures(raw.at(0), meta);
        EXPECT_GE(eng.size(), derived_names.size());
        return std::vector<float>(eng.end() - static_cast<long>(derived_names.size()), eng.end());
    };

    // Real split-k ladder value: log2(8) = 3, not autodeduce.
    {
        const auto d = derived_of("8");
        EXPECT_FLOAT_EQ(d.at(log2_idx), std::log2(8.0f)) << "log2_splitk(8) must be 3";
        EXPECT_FLOAT_EQ(d.at(auto_idx), 0.0f) << "is_autodeduce must be 0 for SplitK=8";
    }
    // Autodeduce (-1): log2_splitk collapses to 0, is_autodeduce flags the mode.
    {
        const auto d = derived_of("-1");
        EXPECT_FLOAT_EQ(d.at(log2_idx), 0.0f) << "log2_splitk must be 0 for autodeduce SplitK";
        EXPECT_FLOAT_EQ(d.at(auto_idx), 1.0f) << "is_autodeduce must be 1 for SplitK=-1";
    }
}

// === INSTANTIATION ===

// Helper function to generate test parameters for both 2D and 3D solvers
std::vector<CandidateSelectionParams> GenerateCandidateSelectionParams()
{
    // Note: Using DeviceGroupedConvBwdWeight_Xdl_CShuffle for all as it's a common kernel
    // that exists in the metadata for testing infrastructure (not testing kernel accuracy)
    return {
        // 2D solvers — cover Fwd/Bwd/Wrw so per-solver metadata differences (e.g. the Fwd
        // input encoder gaining an INT8 precision class -> 44 vs 43 features) are exercised.
        {"gfx942",
         "ConvHipImplicitGemmGroupFwdXdlops",
         "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
         8},
        {"gfx942",
         "ConvHipImplicitGemmGroupBwdXdlops",
         "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1",
         8},
        {"gfx942",
         "ConvHipImplicitGemmGroupWrwXdlops",
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle",
         8},
        // 3D solvers (engineered input + kernel-config path)
        {"gfx942",
         "ConvHipImplicitGemm3DGroupWrwXdlops",
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle",
         8},
    };
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_CandidateSelection_FP32,
                         ::testing::ValuesIn(GenerateCandidateSelectionParams()),
                         [](const ::testing::TestParamInfo<CandidateSelectionParams>& testInfo) {
                             std::ostringstream os;
                             PrintTo(testInfo.param, &os);
                             return os.str();
                         });

#else
// Add a dummy test when AI kernel tuning is disabled
TEST(CPU_CandidateSelectionDisabled_NONE, FeatureDisabled)
{
    GTEST_SKIP() << "AI candidate selection features are disabled in this build";
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
