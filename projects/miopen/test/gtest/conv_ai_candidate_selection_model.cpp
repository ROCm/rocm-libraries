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
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <chrono>

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

auto GetCandidateSelectionModelFiles(const CandidateSelectionParams& params)
{
    const auto db_path = miopen::GetSystemDbPath();
    return std::vector<miopen::fs::path>{
        db_path / (params.arch + "_" + params.solver + "_input_encoder.tn.model"),
        db_path / (params.arch + "_" + params.solver + "_kernel_config_encoder.tn.model"),
        db_path / (params.arch + "_" + params.solver + "_metadata.tn.model")};
}

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

// Build a minimal 2D conv ProblemDescription for use as a cache key in tests.
miopen::conv::ProblemDescription
MakeTestProblem(int n = 1, int c = 4, int h = 8, int w = 8, int k = 8, int y = 3, int x = 3)
{
    miopen::TensorDescriptor in_desc(miopenFloat, {n, c, h, w});
    miopen::TensorDescriptor wei_desc(miopenFloat, {k, c, y, x});
    miopen::TensorDescriptor out_desc(miopenFloat, {n, k, h - y + 1, w - x + 1});
    miopen::ConvolutionDescriptor conv_desc(
        2, miopenConvolution, miopenPaddingDefault, {0, 0}, {1, 1}, {1, 1});
    return miopen::conv::ProblemDescription(
        in_desc, wei_desc, out_desc, conv_desc, miopen::conv::Direction::BackwardWeights);
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
        const auto missing      = GetMissingModelFiles(params);

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

TEST_P(GPU_CandidateSelection_FP32, EncodeKernelConfigs_Test)
{
    const auto& params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    size_t feature_size = meta.output_params().size() - meta.GetConstantOutputIndices().size();
    std::vector<std::vector<float>> encoded_candidates(100, std::vector<float>(feature_size, 2.0f));
    auto encoded = model.EncodeKernelConfigs(encoded_candidates);
    ASSERT_FALSE(encoded.empty()) << "EncodeKernelConfigs returned empty vector!";
    for(const auto& vec : encoded)
        ASSERT_FALSE(vec.empty()) << "EncodeKernelConfigs returned a candidate with empty vector!";
}

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
    EXPECT_NO_THROW(result = EncodeKernelParams(bad_params, meta));

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
    auto encoded_features    = model.EncodeInputFeatures(features);
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto encoded_candidates  = EncodeKernelParams(valid_kernel_params, meta);
    auto encoded_configs     = model.EncodeKernelConfigs(encoded_candidates);
    std::vector<std::pair<int, float>> ids =
        model.SelectBestCandidateIndices(encoded_features, encoded_configs);
    ASSERT_FALSE(ids.empty()) << "No candidates were selected!";
    for(const auto& candidate : ids)
    {
        const int idx = candidate.first;
        ASSERT_GE(idx, 0) << "Candidate index is negative!";
        ASSERT_LT(idx, static_cast<int>(valid_kernel_params.size()))
            << "Candidate index " << idx << " out of range [0, " << valid_kernel_params.size() - 1
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
    auto problem             = MakeTestProblem();
    auto result              = ModelSelectBestCandidate(params.arch,
                                           params.solver,
                                           problem,
                                           features,
                                           valid_kernel_params,
                                           /*use_split_k=*/false,
                                           accept_all_combinations);
    for(const auto& idx : result.kernel_indices)
    {
        ASSERT_GE(idx, 0) << "Candidate index is negative!";
        ASSERT_LT(idx, static_cast<int>(valid_kernel_params.size()))
            << "Candidate index " << idx << " out of range [0, " << valid_kernel_params.size() - 1
            << "]";
    }
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

// === CACHE TESTS ===

// Calling ModelSelectBestCandidate twice with the same arguments must return
// identical results. The second call hits the in-process RAM cache.
TEST_P(GPU_CandidateSelection_FP32, CandidateSelectionRamCache_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto problem             = MakeTestProblem();

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto result1  = ModelSelectBestCandidate(params.arch,
                                            params.solver,
                                            problem,
                                            features,
                                            valid_kernel_params,
                                            /*use_split_k=*/false,
                                            accept_all_combinations);
    const auto t1 = std::chrono::high_resolution_clock::now();
    auto result2  = ModelSelectBestCandidate(params.arch,
                                            params.solver,
                                            problem,
                                            features,
                                            valid_kernel_params,
                                            /*use_split_k=*/false,
                                            accept_all_combinations);
    const auto t2 = std::chrono::high_resolution_clock::now();

    ASSERT_FALSE(result1.IsEmpty()) << "First call returned empty result";
    ASSERT_FALSE(result2.IsEmpty()) << "Second call (cache hit) returned empty result";
    ASSERT_EQ(result1.kernel_indices, result2.kernel_indices) << "Cache returned different indices";
    ASSERT_EQ(result1.split_k_values, result2.split_k_values)
        << "Cache returned different split_k values";

    const auto first_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto second_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "[CandidateSelectionRamCache] first=" << first_ms << " ms, second=" << second_ms
              << " ms\n";
}

// Kernel-config embeddings are cached per row via GetOrComputeKernelEmbeddings,
// which is exercised through ModelSelectBestCandidate. Two calls with identical
// features and kernel params but different problems bypass the B-RAM cache yet
// share the per-row embedding cache. Because both inputs to the scorer are then
// identical (same Tower 1 output, same cached Tower 2 embeddings), the rankings
// must be deterministically the same.
TEST_P(GPU_CandidateSelection_FP32, KernelEmbeddingCache_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);

    // Call 1: populates the B-RAM cache (keyed on problem_a) and the per-row
    // embedding cache for all kernel rows.
    auto problem_a = MakeTestProblem(1, 4, 8, 8, 8, 3, 3);
    auto result_a  = ModelSelectBestCandidate(params.arch,
                                             params.solver,
                                             problem_a,
                                             features,
                                             valid_kernel_params,
                                             /*use_split_k=*/false,
                                             accept_all_combinations);

    // Call 2: different N → different B-RAM key → B-RAM cache miss.
    // Same features and same kernel params → embedding cache hit for all rows.
    // Identical scorer inputs must produce identical rankings.
    auto problem_b = MakeTestProblem(2, 4, 8, 8, 8, 3, 3);
    auto result_b  = ModelSelectBestCandidate(params.arch,
                                             params.solver,
                                             problem_b,
                                             features,
                                             valid_kernel_params,
                                             /*use_split_k=*/false,
                                             accept_all_combinations);

    ASSERT_FALSE(result_a.IsEmpty()) << "First call returned empty result";
    ASSERT_FALSE(result_b.IsEmpty())
        << "Second call returned empty result (embedding cache may be corrupt)";
    ASSERT_EQ(result_a.kernel_indices, result_b.kernel_indices)
        << "Rankings differ despite identical features and kernel set";
    ASSERT_EQ(result_a.split_k_values, result_b.split_k_values);
}

// use_split_k=false and use_split_k=true must produce independent cache entries.
TEST_P(GPU_CandidateSelection_FP32, CandidateSelectionRamCacheIsolation_Test)
{
    const auto& params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto problem             = MakeTestProblem();

    auto result_no_splitk = ModelSelectBestCandidate(params.arch,
                                                     params.solver,
                                                     problem,
                                                     features,
                                                     valid_kernel_params,
                                                     /*use_split_k=*/false,
                                                     accept_all_combinations);
    auto result_splitk    = ModelSelectBestCandidate(params.arch,
                                                  params.solver,
                                                  problem,
                                                  features,
                                                  valid_kernel_params,
                                                  /*use_split_k=*/true,
                                                  accept_all_combinations);

    ASSERT_FALSE(result_no_splitk.IsEmpty()) << "use_split_k=false returned empty";
    ASSERT_FALSE(result_splitk.IsEmpty()) << "use_split_k=true returned empty";

    auto result_no_splitk_cached = ModelSelectBestCandidate(params.arch,
                                                            params.solver,
                                                            problem,
                                                            features,
                                                            valid_kernel_params,
                                                            /*use_split_k=*/false,
                                                            accept_all_combinations);
    auto result_splitk_cached    = ModelSelectBestCandidate(params.arch,
                                                         params.solver,
                                                         problem,
                                                         features,
                                                         valid_kernel_params,
                                                         /*use_split_k=*/true,
                                                         accept_all_combinations);

    ASSERT_EQ(result_no_splitk.kernel_indices, result_no_splitk_cached.kernel_indices)
        << "Cache isolation failed: use_split_k=false entry was overwritten";
    ASSERT_EQ(result_splitk.kernel_indices, result_splitk_cached.kernel_indices)
        << "Cache isolation failed: use_split_k=true entry was overwritten";

    EXPECT_NE(result_no_splitk.split_k_values, result_splitk.split_k_values)
        << "Expected use_split_k=false and use_split_k=true to produce different split_k_values";
}

// === INSTANTIATION ===

// Helper function to generate test parameters for both 2D and 3D solvers
std::vector<CandidateSelectionParams> GenerateCandidateSelectionParams()
{
    // Note: Using DeviceGroupedConvBwdWeight_Xdl_CShuffle for all as it's a common kernel
    // that exists in the metadata for testing infrastructure (not testing kernel accuracy)
    return {
        // 2D solvers
        {"gfx942",
         "ConvHipImplicitGemmGroupWrwXdlops",
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle",
         8},
        // 3D solvers
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
