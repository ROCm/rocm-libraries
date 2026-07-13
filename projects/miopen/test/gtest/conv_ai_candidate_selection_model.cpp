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
#include <miopen/conv/heuristics/ai_heuristics.hpp> // common::EngineeredConvFeatures
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

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
    // Fill with class index 0, which is in range for every categorical output encoding (and is not
    // the -1 missing-value token). A categorical value >= its class count is rejected by
    // AppendKernelConfigOneHot, so an arbitrary fill like 2.0 would (correctly) throw.
    std::vector<std::vector<float>> encoded_candidates(100, std::vector<float>(feature_size, 0.0f));
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
    auto encoded_features    = model.EncodeInputFeatures(features);
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto encoded_candidates  = EncodeKernelParams(valid_kernel_params, meta, /*use_split_k=*/false);
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
    auto result              = ModelSelectBestCandidate(params.arch,
                                           params.solver,
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

    // TileMathThreadGroupSize is a live numeric feature (TileLoad is constant in metadata).
    const int math_slot = EncodedSlotForSuffix(meta, "TileMathThreadGroupSize");
    ASSERT_GE(math_slot, 0) << "TileMathThreadGroupSize not found as a live encoded feature";
    EXPECT_FLOAT_EQ(encoded[0][math_slot], 256.0f) << "math part of '128l+256m' not decoded to 256";
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

    ASSERT_EQ(meta.GetConditionalLayout(kernel), nullptr)
        << "test kernel is expected to be static (no conditional layout)";

    // Build a candidate from the static mapping (all numeric "1"s), as GenerateValidKernelParams
    // does.
    auto candidates = GenerateValidKernelParams(meta, kernel, 1);
    auto encoded    = EncodeKernelParams(candidates, meta, /*use_split_k=*/false);
    ASSERT_EQ(encoded.size(), 1u) << "static kernel candidate was unexpectedly skipped";
    EXPECT_EQ(encoded[0].size(),
              meta.output_params().size() - meta.GetConstantOutputIndices().size());
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
        // 3D solvers (raw, non-engineered feature path)
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
