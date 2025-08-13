/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#include <miopen/filesystem.hpp>

using namespace miopen::ai::tuning::candidate_selection;
using miopen::ai::tuning::candidate_selection::ExpandKernelParamsWithSplitK;
using miopen::solver::conv::GenerateSplitK;

class CandidateSelectionTest : public ::testing::Test
{
protected:
    std::string arch        = "gfx942";
    std::string solver      = "ConvHipImplicitGemm3DGroupWrwXdlops";
    std::string kernel_name = "DeviceGroupedConvBwdWeight_Xdl_CShuffle";
};

// Helper function to generate valid_kernel_params for a given kernel_name and metadata
std::vector<std::vector<std::string>> GenerateValidKernelParams(
    const CandidateSelectionMetadata& meta, const std::string& kernel_name, int num_candidates = 3)
{
    const auto& kernel_str_mapping = meta.GetKernelStrMapping(kernel_name);
    std::vector<std::vector<std::string>> valid_kernel_params;

    for(int i = 0; i < num_candidates; ++i)
    {
        std::vector<std::string> candidate(meta.output_params().size(), "nan");
        candidate[0] = kernel_name; // first element is kernel_name

        for(const auto& kv : kernel_str_mapping)
        {
            const std::string& param_name = kv.second;
            const std::string& index      = kv.first;
            const int index_int           = std::stoi(index);
            if(param_name.find("kernel_name") != std::string::npos)
            {
                continue; // Skip kernel_name
            }
            auto it = meta.sequence_encodings().find(param_name);
            if(it == meta.sequence_encodings().end())
            {
                candidate[index_int] = "0";
            }
            else
            {
                const auto& encodings_map = it->second;
                candidate[index_int]      = encodings_map.begin()->first;
            }
        }
        valid_kernel_params.push_back(candidate);
    }
    return valid_kernel_params;
}

struct FilesExistParams
{
    std::string arch;
    std::string solver;
};
class CPU_FilesExist : public ::testing::TestWithParam<FilesExistParams>
{
};
TEST_P(CPU_FilesExist, FilesExist)
{
    auto params        = GetParam();
    auto db_path       = miopen::GetSystemDbPath();
    auto input_encoder = db_path / (params.arch + "_" + params.solver + "_input_encoder.tn.model");
    auto kernel_config_encoder =
        db_path / (params.arch + "_" + params.solver + "_kernel_config_encoder.tn.model");
    auto metadata = db_path / (params.arch + "_" + params.solver + "_metadata.tn.model");

    ASSERT_TRUE(miopen::fs::exists(input_encoder)) << "Input encoder file missing!";
    ASSERT_TRUE(miopen::fs::exists(kernel_config_encoder)) << "Kernel config encoder file missing!";
    ASSERT_TRUE(miopen::fs::exists(metadata)) << "Metadata file missing!";
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_FilesExist,
                         ::testing::Values(FilesExistParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct MetadataAndModelInitParams
{
    std::string arch;
    std::string solver;
};
class CPU_MetadataAndModelInit : public ::testing::TestWithParam<MetadataAndModelInitParams>
{
};
TEST_P(CPU_MetadataAndModelInit, MetadataAndModelInit)
{
    auto params = GetParam();
    ASSERT_NO_THROW({
        CandidateSelectionMetadata meta(params.arch, params.solver);
        CandidateSelectionModel model(params.arch, params.solver);
    });
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_MetadataAndModelInit,
                         ::testing::Values(MetadataAndModelInitParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct ModelCachingParams
{
    std::string arch;
    std::string solver;
};
class CPU_ModelCaching : public ::testing::TestWithParam<ModelCachingParams>
{
};
TEST_P(CPU_ModelCaching, ModelCaching)
{
    auto params  = GetParam();
    auto& model1 = GetCandidateSelectionModel(params.arch, params.solver);
    auto& model2 = GetCandidateSelectionModel(params.arch, params.solver);
    ASSERT_EQ(&model1, &model2)
        << "GetCandidateSelectionModel did not return the same cached object!";
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_ModelCaching,
                         ::testing::Values(ModelCachingParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));
struct EncodeInputFeaturesParams
{
    std::string arch;
    std::string solver;
};
class CPU_EncodeInputFeatures : public ::testing::TestWithParam<EncodeInputFeaturesParams>
{
};
TEST_P(CPU_EncodeInputFeatures, EncodeInputFeatures)
{
    auto params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    auto encoded = model.EncodeInputFeatures(features);
    ASSERT_FALSE(encoded.empty()) << "EncodeInputFeatures returned empty vector!";
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_EncodeInputFeatures,
                         ::testing::Values(EncodeInputFeaturesParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct EncodeKernelConfigsParams
{
    std::string arch;
    std::string solver;
};
class CPU_EncodeKernelConfigs : public ::testing::TestWithParam<EncodeKernelConfigsParams>
{
};
TEST_P(CPU_EncodeKernelConfigs, EncodeKernelConfigs)
{
    auto params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    size_t feature_size = meta.output_params().size() - meta.GetConstantOutputIndices().size();
    std::vector<std::vector<float>> encoded_candidates(100, std::vector<float>(feature_size, 2.0f));
    auto encoded = model.EncodeKernelConfigs(encoded_candidates);
    ASSERT_FALSE(encoded.empty()) << "EncodeKernelConfigs returned empty vector!";
    for(const auto& vec : encoded)
    {
        ASSERT_FALSE(vec.empty()) << "EncodeKernelConfigs returned a candidate with empty vector!";
    }
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_EncodeKernelConfigs,
                         ::testing::Values(EncodeKernelConfigsParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct EncodeInputFeaturesEdgeCasesParams
{
    std::string arch;
    std::string solver;
};
class CPU_EncodeInputFeaturesEdgeCases
    : public ::testing::TestWithParam<EncodeInputFeaturesEdgeCasesParams>
{
};
TEST_P(CPU_EncodeInputFeaturesEdgeCases, EncodeInputFeaturesEdgeCases)
{
    auto params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);

    std::map<std::string, float> empty_features;
    EXPECT_THROW(model.EncodeInputFeatures(empty_features), std::exception);

    std::map<std::string, float> long_features;
    for(const auto& name : meta.input_params())
    {
        long_features[name] = 1.0f;
    }
    long_features["extra_param"] = 2.0f;
    EXPECT_NO_THROW({
        auto encoded = model.EncodeInputFeatures(long_features);
        ASSERT_FALSE(encoded.empty());
    });

    if(!meta.GetConstantInputIndices().empty())
    {
        std::map<std::string, float> features;
        for(const auto& name : meta.input_params())
        {
            features[name] = 1.0f;
        }
        for(auto idx : meta.GetConstantInputIndices())
        {
            if(idx < meta.input_params().size())
                features[meta.input_params()[idx]] = 42.0f;
        }
        EXPECT_NO_THROW({
            auto encoded = model.EncodeInputFeatures(features);
            ASSERT_FALSE(encoded.empty());
        });
    }
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_EncodeInputFeaturesEdgeCases,
                         ::testing::Values(EncodeInputFeaturesEdgeCasesParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct EncodeKernelConfigsEdgeCasesParams
{
    std::string arch;
    std::string solver;
};
class CPU_EncodeKernelConfigsEdgeCases
    : public ::testing::TestWithParam<EncodeKernelConfigsEdgeCasesParams>
{
};
TEST_P(CPU_EncodeKernelConfigsEdgeCases, EncodeKernelConfigsEdgeCases)
{
    auto params = GetParam();
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
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_EncodeKernelConfigsEdgeCases,
                         ::testing::Values(EncodeKernelConfigsEdgeCasesParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct KernelStrMappingUnknownKernelThrowsParams
{
    std::string arch;
    std::string solver;
};
class CPU_KernelStrMappingUnknownKernelThrows
    : public ::testing::TestWithParam<KernelStrMappingUnknownKernelThrowsParams>
{
};
TEST_P(CPU_KernelStrMappingUnknownKernelThrows, KernelStrMappingUnknownKernelThrows)
{
    auto params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    EXPECT_THROW(meta.GetKernelStrMapping("unknown_kernel_name"), std::exception);
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_KernelStrMappingUnknownKernelThrows,
                         ::testing::Values(KernelStrMappingUnknownKernelThrowsParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct OutputConstantRetrievalParams
{
    std::string arch;
    std::string solver;
};
class CPU_OutputConstantRetrieval : public ::testing::TestWithParam<OutputConstantRetrievalParams>
{
};
TEST_P(CPU_OutputConstantRetrieval, OutputConstantRetrieval)
{
    auto params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    if(!meta.output_params().empty())
    {
        auto known = meta.GetOutputConstant(meta.output_params()[0]);
        SUCCEED();
    }
    auto unknown = meta.GetOutputConstant("nonexistent_param");
    EXPECT_EQ(unknown, std::nullopt);
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_OutputConstantRetrieval,
                         ::testing::Values(OutputConstantRetrievalParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct InputOutputParamIndexThrowsParams
{
    std::string arch;
    std::string solver;
};
class CPU_InputOutputParamIndexThrows
    : public ::testing::TestWithParam<InputOutputParamIndexThrowsParams>
{
};
TEST_P(CPU_InputOutputParamIndexThrows, InputOutputParamIndexThrows)
{
    auto params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    EXPECT_THROW(meta.GetInputParamIndex("nonexistent_param"), std::exception);
    EXPECT_THROW(meta.GetOutputParamIndex("nonexistent_param"), std::exception);
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_InputOutputParamIndexThrows,
                         ::testing::Values(InputOutputParamIndexThrowsParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));
struct EncodeKernelParamsBadValueThrowsParams
{
    std::string arch;
    std::string solver;
    std::string kernel_name;
};
class CPU_EncodeKernelParamsBadValueThrows
    : public ::testing::TestWithParam<EncodeKernelParamsBadValueThrowsParams>
{
};
TEST_P(CPU_EncodeKernelParamsBadValueThrows, EncodeKernelParamsBadValueThrows)
{
    auto params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::vector<std::vector<std::string>> bad_params = {
        {params.kernel_name, "nonexistent_value", "nan"}};
    EXPECT_THROW(EncodeKernelParams(bad_params, meta), std::exception);
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_EncodeKernelParamsBadValueThrows,
                         ::testing::Values(EncodeKernelParamsBadValueThrowsParams{
                             "gfx942",
                             "ConvHipImplicitGemm3DGroupWrwXdlops",
                             "DeviceGroupedConvBwdWeight_Xdl_CShuffle"}));

struct SelectBestCandidateValidParams
{
    std::string arch;
    std::string solver;
    std::string kernel_name;
};
class CPU_SelectBestCandidateValid : public ::testing::TestWithParam<SelectBestCandidateValidParams>
{
};
TEST_P(CPU_SelectBestCandidateValid, SelectBestCandidateValid)
{
    auto params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    auto encoded_features    = model.EncodeInputFeatures(features);
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto encoded_candidates  = EncodeKernelParams(valid_kernel_params, meta);
    auto encoded_configs     = model.EncodeKernelConfigs(encoded_candidates);
    int idx                  = model.SelectBestCandidateIdx(encoded_features, encoded_configs);
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(valid_kernel_params.size()));
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_SelectBestCandidateValid,
                         ::testing::Values(SelectBestCandidateValidParams{
                             "gfx942",
                             "ConvHipImplicitGemm3DGroupWrwXdlops",
                             "DeviceGroupedConvBwdWeight_Xdl_CShuffle"}));

struct SelectBestCandidateEmptyInputParams
{
    std::string arch;
    std::string solver;
};
class CPU_SelectBestCandidateEmptyInput
    : public ::testing::TestWithParam<SelectBestCandidateEmptyInputParams>
{
};
TEST_P(CPU_SelectBestCandidateEmptyInput, SelectBestCandidateEmptyInput)
{
    auto params = GetParam();
    CandidateSelectionModel model(params.arch, params.solver);
    std::vector<float> encoded_features;
    std::vector<std::vector<float>> encoded_configs;
    EXPECT_THROW(model.SelectBestCandidateIdx(encoded_features, encoded_configs), std::exception);
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_SelectBestCandidateEmptyInput,
                         ::testing::Values(SelectBestCandidateEmptyInputParams{
                             "gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"}));

struct ModelSelectBestCandidateParams
{
    std::string arch;
    std::string solver;
    std::string kernel_name;
};
class CPU_ModelSelectBestCandidate : public ::testing::TestWithParam<ModelSelectBestCandidateParams>
{
};
TEST_P(CPU_ModelSelectBestCandidate, ModelSelectBestCandidate)
{
    auto params = GetParam();
    CandidateSelectionMetadata meta(params.arch, params.solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    auto valid_kernel_params = GenerateValidKernelParams(meta, params.kernel_name, 3);
    auto result              = ModelSelectBestCandidate(
        params.arch, params.solver, features, valid_kernel_params, /*use_split_k=*/false);
    ASSERT_GE(result.kernel_index, 0);
    ASSERT_LT(result.kernel_index, static_cast<int>(valid_kernel_params.size()));
}
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_ModelSelectBestCandidate,
                         ::testing::Values(ModelSelectBestCandidateParams{
                             "gfx942",
                             "ConvHipImplicitGemm3DGroupWrwXdlops",
                             "DeviceGroupedConvBwdWeight_Xdl_CShuffle"}));

struct ExpandKernelParamsWithSplitKParams
{
    int split_k;
};
class CPU_ExpandKernelParamsWithSplitK
    : public ::testing::TestWithParam<ExpandKernelParamsWithSplitKParams>
{
};
TEST_P(CPU_ExpandKernelParamsWithSplitK, ExpandKernelParamsWithSplitK)
{
    auto params                                   = GetParam();
    std::vector<std::vector<std::string>> kernels = {{"typeA", "p1"}, {"typeB", "p2"}};
    std::vector<int> indexes                      = {0, 1};
    std::vector<int> split_ks                     = GenerateSplitK(params.split_k);
    auto [expanded, mapping] = ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

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
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_ExpandKernelParamsWithSplitK,
                         ::testing::Values(ExpandKernelParamsWithSplitKParams{8}));

struct ExpandKernelParamsWithSplitKFunctionalityParams
{
    int split_k;
};
class CPU_ExpandKernelParamsWithSplitKFunctionality
    : public ::testing::TestWithParam<ExpandKernelParamsWithSplitKFunctionalityParams>
{
};
TEST_P(CPU_ExpandKernelParamsWithSplitKFunctionality, ExpandKernelParamsWithSplitKFunctionality)
{
    auto params                                   = GetParam();
    std::vector<std::vector<std::string>> kernels = {
        {"DeviceGroupedConvBwdWeight_Xdl_CShuffle", "p1"}};
    std::vector<int> indexes  = {0};
    std::vector<int> split_ks = GenerateSplitK(params.split_k);
    auto [expanded, mapping]  = ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

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
INSTANTIATE_TEST_SUITE_P(CPU_,
                         CPU_ExpandKernelParamsWithSplitKFunctionality,
                         ::testing::Values(ExpandKernelParamsWithSplitKFunctionalityParams{8}));
