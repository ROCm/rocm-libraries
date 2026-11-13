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
#include <miopen/solver/ck_impl_lib_loader.hpp> // CkImplLibLoader, CKSolverType (CK enumeration)
#include <miopen/env_debug.hpp>                 // debug::env::UpdateEnvVariable (table toggle)
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/md5.hpp> // miopen::md5 (loader md5-mismatch test)
#include <random>
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
#include <array>
#include <cstdint>
#include <fstream>

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
// model constructor does not load it, candidate selection reads the precomputed config vocabulary,
// and the input-feature tests use the input encoder. The fixture already skips/fails on missing
// input encoder + metadata.

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

// (The config side is now the precomputed vocabulary ranked by SelectBestVocabCandidate -- there is
// no per-candidate config encode to assert here; the end-to-end selection is exercised by the
// driver and the SelectBestVocab* tests below.)

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
    const auto& vocab      = GetConfigEmbeddingVocab(params.arch, params.solver);
    const auto& emb_matrix = GetConfigEmbeddingMatrix(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty()) << "shipped config-embedding vocab is empty";
    std::vector<std::vector<float>> encoded_configs;
    for(std::size_t i = 0; i < 3 && i < vocab.size(); ++i)
        encoded_configs.push_back(emb_matrix[i]);
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

namespace {
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

// ============================================================================
// lazy-applicability walk (SelectBestVocabCandidate)
// ============================================================================
namespace {
// Bare GetTypeString from a vocab id: strip a trailing "+<split_k>" after the final '>'.
std::string VocabBase(const std::string& kid)
{
    const auto gt   = kid.rfind('>');
    const auto plus = kid.find('+', gt == std::string::npos ? 0 : gt);
    return plus == std::string::npos ? kid : kid.substr(0, plus);
}
// All-ones input features for a solver (same convention as SelectBestCandidateValid_Test).
std::map<std::string, float> AllOnesFeatures(const std::string& arch, const std::string& solver)
{
    CandidateSelectionMetadata meta(arch, solver);
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;
    return features;
}
} // namespace

// Best-first: when every candidate is applicable, the walk returns the highest-ranked vocab base.
// The ranking is recomputed exactly as SelectBestCandidateIndices does (inner product, NaN last).
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabBestFirst_Test)
{
    const auto& params = GetParam();
    const auto& vocab  = GetConfigEmbeddingVocab(params.arch, params.solver);
    const auto& emb    = GetConfigEmbeddingMatrix(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty());
    const auto features = AllOnesFeatures(params.arch, params.solver);
    CandidateSelectionModel model(params.arch, params.solver);
    const auto encoded_features = model.EncodeInputFeatures(features);
    int best                    = -1;
    float best_score            = 0.0f;
    for(std::size_t i = 0; i < vocab.size(); ++i)
    {
        if(emb[i].size() != encoded_features.size())
            continue;
        const float s =
            std::inner_product(emb[i].begin(), emb[i].end(), encoded_features.begin(), 0.0f);
        if(std::isnan(s))
            continue;
        if(best < 0 || s > best_score)
        {
            best       = static_cast<int>(i);
            best_score = s;
        }
    }
    ASSERT_GE(best, 0);
    auto accept_all = [](const std::string&, int, bool) { return true; };
    auto result     = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, accept_all);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->first, VocabBase(vocab[best]))
        << "accept-all walk did not return the top-ranked candidate";
}

// First applicable wins: when only one vocab base is applicable, the walk returns exactly it,
// regardless of its rank (i.e. it keeps walking past rejected higher-ranked candidates).
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabFirstApplicableWins_Test)
{
    const auto& params = GetParam();
    const auto& vocab  = GetConfigEmbeddingVocab(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty());
    const std::string target = VocabBase(vocab[vocab.size() / 2]);
    const auto features      = AllOnesFeatures(params.arch, params.solver);
    auto only_target         = [&](const std::string& base, int, bool) { return base == target; };
    auto result              = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, only_target);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->first, target);
}

// TF32 two-pass: a predicate that only accepts on the FP32 fallback pass is reached iff use_tf32,
// mirroring the enumeration order (TF32 set first, FP32 set only if nothing applied).
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabTwoPassTf32_Test)
{
    const auto& params = GetParam();
    const auto& vocab  = GetConfigEmbeddingVocab(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty());
    const std::string target = VocabBase(vocab.front());
    const auto features      = AllOnesFeatures(params.arch, params.solver);
    auto fp32_only           = [&](const std::string& base, int, bool fp32_pass) {
        return fp32_pass && base == target;
    };
    // use_tf32 -> the FP32 fallback pass runs and finds the target.
    auto tf32_on = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/true, fp32_only);
    ASSERT_TRUE(tf32_on.has_value());
    EXPECT_EQ(tf32_on->first, target);
    // no TF32 -> only the first pass runs, the target is never offered, nothing applies.
    auto tf32_off = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, fp32_only);
    EXPECT_FALSE(tf32_off.has_value());
}

// Nothing applicable in either pass -> nullopt (caller falls to the non-AI default).
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabNoneApplicable_Test)
{
    const auto& params  = GetParam();
    const auto features = AllOnesFeatures(params.arch, params.solver);
    auto reject_all     = [](const std::string&, int, bool) { return false; };
    EXPECT_FALSE(SelectBestVocabCandidate(
                     params.arch, params.solver, features, /*use_tf32=*/true, reject_all)
                     .has_value());
}

namespace {
// RAII: restore an env var to "0" on scope exit, so a failed EXPECT mid-test cannot leak the
// setting into later tests sharing the process.
struct ScopedEnvVar
{
    const char* key;
    ScopedEnvVar(const char* k, const char* v) : key(k)
    {
        miopen::debug::env::UpdateEnvVariable(key, v);
    }
    ~ScopedEnvVar() { miopen::debug::env::UpdateEnvVariable(key, "0"); }
};
} // namespace

// Kill switch: MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB empties the vocab -> nullopt even
// with an accept-all predicate.
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabKillSwitch_Test)
{
    const auto& params  = GetParam();
    const auto features = AllOnesFeatures(params.arch, params.solver);
    auto accept_all     = [](const std::string&, int, bool) { return true; };
    ScopedEnvVar guard("MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB", "1");
    auto result = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, accept_all);
    EXPECT_FALSE(result.has_value());
}

// LGBM-only: MIOPEN_DEBUG_LGBM_ONLY turns the two-tower (lazy) path off -> nullopt.
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabLgbmOnly_Test)
{
    const auto& params  = GetParam();
    const auto features = AllOnesFeatures(params.arch, params.solver);
    auto accept_all     = [](const std::string&, int, bool) { return true; };
    ScopedEnvVar guard("MIOPEN_DEBUG_LGBM_ONLY", "1");
    auto result = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, accept_all);
    EXPECT_FALSE(result.has_value());
}

// Deterministic split-k gate: a predicate that rejects any split_k != 1 must yield either no pick
// (the vocab has no split_k=1 entry) or a pick with split_k == 1 -- never a split_k>1 kernel.
TEST_P(GPU_CandidateSelection_FP32, SelectBestVocabRejectsNonUnitSplitK_Test)
{
    const auto& params     = GetParam();
    const auto features    = AllOnesFeatures(params.arch, params.solver);
    auto unit_split_k_only = [](const std::string&, int split_k, bool) { return split_k == 1; };
    auto result            = SelectBestVocabCandidate(
        params.arch, params.solver, features, /*use_tf32=*/false, unit_split_k_only);
    if(result.has_value())
        EXPECT_EQ(result->second, 1);
}

namespace {
// Map a candidate-selection solver name to its CK loader solver type.
std::optional<miopen::solver::CKSolverType> CKSolverTypeForSolver(const std::string& s)
{
    using miopen::solver::CKSolverType;
    if(s == "ConvHipImplicitGemmGroupFwdXdlops")
        return CKSolverType::GrpConvFwd;
    if(s == "ConvHipImplicitGemmGroupBwdXdlops")
        return CKSolverType::GrpConvBwd;
    if(s == "ConvHipImplicitGemmGroupWrwXdlops")
        return CKSolverType::GrpConvWrw;
    if(s == "ConvHipImplicitGemm3DGroupFwdXdlops")
        return CKSolverType::GrpConv3dFwd;
    if(s == "ConvHipImplicitGemm3DGroupBwdXdlops")
        return CKSolverType::GrpConv3dBwd;
    if(s == "ConvHipImplicitGemm3DGroupWrwXdlops")
        return CKSolverType::GrpConv3dWrw;
    return std::nullopt;
}

// Strip the "+<split_k>" suffix a split-k solver appends after the final '>' (mirrors the loader).
std::string VocabBaseId(const std::string& id)
{
    const auto gt   = id.rfind('>');
    const auto plus = id.find('+', gt == std::string::npos ? 0 : gt);
    return plus == std::string::npos ? id : id.substr(0, plus);
}
} // namespace

// The shipped config-embedding vocab must cover the build's live CK instance set: the lazy walk
// only ranks kernels that are in the vocab, so a regeneration that truncated or mismatched it would
// silently make those kernels unselectable. ASSERT the vocab is present, and EXPECT most live CK
// instances are covered (lenient, to tolerate modest CK version drift between the shipped table and
// the build's CK plugin). Runs for the current device arch only.
TEST_P(GPU_CandidateSelection_FP32, ShippedVocabCoversCKInstances_Test)
{
    const auto& params      = GetParam();
    const auto current_arch = handle.GetDeviceName();
    if(params.arch != current_arch)
        GTEST_SKIP() << "CK enumeration uses the current device's plugin; skipping foreign arch "
                     << params.arch;

    const auto ck_solver = CKSolverTypeForSolver(params.solver);
    ASSERT_TRUE(ck_solver.has_value()) << "No CKSolverType mapping for solver " << params.solver;

    const auto ck_instances =
        miopen::solver::CkImplLibLoader::Get(current_arch).GetAllKernelTypeStrings(*ck_solver);
    ASSERT_FALSE(ck_instances.empty()) << "No CK instances enumerated for " << params.solver;

    const auto& vocab = GetConfigEmbeddingVocab(params.arch, params.solver);
    ASSERT_FALSE(vocab.empty()) << "Shipped config-embedding vocab is empty for " << params.arch
                                << "_" << params.solver
                                << " -- the table is missing or was truncated to nothing.";

    std::set<std::string> vocab_bases;
    for(const auto& id : vocab)
        vocab_bases.insert(VocabBaseId(id));

    std::size_t covered = 0;
    std::string first_missing;
    for(const auto& k : ck_instances)
    {
        if(vocab_bases.count(k) != 0)
            ++covered;
        else if(first_missing.empty())
            first_missing = k;
    }
    const double fraction = static_cast<double>(covered) / static_cast<double>(ck_instances.size());
    EXPECT_GE(fraction, 0.5)
        << "Shipped vocab covers only " << covered << " / " << ck_instances.size()
        << " live CK instances for " << params.arch << "_" << params.solver
        << " -- likely a stale or truncated table; regenerate it. First uncovered: "
        << first_missing;
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_CandidateSelection_FP32,
                         ::testing::ValuesIn(GenerateCandidateSelectionParams()),
                         [](const ::testing::TestParamInfo<CandidateSelectionParams>& testInfo) {
                             std::ostringstream os;
                             PrintTo(testInfo.param, &os);
                             return os.str();
                         });

// ============================================================================
// config-embedding table loader
// ============================================================================
// Every shipped table loads into a non-empty vocabulary: well-formed MICE, a v3 vocab block, and an
// md5 that matches the installed input encoder (a mismatch rejects the table -> empty vocab).
TEST(CPU_ConfigEmbeddingTableLoader_NONE, AllShippedTablesWellFormed)
{
    const std::array<const char*, 2> arches  = {"gfx942", "gfx950"};
    const std::array<const char*, 6> solvers = {"ConvHipImplicitGemmGroupFwdXdlops",
                                                "ConvHipImplicitGemmGroupBwdXdlops",
                                                "ConvHipImplicitGemmGroupWrwXdlops",
                                                "ConvHipImplicitGemm3DGroupFwdXdlops",
                                                "ConvHipImplicitGemm3DGroupBwdXdlops",
                                                "ConvHipImplicitGemm3DGroupWrwXdlops"};
    const auto db                            = miopen::GetSystemDbPath();
    int present                              = 0;
    for(const auto* arch : arches)
        for(const auto* solver : solvers)
        {
            const auto path =
                db / (std::string(arch) + "_" + solver + "_kernel_config_embeddings.bin");
            if(!miopen::fs::exists(path))
                continue;
            ++present;
            EXPECT_FALSE(GetConfigEmbeddingVocab(arch, solver).empty())
                << arch << "_" << solver
                << ": table present but vocab empty (rejected / md5 mismatch / no v3 block)";
        }
    // This suite is compiled only with MIOPEN_ENABLE_AI_KERNEL_TUNING (see the guard at the top of
    // the file). All 12 (gfx942 + gfx950) x 6-solver tables ship with this PR, so require every one
    // -- EXPECT_GT(present, 0) would stay green if 11 of 12 .bin files were dropped from packaging.
    EXPECT_EQ(present, static_cast<int>(arches.size() * solvers.size()))
        << "expected all " << arches.size() * solvers.size()
        << " shipped config-embedding tables installed at " << db << ", found " << present;
}

// The loader loads a well-formed v3 table and fails closed on every malformed one -- bad magic, a
// non-v3 version, a row-region size mismatch, an md5 that does not match the input encoder (stale
// table vs refreshed model), and a truncated kernel-id block. Crafted in a temp directory and read
// through LoadConfigEmbeddingVocabFromFile, so nothing is written into the installed system db
// (which is normally read-only, and would make the test GTEST_SKIP / silently never run).
TEST(CPU_ConfigEmbeddingTableLoader_NONE, RejectsMalformedTables)
{
    namespace fs = miopen::fs;
    std::random_device rd;
    const auto dir =
        fs::temp_directory_path() / ("miopen_cet_" + std::to_string(rd()) + std::to_string(rd()));
    fs::create_directories(dir);

    auto put_u32 = [](std::vector<char>& b, std::uint32_t v) {
        const char* c = reinterpret_cast<const char*>(&v);
        b.insert(b.end(), c, c + 4);
    };
    auto put_f32 = [](std::vector<char>& b, float v) {
        const char* c = reinterpret_cast<const char*>(&v);
        b.insert(b.end(), c, c + 4);
    };
    // Serialize a MICE table: magic[4] + 6 u32 header (+ 32-byte md5 for v3) + num_rows (key,emb)
    // rows; for v3 also append the length-prefixed kernel-id block. truncate_ids makes the last
    // id's length prefix run past EOF. claimed_rows can exceed the rows actually written (size
    // mismatch).
    auto make = [&](const char* magic,
                    std::uint32_t version,
                    std::uint32_t key_dim,
                    std::uint32_t emb_dim,
                    std::uint32_t written_rows,
                    std::uint32_t claimed_rows,
                    const std::string& md5_str,
                    const std::vector<std::string>& ids,
                    bool truncate_ids) {
        std::vector<char> b;
        b.insert(b.end(), magic, magic + 4);
        put_u32(b, version);
        put_u32(b, key_dim);
        put_u32(b, emb_dim);
        put_u32(b, claimed_rows);
        put_u32(b, 0); // key_dtype = fp32
        put_u32(b, 0); // emb_dtype = fp32
        if(version == 3)
        {
            std::string h = md5_str;
            h.resize(32, '0');
            b.insert(b.end(), h.begin(), h.end());
        }
        for(std::uint32_t r = 0; r < written_rows; ++r)
        {
            for(std::uint32_t k = 0; k < key_dim; ++k)
                put_f32(b, static_cast<float>(r + k));
            for(std::uint32_t e = 0; e < emb_dim; ++e)
                put_f32(b, 0.5f * static_cast<float>(r) + static_cast<float>(e));
        }
        if(version == 3)
        {
            for(std::uint32_t r = 0; r < written_rows; ++r)
            {
                if(truncate_ids && r + 1 == written_rows)
                {
                    put_u32(b, 1u << 30); // length prefix runs far past EOF
                    break;
                }
                const std::string id = r < ids.size() ? ids[r] : std::string();
                put_u32(b, static_cast<std::uint32_t>(id.size()));
                b.insert(b.end(), id.begin(), id.end());
            }
        }
        return b;
    };
    auto write = [&](const std::string& name, const std::vector<char>& bytes) {
        std::ofstream os(dir / name, std::ios::binary);
        os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };
    auto vocab_of = [&](const std::string& name, const std::string& enc) {
        return LoadConfigEmbeddingVocabFromFile((dir / name).string(), enc);
    };

    // Input encoder the table hashes against, with the matching md5 for the positive/mismatch
    // cases.
    const std::string enc_bytes = "miopen-test-input-encoder-bytes";
    write("enc.bin", std::vector<char>(enc_bytes.begin(), enc_bytes.end()));
    const auto enc_path                = (dir / "enc.bin").string();
    const std::string good_md5         = miopen::md5(enc_bytes);
    const std::vector<std::string> ids = {"KernelA", "KernelB"};

    // Positive control: a well-formed v3 table loads to a 2-entry vocab.
    write("good.bin", make("MICE", 3, 2, 3, 2, 2, good_md5, ids, false));
    EXPECT_EQ(vocab_of("good.bin", enc_path).size(), 2u);

    // bad magic.
    write("badmagic.bin", make("XXXX", 3, 2, 3, 2, 2, good_md5, ids, false));
    EXPECT_TRUE(vocab_of("badmagic.bin", enc_path).empty());

    // non-v3 version (no kernel-id block -> no usable vocab).
    write("badver.bin", make("MICE", 2, 2, 3, 2, 2, good_md5, ids, false));
    EXPECT_TRUE(vocab_of("badver.bin", enc_path).empty());

    // size mismatch: header claims 10 rows, only 2 written.
    write("badsize.bin", make("MICE", 3, 2, 3, 2, 10, good_md5, ids, false));
    EXPECT_TRUE(vocab_of("badsize.bin", enc_path).empty());

    // md5 mismatch: table hash does not match the input encoder (stale table / refreshed model).
    write("badmd5.bin", make("MICE", 3, 2, 3, 2, 2, std::string(32, 'f'), ids, false));
    EXPECT_TRUE(vocab_of("badmd5.bin", enc_path).empty());

    // truncated kernel-id block: a length prefix runs past EOF.
    write("badids.bin", make("MICE", 3, 2, 3, 2, 2, good_md5, ids, true));
    EXPECT_TRUE(vocab_of("badids.bin", enc_path).empty());

    fs::remove_all(dir);
}

#else
// Add a dummy test when AI kernel tuning is disabled
TEST(CPU_CandidateSelectionDisabled_NONE, FeatureDisabled)
{
    GTEST_SKIP() << "AI candidate selection features are disabled in this build";
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
