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

// Tests for ND (2D and 3D) convolution AI heuristics (TunaNetND)
// Build: make test_conv_ai_nd_heuristics
// Run all: ./bin/test_conv_ai_nd_heuristics
// Run specific:
// ./bin/test_conv_ai_nd_heuristics --gtest_filter=*2D*
// ./bin/test_conv_ai_nd_heuristics --gtest_filter=*3D*
// Enable logs: MIOPEN_LOG_LEVEL=6 ./bin/test_conv_ai_nd_heuristics

#include <gtest/gtest.h>
#include <miopen/config.h>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <fdeep/fdeep.hpp>
#include <sstream>
#include <string>
#include <vector>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

namespace {

using namespace miopen;
using namespace miopen::ai;

// Test parameters for ND convolution tests
struct ConvNDParams
{
    int spatial_dim; // 2 for 2D, 3 for 3D
    std::string test_name;

    ConvNDParams(int dim, std::string name) : spatial_dim(dim), test_name(std::move(name)) {}
};

// Test suite for ND convolution AI heuristics (TunaNetND)
class GPU_ConvNDAIHeuristics_FP32 : public ::testing::TestWithParam<ConvNDParams>
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;
    std::string device_name; // Detected device name (e.g., "gfx942", "gfx950")
    int spatial_dim;         // 2 or 3

    GPU_ConvNDAIHeuristics_FP32() : ctx(&handle), spatial_dim(2) {}

    void SetUp() override
    {
        // Get test parameters
        const auto& params = GetParam();
        spatial_dim        = params.spatial_dim;

        // Get actual device name from handle
        device_name = handle.GetDeviceName();

        // Check if device is in the list of devices that support ND models
        if(device_name != "gfx942" && device_name != "gfx950")
        {
            GTEST_SKIP() << "Device " << device_name << " not in supported list for " << spatial_dim
                         << "D TunaNet models";
        }

        // Skip if model files don't exist for this device
        if(!ModelFilesExist(device_name, spatial_dim))
        {
            GTEST_SKIP() << device_name << " " << spatial_dim << "D model files not found";
        }
    }

    // Helper to create a 2D convolution problem (NCHW layout)
    miopen::conv::ProblemDescription
    Create2DProblem(int n                             = 1,
                    int c                             = 4,
                    int h                             = 8,
                    int w                             = 8,
                    int k                             = 8,
                    int y                             = 3,
                    int x                             = 3,
                    int pad_h                         = 0,
                    int pad_w                         = 0,
                    int stride_h                      = 1,
                    int stride_w                      = 1,
                    int dilation_h                    = 1,
                    int dilation_w                    = 1,
                    miopen::conv::Direction direction = miopen::conv::Direction::Forward,
                    miopenDataType_t dataType         = miopenFloat)
    {
        // Create tensors for 2D convolution (NCHW layout)
        miopen::TensorDescriptor inputTensor(dataType, {n, c, h, w});
        miopen::TensorDescriptor weightsTensor(dataType, {k, c, y, x});

        // Calculate output dimensions
        int out_h = (h + 2 * pad_h - dilation_h * (y - 1) - 1) / stride_h + 1;
        int out_w = (w + 2 * pad_w - dilation_w * (x - 1) - 1) / stride_w + 1;
        miopen::TensorDescriptor outputTensor(dataType, {n, k, out_h, out_w});

        // Create convolution descriptor
        miopen::ConvolutionDescriptor convDesc(2, // spatial_dim = 2 for 2D
                                               miopenConvolution,
                                               miopenPaddingDefault,
                                               {pad_h, pad_w},
                                               {stride_h, stride_w},
                                               {dilation_h, dilation_w},
                                               {0, 0}, // trans_output_pads for 2D
                                               1,      // group_count
                                               1.0f);  // lowp_quant

        return miopen::conv::ProblemDescription(
            direction == miopen::conv::Direction::Forward ? inputTensor : outputTensor,
            weightsTensor,
            direction == miopen::conv::Direction::Forward ? outputTensor : inputTensor,
            convDesc,
            direction);
    }

    // Helper to create a 3D convolution problem (NCDHW layout)
    miopen::conv::ProblemDescription
    Create3DProblem(int n                             = 1,
                    int c                             = 4,
                    int d                             = 8,
                    int h                             = 8,
                    int w                             = 8,
                    int k                             = 8,
                    int z                             = 3,
                    int y                             = 3,
                    int x                             = 3,
                    int pad_d                         = 0,
                    int pad_h                         = 0,
                    int pad_w                         = 0,
                    int stride_d                      = 1,
                    int stride_h                      = 1,
                    int stride_w                      = 1,
                    int dilation_d                    = 1,
                    int dilation_h                    = 1,
                    int dilation_w                    = 1,
                    miopen::conv::Direction direction = miopen::conv::Direction::Forward,
                    miopenDataType_t dataType         = miopenFloat)
    {
        // Create tensors for 3D convolution (NCDHW layout)
        miopen::TensorDescriptor inputTensor(dataType, {n, c, d, h, w});
        miopen::TensorDescriptor weightsTensor(dataType, {k, c, z, y, x});

        // Calculate output dimensions
        int out_d = (d + 2 * pad_d - dilation_d * (z - 1) - 1) / stride_d + 1;
        int out_h = (h + 2 * pad_h - dilation_h * (y - 1) - 1) / stride_h + 1;
        int out_w = (w + 2 * pad_w - dilation_w * (x - 1) - 1) / stride_w + 1;
        miopen::TensorDescriptor outputTensor(dataType, {n, k, out_d, out_h, out_w});

        // Create convolution descriptor
        miopen::ConvolutionDescriptor convDesc(3, // spatial_dim = 3 for 3D
                                               miopenConvolution,
                                               miopenPaddingDefault,
                                               {pad_d, pad_h, pad_w},
                                               {stride_d, stride_h, stride_w},
                                               {dilation_d, dilation_h, dilation_w},
                                               {0, 0, 0}, // trans_output_pads for 3D
                                               1,         // group_count
                                               1.0f);     // lowp_quant

        return miopen::conv::ProblemDescription(
            direction == miopen::conv::Direction::Forward ? inputTensor : outputTensor,
            weightsTensor,
            direction == miopen::conv::Direction::Forward ? outputTensor : inputTensor,
            convDesc,
            direction);
    }

    // Helper to create a problem based on spatial_dim
    miopen::conv::ProblemDescription
    CreateProblem(miopen::conv::Direction direction = miopen::conv::Direction::Forward,
                  miopenDataType_t dataType         = miopenFloat)
    {
        if(spatial_dim == 2)
            return Create2DProblem(1, 4, 8, 8, 8, 3, 3, 0, 0, 1, 1, 1, 1, direction, dataType);
        else
            return Create3DProblem(
                1, 4, 8, 8, 8, 8, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, direction, dataType);
    }

    // Helper to check if required model files exist for a device
    bool ModelFilesExist(const std::string& device, int dim)
    {
        // For 3D: use device_3d (e.g., "gfx942_3d")
        // For 2D: use device (e.g., "gfx942")
        std::string arch = (dim == 3) ? device + "_3d" : device;

        auto model_path    = GetSystemDbPath() / (arch + ".tn.model");
        auto metadata_path = GetSystemDbPath() / (arch + "_metadata.tn.model");
        return fs::exists(model_path) && fs::exists(metadata_path);
    }

    // Get the appropriate layout strings for the dimension
    std::string GetDefaultLayout() const { return spatial_dim == 2 ? "NCHW" : "NCDHW"; }

    std::string GetAlternativeLayout() const { return spatial_dim == 2 ? "NHWC" : "NDHWC"; }
};

// --- MetadataND tests ---
// cppcheck-suppress syntaxError
TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_LoadValidArchitecture)
{
    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());

    // Expected model prefix depends on dimension
    std::string expected_prefix = (spatial_dim == 3) ? device_name + "_3d" : device_name;
    EXPECT_EQ(metadata.GetModelPrefix(), expected_prefix);
    EXPECT_GT(metadata.GetNumInputs(), 0);
    EXPECT_GT(metadata.GetNumOutputs(), 0);
    EXPECT_GT(metadata.GetNumSolvers(), 0);
    EXPECT_FALSE(metadata.GetFeatures().empty());
    EXPECT_FALSE(metadata.GetSolverMap().empty());
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_EncodeDirection)
{
    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());
    auto fwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::Forward);
    auto bwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardData);
    auto wrw_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardWeights);
    EXPECT_NE(fwd_encoded, bwd_encoded);
    EXPECT_NE(fwd_encoded, wrw_encoded);
    EXPECT_NE(bwd_encoded, wrw_encoded);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_EncodePrecision)
{
    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());
    auto fp32_encoded = metadata.EncodePrecision(miopenFloat);
    auto fp16_encoded = metadata.EncodePrecision(miopenHalf);
    auto bf16_encoded = metadata.EncodePrecision(miopenBFloat16);
    EXPECT_NE(fp32_encoded, fp16_encoded);
    EXPECT_NE(fp32_encoded, bf16_encoded);
    EXPECT_NE(fp16_encoded, bf16_encoded);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_EncodePrecisionUnsupportedThrows)
{
    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());
    // A datatype no model encodes (here miopenInt32) must throw so the caller falls back to the
    // non-AI heuristic, rather than silently returning an out-of-range index that would feed an
    // all-zero precision one-hot into the model.
    EXPECT_ANY_THROW(metadata.EncodePrecision(miopenInt32));
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_EncodeLayouts)
{
    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());

    std::string default_layout     = GetDefaultLayout();
    std::string alternative_layout = GetAlternativeLayout();

    auto default_in  = metadata.EncodeInLayout(default_layout);
    auto alt_in      = metadata.EncodeInLayout(alternative_layout);
    auto default_out = metadata.EncodeOutLayout(default_layout);
    auto alt_out     = metadata.EncodeOutLayout(alternative_layout);

    EXPECT_EQ(default_in, 0);
    EXPECT_EQ(alt_in, 1);
    EXPECT_EQ(default_out, 0);
    EXPECT_EQ(alt_out, 1);

    auto invalid = metadata.EncodeInLayout("INVALID_LAYOUT");
    EXPECT_EQ(invalid, 0);

    // Test wrong dimension layout
    std::string wrong_dim_layout = spatial_dim == 2 ? "NCDHW" : "NCHW";
    auto invalid2                = metadata.EncodeInLayout(wrong_dim_layout);
    EXPECT_EQ(invalid2, 0);
}

// --- ModelND tests ---
TEST_P(GPU_ConvNDAIHeuristics_FP32, GetNDModel_SupportedDevice)
{
    auto model = immed_mode::GetNDModel(device_name, spatial_dim);
    ASSERT_NE(model, nullptr);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, ModelND_IsProblemSupported_CorrectDimension)
{
    auto model = immed_mode::GetNDModel(device_name, spatial_dim);
    ASSERT_NE(model, nullptr);
    auto problem = CreateProblem();
    EXPECT_TRUE(model->IsProblemSupported(problem, ctx));
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, ModelND_IsProblemSupported_WrongDimension)
{
    auto model = immed_mode::GetNDModel(device_name, spatial_dim);
    ASSERT_NE(model, nullptr);

    // Create a problem with the opposite dimension
    miopen::conv::ProblemDescription wrong_dim_problem;
    if(spatial_dim == 2)
    {
        wrong_dim_problem = Create3DProblem();
    }
    else
    {
        wrong_dim_problem = Create2DProblem();
    }

    EXPECT_FALSE(model->IsProblemSupported(wrong_dim_problem, ctx));
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, ModelND_Forward_ReturnsValidPredictions)
{
    auto model = immed_mode::GetNDModel(device_name, spatial_dim);
    ASSERT_NE(model, nullptr);
    auto problem = CreateProblem();
    ASSERT_TRUE(model->IsProblemSupported(problem, ctx));
    auto predictions = model->Forward(problem);
    EXPECT_FALSE(predictions.empty());
    const auto& solver_map = model->GetSolverMap();
    EXPECT_EQ(predictions.size(), solver_map.size());
    auto max_it = std::max_element(predictions.begin(), predictions.end());
    EXPECT_NE(max_it, predictions.end());
    auto min_it = std::min_element(predictions.begin(), predictions.end());
    EXPECT_NE(*max_it, *min_it);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, PredictSolver_ReturnsSolvers)
{
    auto problem = CreateProblem();
    auto solvers = immed_mode::PredictSolver(problem, ctx, device_name);
    EXPECT_FALSE(solvers.empty());
    for(auto solver_id : solvers)
        EXPECT_GT(solver_id, 0);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, PredictSolver_UsesCaching)
{
    auto problem  = CreateProblem();
    auto solvers1 = immed_mode::PredictSolver(problem, ctx, device_name);
    auto solvers2 = immed_mode::PredictSolver(problem, ctx, device_name);
    EXPECT_EQ(solvers1, solvers2);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, ModelND_DifferentProblemSizes)
{
    auto model = immed_mode::GetNDModel(device_name, spatial_dim);
    ASSERT_NE(model, nullptr);

    if(spatial_dim == 3)
    {
        std::vector<std::tuple<int, int, int, int, int>> test_cases_3d = {
            {1, 64, 16, 16, 16}, {4, 128, 32, 32, 32}, {8, 256, 8, 8, 8}};
        for(const auto& [n, c, d, h, w] : test_cases_3d)
        {
            auto problem = Create3DProblem(n, c, d, h, w);
            if(model->IsProblemSupported(problem, ctx))
            {
                auto predictions = model->Forward(problem);
                EXPECT_FALSE(predictions.empty());
            }
        }
    }
    else
    {
        std::vector<std::tuple<int, int, int, int>> test_cases_2d = {
            {1, 64, 16, 16}, {4, 128, 32, 32}, {8, 256, 8, 8}};
        for(const auto& [n, c, h, w] : test_cases_2d)
        {
            auto problem = Create2DProblem(n, c, h, w);
            if(model->IsProblemSupported(problem, ctx))
            {
                auto predictions = model->Forward(problem);
                EXPECT_FALSE(predictions.empty());
            }
        }
    }
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, MetadataND_OptionalPattern)
{
    immed_mode::MetadataND invalid_metadata("nonexistent", spatial_dim);
    EXPECT_FALSE(invalid_metadata.IsValid());
    EXPECT_EQ(invalid_metadata.GetNumInputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumOutputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumSolvers(), 0);
    EXPECT_TRUE(invalid_metadata.GetFeatures().empty());
    EXPECT_TRUE(invalid_metadata.GetSolverMap().empty());
    EXPECT_EQ(invalid_metadata.EncodeDirection(miopen::conv::Direction::Forward), 0);
    EXPECT_EQ(invalid_metadata.EncodePrecision(miopenFloat), 0);
    EXPECT_EQ(invalid_metadata.EncodeInLayout(GetDefaultLayout()), 0);
}

TEST_P(GPU_ConvNDAIHeuristics_FP32, TestMIOpenDriverEquivalent)
{
    miopen::conv::ProblemDescription problem;
    if(spatial_dim == 3)
    {
        problem = Create3DProblem(
            1, 4, 8, 8, 8, 8, 3, 3, 3, 0, 0, 0, 3, 3, 3, 1, 1, 1, miopen::conv::Direction::Forward);
    }
    else
    {
        problem = Create2DProblem(
            1, 4, 8, 8, 8, 3, 3, 0, 0, 3, 3, 1, 1, miopen::conv::Direction::Forward);
    }

    auto solver_ids = immed_mode::PredictSolver(problem, ctx, device_name);
    EXPECT_FALSE(solver_ids.empty()) << "PredictSolver should return solvers";
}

// Test case generator
std::vector<ConvNDParams> GenerateConvNDParams()
{
    return {
        ConvNDParams(2, "Conv2D"),
        ConvNDParams(3, "Conv3D"),
    };
}

// The TunaNet 2D engineered input vector must stay stable across refactors. This pins the assembly
// order and the metadata-driven one-hot widths/indices (layouts, precision, direction) and num_cu,
// by composing the expected vector from the metadata; the derived-feature math is pinned
// deterministically by CPU_ConvAiEngineeredConvFeatures_NONE.Golden.
TEST_P(GPU_ConvNDAIHeuristics_FP32, ExtractTunaNet2dFeaturesGolden)
{
    if(spatial_dim != 2)
        GTEST_SKIP() << "Engineered input features are 2D-only";

    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());

    // Fixed 2D forward problem (NCHW, FP32): out dims equal in dims for 3x3, pad 1, stride 1.
    const auto problem = Create2DProblem(
        1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopen::conv::Direction::Forward, miopenFloat);

    const auto features = immed_mode::ExtractTunaNetND2dFeatures(problem, /*isFwd=*/true, metadata);

    std::vector<float> expected;
    const auto append_one_hot = [&](size_t index, size_t width) {
        std::vector<float> one_hot(width, 0.0f);
        if(index < width)
            one_hot.at(index) = 1.0f;
        expected.insert(expected.end(), one_hot.begin(), one_hot.end());
    };
    append_one_hot(metadata.EncodeInLayout(problem.GetInLayout()),
                   metadata.GetInLayoutClassCount());
    append_one_hot(metadata.EncodeFilLayout(problem.GetWeightsLayout()),
                   metadata.GetFilLayoutClassCount());
    append_one_hot(metadata.EncodeOutLayout(problem.GetOutLayout()),
                   metadata.GetOutLayoutClassCount());
    append_one_hot(metadata.EncodePrecision(problem.GetInDataType()),
                   metadata.GetPrecisionClassCount());
    append_one_hot(metadata.EncodeDirection(problem.GetDirection()),
                   metadata.GetDirectionClassCount());

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

    const auto derived = common::EngineeredConvFeatures(
        1, 64, 64, 56, 56, 56, 56, 3, 3, 1, metadata.GetNumCu(), common::ConvDirection::Forward);
    expected.insert(expected.end(), derived.begin(), derived.end());

    ASSERT_EQ(features.size(), expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(features[i], expected[i]) << "TunaNet feature mismatch at index " << i;
}

// Hardcoded counterpart to ExtractTunaNet2dFeaturesGolden: the one-hot widths/indices and num_cu
// are written as literals rather than re-derived from the metadata, so a retrain that silently
// changes the precision class count (e.g. dropping INT8: 4 -> 3) or num_cu is caught here -- the
// golden test above cannot catch it because it composes the expected vector from the same metadata
// accessors.
TEST_P(GPU_ConvNDAIHeuristics_FP32, ExtractTunaNet2dFeaturesHardcoded)
{
    if(spatial_dim != 2)
        GTEST_SKIP() << "Engineered input features are 2D-only";
    if(device_name != "gfx942" && device_name != "gfx950")
        GTEST_SKIP() << "Hardcoded TunaNet 2D metadata contract is gfx942/gfx950 only";

    immed_mode::MetadataND metadata(device_name, spatial_dim);
    ASSERT_TRUE(metadata.IsValid());

    // Contract of the shipped gfx942/gfx950 TunaNet 2D metadata, as literals.
    ASSERT_EQ(metadata.GetInLayoutClassCount(), 2u); // NCHW, NHWC
    ASSERT_EQ(metadata.GetFilLayoutClassCount(), 2u);
    ASSERT_EQ(metadata.GetOutLayoutClassCount(), 2u);
    ASSERT_EQ(metadata.GetPrecisionClassCount(), 4u); // BF16, FP16, FP32, INT8
    ASSERT_EQ(metadata.GetDirectionClassCount(), 3u); // B, F, W
    const size_t expected_num_cu = (device_name == "gfx942") ? 304u : 256u;
    ASSERT_EQ(metadata.GetNumCu(), expected_num_cu);

    const auto problem = Create2DProblem(
        1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, miopen::conv::Direction::Forward, miopenFloat);
    const auto features = immed_mode::ExtractTunaNetND2dFeatures(problem, /*isFwd=*/true, metadata);

    // Categorical one-hots for an NCHW / FP32 / Forward problem, written out by hand:
    //   in/fil/out layout = NCHW (index 0 of 2); precision = FP32 (index 2 of 4);
    //   direction = Forward 'F' (index 1 of 3).
    std::vector<float> expected = {
        1.0f,
        0.0f, // in_layout  NCHW
        1.0f,
        0.0f, // fil_layout NCHW
        1.0f,
        0.0f, // out_layout NCHW
        0.0f,
        0.0f,
        1.0f,
        0.0f, // precision  FP32
        0.0f,
        1.0f,
        0.0f, // direction  Forward
    };
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
    const auto derived = common::EngineeredConvFeatures(
        1, 64, 64, 56, 56, 56, 56, 3, 3, 1, expected_num_cu, common::ConvDirection::Forward);
    expected.insert(expected.end(), derived.begin(), derived.end());

    ASSERT_EQ(features.size(), expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(features[i], expected[i])
            << "TunaNet hardcoded feature mismatch at index " << i;
}

// Test name generator
std::string ConvNDParamName(const ::testing::TestParamInfo<ConvNDParams>& info)
{
    return info.param.test_name;
}

// Instantiate tests for both 2D and 3D
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvNDAIHeuristics_FP32,
                         ::testing::ValuesIn(GenerateConvNDParams()),
                         ConvNDParamName);

// Golden vectors for the derived-feature block shared by the TunaNet (ExtractTunaNetND2dFeatures)
// and candidate-selection input encoders via common::EngineeredConvFeatures. Pins the shared math
// so a hasty change on either path is caught. Uses C_in != C_out and all three directions so the
// direction-dependent GEMM (M, N, K) assignment is fully exercised (C_in == C_out would hide it).
// Pure CPU math -- no model files or device required.
TEST(CPU_ConvAiEngineeredConvFeatures_NONE, Golden)
{
    // N=2, C_in=64, C_out=128, 56x56, 3x3, g=1, num_cu=304. Only the GEMM-dimension features
    // (indices 1..6) differ between directions; the rest are direction-independent.
    struct Case
    {
        common::ConvDirection dir;
        std::vector<float> expected;
    };
    const std::vector<Case> cases = {
        {common::ConvDirection::Forward,
         {20.645136f,
          8.74401f,
          4.1743873f,
          7.0501225f,
          98.0f,
          5.4444444f,
          0.055555556f,
          19.951988f,
          7.8792317f,
          1.0f,
          0.002869898f,
          0.5f,
          0.015625f,
          4.0430513f,
          4.0430513f,
          4.1743873f,
          4.8598124f,
          1.0986123f}},
        {common::ConvDirection::BackwardData,
         {20.645136f,
          4.8598124f,
          6.3578423f,
          8.74401f,
          0.22222222f,
          0.020408163f,
          0.091836735f,
          19.951988f,
          7.8792317f,
          1.0f,
          0.002869898f,
          0.5f,
          0.015625f,
          4.0430513f,
          4.0430513f,
          4.1743873f,
          4.8598124f,
          1.0986123f}},
        {common::ConvDirection::BackwardWeights,
         {20.645136f,
          8.74401f,
          6.3578423f,
          4.8598124f,
          10.888889f,
          49.0f,
          4.5f,
          19.951988f,
          7.8792317f,
          1.0f,
          0.002869898f,
          0.5f,
          0.015625f,
          4.0430513f,
          4.0430513f,
          4.1743873f,
          4.8598124f,
          1.0986123f}},
    };

    for(const auto& c : cases)
    {
        const auto derived = common::EngineeredConvFeatures(
            /*N=*/2,
            /*C_in=*/64,
            /*C_out=*/128,
            /*H_in=*/56,
            /*W_in=*/56,
            /*H_out=*/56,
            /*W_out=*/56,
            /*K_h=*/3,
            /*K_w=*/3,
            /*groups=*/1,
            /*num_cu=*/304,
            c.dir);
        ASSERT_EQ(derived.size(), c.expected.size());
        for(std::size_t i = 0; i < c.expected.size(); ++i)
            EXPECT_FLOAT_EQ(derived[i], c.expected[i])
                << "derived feature mismatch at index " << i << " (direction "
                << static_cast<int>(c.dir) << ")";
    }
}

// Every bundled fdeep model file must be loadable by the deployed frugally-deep. This fails
// loudly and directly if a model is exported in a format the linked fdeep cannot parse (e.g. a
// Keras 3 export against a Keras 2-era fdeep). Without this, such a breakage only surfaces
// indirectly downstream -- and because the runtime now degrades gracefully to the non-AI
// heuristic on a load failure, it would otherwise be masked in CI. The .tn.model files are
// frugally-deep models EXCEPT the "*_metadata.tn.model" siblings, which are plain JSON read by
// MIOpen directly (not by fdeep). Pure CPU fdeep load -- no device required for the work itself.
TEST(GPU_ConvAIModelLoad_FP32, BundledFdeepModelsLoad)
{
    const auto db = GetSystemDbPath();
    ASSERT_TRUE(fs::exists(db)) << "system db path does not exist: " << db;

    std::vector<std::string> failures;
    std::size_t checked = 0;
    for(const auto& entry : fs::directory_iterator(db))
    {
        const auto path = entry.path();
        const auto name = path.filename().string();
        if(!name.ends_with(".tn.model") || name.find("metadata") != std::string::npos)
            continue;
        ++checked;
        try
        {
            fdeep::load_model(path.string(), true, fdeep::dev_null_logger);
        }
        catch(const std::exception& e)
        {
            failures.push_back(name + " -> " + e.what());
        }
    }
    ASSERT_GT(checked, 0u) << "no fdeep .tn.model files found in " << db;

    std::ostringstream oss;
    for(const auto& f : failures)
        oss << "\n  " << f;
    EXPECT_TRUE(failures.empty()) << "frugally-deep failed to load " << failures.size() << " of "
                                  << checked << " bundled model(s):" << oss.str();
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
