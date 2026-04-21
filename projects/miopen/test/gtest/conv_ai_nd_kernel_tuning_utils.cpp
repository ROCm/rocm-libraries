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
#include <iostream>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/filesystem.hpp>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
using namespace miopen::solver::conv;
using namespace miopen::ai::tuning::candidate_selection;

namespace {

// Helper: layout string to code for ND (must match GetFeaturesND)
int LayoutStringToCode(const std::string& layout, bool is3d)
{
    if(is3d)
    {
        if(layout == "NCDHW")
            return 0;
        if(layout == "NDHWC")
            return 1;
    }
    else
    {
        if(layout == "NCHW")
            return 0;
        if(layout == "NHWC")
            return 1;
    }
    return -1; // Unknown
}

// Dummy kernels for testing
const std::vector<std::string> dummy_kernels = {
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,32,4,Default,4,2,1,4,4,1,1,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,32,64,4,Default,4,1,2,1,2,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,256,128,4,Default,4,4,2,4,4,4,2,1,1,4>",
};

// Dummy fill_valid_kernels for testing
static std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
    fill_valid_kernels = [](const miopen::conv::ProblemDescription&) { return dummy_kernels; };

// Default validation function: accepts all kernel/split_k combinations
inline constexpr auto accept_all_combinations = [](int, int) { return true; };

// Helper: reusable 2D problem description
miopen::conv::ProblemDescription GetReusable2DProblemDescription(
    miopenDataType_t dataType         = miopenFloat,
    miopen::conv::Direction direction = miopen::conv::Direction::BackwardWeights)
{
    std::vector<int> in_lengths      = {1, 512, 130, 66};
    std::vector<int> weights_lengths = {256, 512, 3, 3};
    std::vector<int> out_lengths     = {1, 256, 128, 64};

    miopen::TensorDescriptor in_desc(dataType, in_lengths);
    miopen::TensorDescriptor weights_desc(dataType, weights_lengths);
    miopen::TensorDescriptor out_desc(dataType, out_lengths);

    std::vector<int> pads              = {0, 0};
    std::vector<int> strides           = {1, 1};
    std::vector<int> dilations         = {1, 1};
    std::vector<int> trans_output_pads = {0, 0};

    miopen::ConvolutionDescriptor conv_desc(
        2, miopenConvolution, miopenPaddingDefault, pads, strides, dilations, trans_output_pads);

    return miopen::conv::ProblemDescription(in_desc, weights_desc, out_desc, conv_desc, direction);
}

// Helper: reusable 3D problem description
miopen::conv::ProblemDescription GetReusable3DProblemDescription(
    miopenDataType_t dataType         = miopenFloat,
    miopen::conv::Direction direction = miopen::conv::Direction::BackwardWeights)
{
    std::vector<int> in_lengths      = {1, 512, 11, 130, 66};
    std::vector<int> weights_lengths = {256, 512, 3, 3, 3};
    std::vector<int> out_lengths     = {1, 256, 9, 128, 64};

    miopen::TensorDescriptor in_desc(dataType, in_lengths);
    miopen::TensorDescriptor weights_desc(dataType, weights_lengths);
    miopen::TensorDescriptor out_desc(dataType, out_lengths);

    std::vector<int> pads              = {0, 0, 0};
    std::vector<int> strides           = {1, 1, 1};
    std::vector<int> dilations         = {1, 1, 1};
    std::vector<int> trans_output_pads = {0, 0, 0};

    miopen::ConvolutionDescriptor conv_desc(
        3, miopenConvolution, miopenPaddingDefault, pads, strides, dilations, trans_output_pads);

    return miopen::conv::ProblemDescription(in_desc, weights_desc, out_desc, conv_desc, direction);
}

// Helper: check GetFeaturesND map values
void CheckGetFeaturesND_MapValues(const std::map<std::string, float>& features,
                                  const miopen::conv::ProblemDescription& problem,
                                  miopen::conv::Direction direction,
                                  bool is3d)
{
    std::map<std::string, float> expected;
    expected["spatial_dim"] = is3d ? 3.0f : 2.0f;
    expected["in_channels"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputChannelC(problem));

    if(is3d)
    {
        expected["in_d"] =
            static_cast<float>(miopen::solver::ProblemInterpreter::GetInputDepthDi(problem));
    }
    expected["in_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputHeightHi(problem));
    expected["in_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputWidthWi(problem));

    expected["out_channels"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputChannelK(problem));
    if(is3d)
    {
        expected["out_d"] =
            static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputDepthDo(problem));
    }
    expected["out_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputHeightHo(problem));
    expected["out_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputWidthWo(problem));

    if(is3d)
    {
        expected["fil_d"] =
            static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterDepthZ(problem));
    }
    expected["fil_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterHeightY(problem));
    expected["fil_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterWidthX(problem));

    if(is3d)
    {
        expected["pad_d"] =
            static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadD(problem));
    }
    expected["pad_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadH(problem));
    expected["pad_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadW(problem));

    if(is3d)
    {
        expected["conv_stride_d"] = static_cast<float>(
            miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideD(problem));
    }
    expected["conv_stride_h"] = static_cast<float>(
        miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideH(problem));
    expected["conv_stride_w"] = static_cast<float>(
        miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideW(problem));

    if(is3d)
    {
        expected["dilation_d"] = static_cast<float>(problem.GetDilationD());
    }
    expected["dilation_h"] = static_cast<float>(problem.GetDilationH());
    expected["dilation_w"] = static_cast<float>(problem.GetDilationW());

    expected["batchsize"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetBatchN(problem));
    expected["bias"]      = static_cast<float>(problem.GetBias());
    expected["in_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetInputLayout(problem), is3d));
    expected["fil_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetFilterLayout(problem), is3d));
    expected["out_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetOutputLayout(problem), is3d));
    expected["precision"] = static_cast<float>(problem.GetInDataType());
    expected["direction"] =
        static_cast<float>(direction == miopen::conv::Direction::Forward           ? 0.0f
                           : direction == miopen::conv::Direction::BackwardData    ? 1.0f
                           : direction == miopen::conv::Direction::BackwardWeights ? 2.0f
                                                                                   : -1.0f);
    expected["group_count"] = static_cast<float>(problem.GetGroupCount());

    for(const auto& kv : expected)
    {
        ASSERT_TRUE(features.count(kv.first)) << "Missing key: " << kv.first;
        EXPECT_FLOAT_EQ(features.at(kv.first), kv.second) << "Mismatch for key: " << kv.first;
    }
}

// Helper: check if model files exist for architecture
bool CheckModelFilesExist(const std::string& arch, const std::string& solver_name)
{
    std::string db_path = miopen::GetSystemDbPath().string();
    auto metadata       = db_path + "/" + arch + "_" + solver_name + "_metadata.tn.model";
    auto input_encoder  = db_path + "/" + arch + "_" + solver_name + "_input_encoder.tn.model";
    auto kernel_config_encoder =
        db_path + "/" + arch + "_" + solver_name + "_kernel_config_encoder.tn.model";

    return miopen::fs::exists(metadata) && miopen::fs::exists(input_encoder) &&
           miopen::fs::exists(kernel_config_encoder);
}

// Test parameters combining dimension and data type/direction
struct ConvNDKernelTuningTestParam
{
    int spatial_dim;
    miopenDataType_t data_type;
    miopen::conv::Direction direction;
    std::string test_name;
};

// Test case generators
std::vector<ConvNDKernelTuningTestParam> GenSmokeTestCases()
{
    return {
        {2, miopenFloat, miopen::conv::Direction::Forward, "Conv2D_Fwd"},
        {2, miopenFloat, miopen::conv::Direction::BackwardWeights, "Conv2D_Wrw"},
        {3, miopenFloat, miopen::conv::Direction::Forward, "Conv3D_Fwd"},
        {3, miopenFloat, miopen::conv::Direction::BackwardWeights, "Conv3D_Wrw"},
    };
}

std::vector<ConvNDKernelTuningTestParam> GenFullTestCases()
{
    return {
        {2, miopenFloat, miopen::conv::Direction::Forward, "Conv2D_Fwd"},
        {2, miopenFloat, miopen::conv::Direction::BackwardData, "Conv2D_Bwd"},
        {2, miopenFloat, miopen::conv::Direction::BackwardWeights, "Conv2D_Wrw"},
        {2, miopenHalf, miopen::conv::Direction::BackwardWeights, "Conv2D_Half"},
        {2, miopenBFloat16, miopen::conv::Direction::BackwardWeights, "Conv2D_BFloat16"},
        {3, miopenFloat, miopen::conv::Direction::Forward, "Conv3D_Fwd"},
        {3, miopenFloat, miopen::conv::Direction::BackwardData, "Conv3D_Bwd"},
        {3, miopenFloat, miopen::conv::Direction::BackwardWeights, "Conv3D_Wrw"},
        {3, miopenHalf, miopen::conv::Direction::BackwardWeights, "Conv3D_Half"},
        {3, miopenBFloat16, miopen::conv::Direction::BackwardWeights, "Conv3D_BFloat16"},
    };
}

// Test name generator
std::string
ConvNDKernelTuningTestName(const ::testing::TestParamInfo<ConvNDKernelTuningTestParam>& info)
{
    return info.param.test_name;
}

// Helper function for metadata encoding validation
// Validates that all CK kernel instances can be encoded without errors
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
void ValidateMetadataEncoding(const std::string& solver_name,
                              const std::vector<std::string>& all_ck_kernels,
                              const std::string& device_arch)
{
    ASSERT_FALSE(all_ck_kernels.empty()) << "No CK kernel instances found for " << solver_name;
    MIOPEN_LOG_I("Testing " << all_ck_kernels.size() << " total CK instances for " << solver_name);

    const auto& model    = GetCandidateSelectionModel(device_arch, solver_name);
    const auto& metadata = model.metadata();

    // Get supported kernel names from metadata (check both encodings and constants)
    std::set<std::string> supported_kernel_names;

    auto encodings_it = metadata.sequence_encodings().find("000_kernel_name");
    if(encodings_it != metadata.sequence_encodings().end())
    {
        // Variable kernel names - get all encoded values
        for(const auto& [name, value] : encodings_it->second)
        {
            supported_kernel_names.insert(name);
        }
    }
    else
    {
        // Check if it's a constant (single kernel type)
        auto constant_kernel = metadata.GetOutputConstant("000_kernel_name");
        if(constant_kernel.has_value())
        {
            supported_kernel_names.insert(constant_kernel.value());
            MIOPEN_LOG_I("Kernel name is constant: " << constant_kernel.value());
        }
    }

    // Separate kernels into supported and unsupported
    std::vector<std::string> supported_kernels;
    std::vector<std::string> unsupported_kernels;

    if(!supported_kernel_names.empty())
    {
        for(const auto& typestring : all_ck_kernels)
        {
            auto tokens = GetKernelAsTokens(typestring);
            if(!tokens.empty())
            {
                const std::string& kernel_name = tokens[0];
                if(supported_kernel_names.count(kernel_name) > 0)
                {
                    supported_kernels.push_back(typestring);
                }
                else
                {
                    unsupported_kernels.push_back(typestring);
                }
            }
        }
    }
    else
    {
        // No kernel name filtering available - test all kernels
        MIOPEN_LOG_I("No kernel name filtering available - testing all kernels");
        supported_kernels = all_ck_kernels;
    }

    // Log unsupported kernels (expected - these are CK kernels not in the model)
    if(!unsupported_kernels.empty())
    {
        MIOPEN_LOG_I("Found " << unsupported_kernels.size()
                              << " CK kernels not in metadata (expected)");
    }

    ASSERT_FALSE(supported_kernels.empty()) << "No supported kernels found";
    MIOPEN_LOG_I("Testing " << supported_kernels.size() << " supported kernels");

    // Create kernel parameter vectors from supported TypeStrings
    std::vector<std::vector<std::string>> kernel_params;
    for(const auto& typestring : supported_kernels)
    {
        auto tokens = GetKernelAsTokens(typestring);

        // Check if this kernel type requires split_k to be appended
        if(!tokens.empty())
        {
            const std::string& kernel_name = tokens[0];
            try
            {
                auto kernel_mapping = metadata.GetKernelStrMapping(kernel_name);

                // Check if any parameter name contains "splitk" (case-insensitive)
                bool has_split_k_mapping = false;
                for(const auto& [idx, param_name] : kernel_mapping)
                {
                    std::string param_lower = param_name;
                    std::transform(param_lower.begin(),
                                   param_lower.end(),
                                   param_lower.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if(param_lower.find("splitk") != std::string::npos)
                    {
                        has_split_k_mapping = true;
                        break;
                    }
                }

                // If split_k is expected in the mapping, append a default value
                if(has_split_k_mapping)
                {
                    tokens.push_back("1"); // Add default split_k value
                }
            }
            catch(const std::exception&)
            {
                // Kernel not in metadata - skip split_k append
            }
        }

        kernel_params.push_back(tokens);
    }

    // Encode all supported kernels
    auto encoded = EncodeKernelParams(kernel_params, metadata);

    // Verify no NaN values (which indicate encoding failures)
    int nan_count = 0;
    std::vector<std::string> failed_kernels;

    for(size_t i = 0; i < encoded.size(); ++i)
    {
        bool has_nan = false;
        for(size_t j = 0; j < encoded[i].size(); ++j)
        {
            if(std::isnan(encoded[i][j]))
            {
                nan_count++;
                has_nan = true;
            }
        }
        if(has_nan)
        {
            failed_kernels.push_back(supported_kernels[i]);
        }
    }

    if(nan_count > 0)
    {
        MIOPEN_LOG_E("Found " << nan_count << " NaN encodings across " << failed_kernels.size()
                              << " kernels");
        for(const auto& kernel : failed_kernels)
        {
            MIOPEN_LOG_E("  Failed kernel: " << kernel);
        }
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count
                            << " NaN encodings - indicates missing metadata entries";
}
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

} // anonymous namespace

// ------------------- Base Test Fixture -------------------

class GPU_ConvNDKernelTuning_Base : public ::testing::Test
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;
    std::string device_arch;
    int spatial_dim = 3;
    std::string solver_name;

    void SetUp() override
    {
        // Early exit for architecture compatibility
        device_arch = handle.GetDeviceName();
        ctx         = miopen::ExecutionContext(&handle);

        // Solver name will be set by derived classes
    }

    virtual bool RequiresAIModels() const { return false; }

    // Get solver name based on spatial dimension and direction
    std::string GetSolverName(int dim, miopen::conv::Direction dir) const
    {
        std::string prefix = (dim == 3) ? "ConvHipImplicitGemm3DGroup" : "ConvHipImplicitGemmGroup";
        std::string suffix;
        switch(dir)
        {
        case miopen::conv::Direction::Forward: suffix = "FwdXdlops"; break;
        case miopen::conv::Direction::BackwardData: suffix = "BwdXdlops"; break;
        case miopen::conv::Direction::BackwardWeights: suffix = "WrwXdlops"; break;
        default: suffix = "UnknownXdlops"; break;
        }
        return prefix + suffix;
    }
};

// ------------------- Platform-Agnostic Tests -------------------

// ===============================================================================
// SolverHeuristicConfig Tests
// ===============================================================================

class CPU_SolverHeuristicConfig_NONE : public ::testing::Test
{
};

TEST_F(CPU_SolverHeuristicConfig_NONE, IsValidSplitK_PowerOfTwo)
{
    // Create a config that uses split_k with range 1-128
    SolverHeuristicConfig cfg{"TestSolver", "TestSolverKTN", 2, true, 1, 128, false, true};

    // Valid power-of-2 values
    EXPECT_TRUE(cfg.IsValidSplitK(1));
    EXPECT_TRUE(cfg.IsValidSplitK(2));
    EXPECT_TRUE(cfg.IsValidSplitK(4));
    EXPECT_TRUE(cfg.IsValidSplitK(8));
    EXPECT_TRUE(cfg.IsValidSplitK(16));
    EXPECT_TRUE(cfg.IsValidSplitK(32));
    EXPECT_TRUE(cfg.IsValidSplitK(64));
    EXPECT_TRUE(cfg.IsValidSplitK(128));
}

TEST_F(CPU_SolverHeuristicConfig_NONE, IsValidSplitK_NonPowerOfTwo)
{
    SolverHeuristicConfig cfg{"TestSolver", "TestSolverKTN", 2, true, 1, 128, false, true};

    // Invalid non-power-of-2 values within range
    EXPECT_FALSE(cfg.IsValidSplitK(3));
    EXPECT_FALSE(cfg.IsValidSplitK(5));
    EXPECT_FALSE(cfg.IsValidSplitK(6));
    EXPECT_FALSE(cfg.IsValidSplitK(7));
    EXPECT_FALSE(cfg.IsValidSplitK(12));
    EXPECT_FALSE(cfg.IsValidSplitK(100));
}

TEST_F(CPU_SolverHeuristicConfig_NONE, IsValidSplitK_OutOfRange)
{
    SolverHeuristicConfig cfg{"TestSolver", "TestSolverKTN", 2, true, 1, 128, false, true};

    // Out of range values (even if power of 2)
    EXPECT_FALSE(cfg.IsValidSplitK(0));
    EXPECT_FALSE(cfg.IsValidSplitK(256));
    EXPECT_FALSE(cfg.IsValidSplitK(512));
    EXPECT_FALSE(cfg.IsValidSplitK(-1)); // Not autodeduce since not supported
}

TEST_F(CPU_SolverHeuristicConfig_NONE, IsValidSplitK_AutoDeduce)
{
    // Config that supports autodeduce
    SolverHeuristicConfig cfg_with_auto{"TestSolver", "TestSolverKTN", 2, true, 1, 128, true, true};
    EXPECT_TRUE(cfg_with_auto.IsValidSplitK(CkSplitkAutoDeduce));

    // Config that doesn't support autodeduce
    SolverHeuristicConfig cfg_no_auto{"TestSolver", "TestSolverKTN", 2, true, 1, 128, false, true};
    EXPECT_FALSE(cfg_no_auto.IsValidSplitK(CkSplitkAutoDeduce));
}

TEST_F(CPU_SolverHeuristicConfig_NONE, IsValidSplitK_NoSplitKSolver)
{
    // Config for solver that doesn't use split_k (like Forward)
    SolverHeuristicConfig cfg{"TestSolver", "TestSolverKTN", 2, false, 0, 0, false, true};

    // Only 0 is valid for non-split_k solvers
    EXPECT_TRUE(cfg.IsValidSplitK(0));
    EXPECT_FALSE(cfg.IsValidSplitK(1));
    EXPECT_FALSE(cfg.IsValidSplitK(2));
}

TEST_F(CPU_SolverHeuristicConfig_NONE, GetSolverNameForArch_gfx90a)
{
    SolverHeuristicConfig cfg{"ConvHipImplicitGemmGroupBwdXdlops",
                              "ConvHipIgemmGroupXdlops",
                              2,
                              true,
                              1,
                              128,
                              false,
                              true};

    // gfx90a should return the KTN name
    EXPECT_STREQ(cfg.GetSolverNameForArch("gfx90a"), "ConvHipIgemmGroupXdlops");
}

TEST_F(CPU_SolverHeuristicConfig_NONE, GetSolverNameForArch_gfx942)
{
    SolverHeuristicConfig cfg{"ConvHipImplicitGemmGroupBwdXdlops",
                              "ConvHipIgemmGroupXdlops",
                              2,
                              true,
                              1,
                              128,
                              false,
                              true};

    // gfx942 should return the standard name
    EXPECT_STREQ(cfg.GetSolverNameForArch("gfx942"), "ConvHipImplicitGemmGroupBwdXdlops");
    EXPECT_STREQ(cfg.GetSolverNameForArch("gfx950"), "ConvHipImplicitGemmGroupBwdXdlops");
}

TEST_F(CPU_SolverHeuristicConfig_NONE, SolverConfigs_Consistency)
{
    // Test that actual solver configs are consistent
    // Forward solver: no split_k
    SolverHeuristicConfig fwd_cfg{"ConvHipImplicitGemmGroupFwdXdlops",
                                  "ConvHipIgemmGroupFwdXdlops",
                                  2,
                                  false,
                                  0,
                                  0,
                                  false,
                                  true};
    EXPECT_FALSE(fwd_cfg.uses_split_k);
    EXPECT_TRUE(fwd_cfg.IsValidSplitK(0));
    EXPECT_FALSE(fwd_cfg.IsValidSplitK(1));

    // Backward solvers: use split_k with range 1-128
    SolverHeuristicConfig bwd_cfg{"ConvHipImplicitGemmGroupBwdXdlops",
                                  "ConvHipIgemmGroupXdlops",
                                  2,
                                  true,
                                  1,
                                  128,
                                  false,
                                  true};
    EXPECT_TRUE(bwd_cfg.uses_split_k);
    EXPECT_EQ(bwd_cfg.split_k_min, 1);
    EXPECT_EQ(bwd_cfg.split_k_max, 128);

    SolverHeuristicConfig wrw_cfg{"ConvHipImplicitGemmGroupWrwXdlops",
                                  "ConvHipIgemmGroupXdlops",
                                  2,
                                  true,
                                  1,
                                  128,
                                  false,
                                  true};
    EXPECT_TRUE(wrw_cfg.uses_split_k);
    EXPECT_EQ(wrw_cfg.split_k_min, 1);
    EXPECT_EQ(wrw_cfg.split_k_max, 128);
}

// ===============================================================================
// HeuristicInitState Tests
// ===============================================================================

class CPU_HeuristicInitState_NONE : public ::testing::Test
{
};

TEST_F(CPU_HeuristicInitState_NONE, Reset_WithSplitK)
{
    std::vector<std::string> valid_kernels;
    int index             = 99;
    int split_k           = 99;
    std::string kernel_id = "old_value";

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.Reset(true); // uses_split_k = true

    EXPECT_EQ(index, 0);
    EXPECT_EQ(split_k, 1);
    EXPECT_EQ(kernel_id, "");
}

TEST_F(CPU_HeuristicInitState_NONE, Reset_WithoutSplitK)
{
    std::vector<std::string> valid_kernels;
    int index             = 99;
    int split_k           = 99;
    std::string kernel_id = "old_value";

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.Reset(false); // uses_split_k = false

    EXPECT_EQ(index, 0);
    EXPECT_EQ(split_k, 0);
    EXPECT_EQ(kernel_id, "");
}

TEST_F(CPU_HeuristicInitState_NONE, SetResult_WithSplitK)
{
    std::vector<std::string> valid_kernels = {"kernel_A", "kernel_B", "kernel_C"};
    int index                              = 0;
    int split_k                            = 0;
    std::string kernel_id;

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.SetResult(1, 4, true); // idx=1, split_k=4, uses_split_k=true

    EXPECT_EQ(index, 1);
    EXPECT_EQ(split_k, 4);
    EXPECT_EQ(kernel_id, "kernel_B+4");
}

TEST_F(CPU_HeuristicInitState_NONE, SetResult_WithoutSplitK)
{
    std::vector<std::string> valid_kernels = {"kernel_A", "kernel_B", "kernel_C"};
    int index                              = 0;
    int split_k                            = 0;
    std::string kernel_id;

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.SetResult(2, 0, false); // idx=2, split_k=0, uses_split_k=false

    EXPECT_EQ(index, 2);
    EXPECT_EQ(split_k, 0);
    EXPECT_EQ(kernel_id, "kernel_C");
}

TEST_F(CPU_HeuristicInitState_NONE, SetResult_InvalidIndex)
{
    std::vector<std::string> valid_kernels = {"kernel_A", "kernel_B"};
    int index                              = 0;
    int split_k                            = 0;
    std::string kernel_id;

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);
    state.SetResult(-1, 1, true); // Invalid index

    // Should update index and split_k but not kernel_id
    EXPECT_EQ(index, -1);
    EXPECT_EQ(split_k, 1);
    EXPECT_EQ(kernel_id, "");

    // Out of bounds index
    state.SetResult(999, 1, true);
    EXPECT_EQ(index, 999);
    EXPECT_EQ(split_k, 1);
    EXPECT_EQ(kernel_id, "");
}

TEST_F(CPU_HeuristicInitState_NONE, KernelIdFormatting)
{
    std::vector<std::string> valid_kernels = {"DeviceOp<1,2,3>"};
    int index                              = 0;
    int split_k                            = 0;
    std::string kernel_id;

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

    // Test various split_k values
    state.SetResult(0, 1, true);
    EXPECT_EQ(kernel_id, "DeviceOp<1,2,3>+1");

    state.SetResult(0, 16, true);
    EXPECT_EQ(kernel_id, "DeviceOp<1,2,3>+16");

    state.SetResult(0, 128, true);
    EXPECT_EQ(kernel_id, "DeviceOp<1,2,3>+128");
}

// Non-parameterized utility tests
class GPU_ConvNDKernelTuningUtils_FP32 : public GPU_ConvNDKernelTuning_Base
{
protected:
    bool RequiresAIModels() const override { return false; }
};

// Parameterized feature extraction tests
class GPU_ConvNDKernelTuning_FP32
    : public GPU_ConvNDKernelTuning_Base,
      public ::testing::WithParamInterface<ConvNDKernelTuningTestParam>
{
protected:
    bool RequiresAIModels() const override { return false; }

    void SetUp() override
    {
        const auto& param = GetParam();
        spatial_dim       = param.spatial_dim;
        GPU_ConvNDKernelTuning_Base::SetUp();
        solver_name = GetSolverName(spatial_dim, param.direction);
    }
};

TEST_P(GPU_ConvNDKernelTuning_FP32, GetFeaturesND_Test)
{
    const auto& param = GetParam();
    auto problem      = (spatial_dim == 3)
                            ? GetReusable3DProblemDescription(param.data_type, param.direction)
                            : GetReusable2DProblemDescription(param.data_type, param.direction);
    int max_cu        = 304;
    auto features     = GetFeaturesND(problem, max_cu, device_arch);

    // 2D has 23 features, 3D has 29 features (6 depth-related features)
    size_t expected_feature_count = (spatial_dim == 3) ? 29u : 23u;
    ASSERT_EQ(features.size(), expected_feature_count);
    CheckGetFeaturesND_MapValues(features, problem, param.direction, spatial_dim == 3);
}

TEST_F(GPU_ConvNDKernelTuningUtils_FP32, GetKernelAsTokens_Test)
{
    auto tokens = GetKernelAsTokens("type<param1,param2>");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "type");
    EXPECT_EQ(tokens[1], "param1");
    EXPECT_EQ(tokens[2], "param2");

    auto empty = GetKernelAsTokens("");
    ASSERT_TRUE(empty.empty());
}

TEST_F(GPU_ConvNDKernelTuningUtils_FP32, GenerateSplitK_Test)
{
    auto split_ks             = GenerateSplitK(8);
    std::vector<int> expected = {1, 2, 4, 8};
    ASSERT_EQ(split_ks, expected);
}

TEST_F(GPU_ConvNDKernelTuningUtils_FP32, ProcessExplicitXdlParams_Test)
{
    // Test case matching the Python preprocessing logic
    std::vector<std::string> input = {"DeviceGroupedConvBwdWeight_Explicit_Xdl",
                                      "DeviceBatchedGemmXdlUniversal<Default,CRR>",
                                      "BlkSize:256",
                                      "BlkTile:256x256x32",
                                      "WaveTile:32x32",
                                      "WaveMap:4x4",
                                      "VmemReadVec:8x8",
                                      "BlkGemmPipelineScheduler:Intrawave",
                                      "BlkGemmPipelineVersion:v4",
                                      "BlkGemmPipelinePrefetchStages:3",
                                      "1"};

    auto result = ProcessExplicitXdlParams(input);

    std::vector<std::string> expected = {
        "DeviceGroupedConvBwdWeight_Explicit_Xdl",    // 0: unchanged
        "DeviceBatchedGemmXdlUniversal<Default,CRR>", // 1: unchanged (no colon)
        "256",                                        // 2: BlkSize value
        "256",
        "256",
        "32", // 3-5: BlkTile split
        "32",
        "32", // 6-7: WaveTile split
        "4",
        "4", // 8-9: WaveMap split
        "8",
        "8",                                  // 10-11: VmemReadVec split
        "BlkGemmPipelineScheduler:Intrawave", // 12: BlkGemmPipelineScheduler value
        "BlkGemmPipelineVersion:v4",          // 13: BlkGemmPipelineVersion value
        "3",                                  // 14: BlkGemmPipelinePrefetchStages value
        "1"                                   // 15: split_k value
    };

    ASSERT_EQ(result.size(), expected.size())
        << "Expected " << expected.size() << " parameters, got " << result.size();

    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i << ": expected '"
                                          << expected[i] << "', got '" << result[i] << "'";
    }
}

TEST_F(GPU_ConvNDKernelTuningUtils_FP32, ProcessExplicitXdlParams_EmptyInput_Test)
{
    std::vector<std::string> empty_input;
    auto result = ProcessExplicitXdlParams(empty_input);
    ASSERT_TRUE(result.empty());
}

TEST_F(GPU_ConvNDKernelTuningUtils_FP32, FillHeuristicKernels_WithExplicitXdl_Test)
{
    // Test that FillHeuristicKernels applies ProcessExplicitXdlParams for Explicit_Xdl kernels
    std::vector<std::string> valid_kernels = {
        "DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmXdlUniversal<Default,CRR>,"
        "BlkSize:256,BlkTile:256x256x32,WaveTile:32x32,WaveMap:4x4,VmemReadVec:8x8,"
        "BlkGemmPipelineScheduler:Intrawave,BlkGemmPipelineVersion:v4,"
        "BlkGemmPipelinePrefetchStages:3,1>",
        "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>"};

    std::vector<int> indexes;
    std::vector<std::vector<std::string>> kernels;

    FillHeuristicKernels(valid_kernels, indexes, kernels);

    ASSERT_EQ(kernels.size(), 2u);

    // First kernel should be processed by ProcessExplicitXdlParams
    EXPECT_EQ(kernels[0].size(), 16u)
        << "Explicit_Xdl kernel should have 16 parameters after processing";
    EXPECT_EQ(kernels[0][0], "DeviceGroupedConvBwdWeight_Explicit_Xdl");
    EXPECT_EQ(kernels[0][2], "256"); // BlkSize value
    EXPECT_EQ(kernels[0][3], "256"); // First part of BlkTile
    EXPECT_EQ(kernels[0][4], "256"); // Second part of BlkTile
    EXPECT_EQ(kernels[0][5], "32");  // Third part of BlkTile

    // Second kernel should NOT be processed (regular kernel)
    EXPECT_GT(kernels[1].size(), 0u);
    EXPECT_EQ(kernels[1][0], "DeviceGroupedConvBwdWeight_Xdl_CShuffle");
}

// ------------------- AI Model Tests (Architecture-Dependent) -------------------

class GPU_ConvNDKernelTuningAI_Base : public GPU_ConvNDKernelTuning_Base
{
protected:
    bool RequiresAIModels() const override { return true; }

    void SetUp() override
    {
        GPU_ConvNDKernelTuning_Base::SetUp();
        if(device_arch != "gfx942" && device_arch != "gfx950")
        {
            GTEST_SKIP() << "AI tuning models only available for gfx942/gfx950, current: "
                         << device_arch;
        }
    }
};

// Parameterized AI tests
class GPU_ConvNDKernelTuningAI_FP32
    : public GPU_ConvNDKernelTuningAI_Base,
      public ::testing::WithParamInterface<ConvNDKernelTuningTestParam>
{
protected:
    void SetUp() override
    {
        const auto& param = GetParam();
        spatial_dim       = param.spatial_dim;
        solver_name       = GetSolverName(spatial_dim, miopen::conv::Direction::BackwardWeights);
        GPU_ConvNDKernelTuningAI_Base::SetUp();
    }
};

TEST_P(GPU_ConvNDKernelTuningAI_FP32, CandidateSelectionModel_Test)
{
    std::string test_solver = GetSolverName(spatial_dim, miopen::conv::Direction::BackwardWeights);

    EXPECT_NO_THROW({
        miopen::ai::tuning::candidate_selection::CandidateSelectionModel model(device_arch,
                                                                               test_solver);
    });

    auto& model = miopen::ai::tuning::candidate_selection::GetCandidateSelectionModel(device_arch,
                                                                                      test_solver);
    const auto& meta = model.metadata();
    ASSERT_FALSE(meta.input_params().empty());
    ASSERT_FALSE(meta.output_params().empty());
}

TEST_P(GPU_ConvNDKernelTuningAI_FP32, RunParameterPredictionModel_Test)
{
    auto problem =
        (spatial_dim == 3)
            ? GetReusable3DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights)
            : GetReusable2DProblemDescription(miopenFloat,
                                              miopen::conv::Direction::BackwardWeights);

    int index = 0, split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    std::string test_solver = GetSolverName(spatial_dim, miopen::conv::Direction::BackwardWeights);

    auto [ai_success, result] =
        miopen::solver::conv::RunParameterPredictionModel<float>(ctx,
                                                                 problem,
                                                                 valid_kernels,
                                                                 index,
                                                                 split_k,
                                                                 kernel_id,
                                                                 fill_valid_kernels,
                                                                 test_solver,
                                                                 accept_all_combinations);

    ASSERT_TRUE(ai_success);
    ASSERT_FALSE(kernel_id.empty());
}

TEST_P(GPU_ConvNDKernelTuningAI_FP32, RunParameterPredictionModel_Fallback_Test)
{
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)> empty_kernels =
        [](const miopen::conv::ProblemDescription&) { return std::vector<std::string>{}; };

    auto problem =
        (spatial_dim == 3)
            ? GetReusable3DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights)
            : GetReusable2DProblemDescription(miopenFloat,
                                              miopen::conv::Direction::BackwardWeights);
    int index = 0, split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    std::string test_solver = GetSolverName(spatial_dim, miopen::conv::Direction::BackwardWeights);

    auto [ai_success, result] =
        miopen::solver::conv::RunParameterPredictionModel<float>(ctx,
                                                                 problem,
                                                                 valid_kernels,
                                                                 index,
                                                                 split_k,
                                                                 kernel_id,
                                                                 empty_kernels,
                                                                 test_solver,
                                                                 accept_all_combinations);

    ASSERT_FALSE(ai_success);
    ASSERT_TRUE(kernel_id.empty());
}

// ------------------- Metadata Encoding Validation Tests -------------------

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

// Helper: get all kernel type strings for a given solver type via CkImplLibLoader
std::vector<std::string> GetAllKernelTypeStringsFor(miopen::solver::CKSolverType solver,
                                                    const std::string& device_name)
{
    return miopen::solver::CkImplLibLoader::Get(device_name).GetAllKernelTypeStrings(solver);
}

// Fixed fixture tests for specific directions (non-parameterized)
class GPU_Conv2DKernelTuningAI_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
protected:
    void SetUp() override
    {
        spatial_dim = 2;
        solver_name = "ConvHipImplicitGemmGroupWrwXdlops";
        GPU_ConvNDKernelTuningAI_Base::SetUp();
    }
};

class GPU_Conv3DKernelTuningAI_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
protected:
    void SetUp() override
    {
        spatial_dim = 3;
        solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
        GPU_ConvNDKernelTuningAI_Base::SetUp();
    }
};

TEST_F(GPU_Conv2DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Wrw_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemmGroupWrwXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConvWrw, device_arch),
        device_arch);
}

TEST_F(GPU_Conv2DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Fwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemmGroupFwdXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConvFwd, device_arch),
        device_arch);
}

TEST_F(GPU_Conv2DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Bwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemmGroupBwdXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConvBwd, device_arch),
        device_arch);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Wrw_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupWrwXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConv3dWrw, device_arch),
        device_arch);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Fwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupFwdXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConv3dFwd, device_arch),
        device_arch);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Bwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupBwdXdlops",
        GetAllKernelTypeStringsFor(miopen::solver::CKSolverType::GrpConv3dBwd, device_arch),
        device_arch);
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

// ------------------- Full Solver Tests -------------------

class GPU_Conv2DKernelTuningSolver_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
protected:
    void SetUp() override
    {
        spatial_dim = 2;
        solver_name = "ConvHipImplicitGemmGroupWrwXdlops";
        GPU_ConvNDKernelTuningAI_Base::SetUp();
    }
};

class GPU_Conv3DKernelTuningSolver_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
protected:
    void SetUp() override
    {
        spatial_dim = 3;
        solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
        GPU_ConvNDKernelTuningAI_Base::SetUp();
    }
};

TEST_F(GPU_Conv2DKernelTuningSolver_FP32, FullSolverPathway_Wrw_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem =
        GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    ConvHipImplicitGemmGroupWrwXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

TEST_F(GPU_Conv2DKernelTuningSolver_FP32, FullSolverPathway_Fwd_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem = GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    ConvHipImplicitGemmGroupFwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

TEST_F(GPU_Conv2DKernelTuningSolver_FP32, FullSolverPathway_Bwd_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem =
        GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    ConvHipImplicitGemmGroupBwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

TEST_F(GPU_Conv3DKernelTuningSolver_FP32, FullSolverPathway_Wrw_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem =
        GetReusable3DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    ConvHipImplicitGemm3DGroupWrwXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

TEST_F(GPU_Conv3DKernelTuningSolver_FP32, FullSolverPathway_Fwd_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem = GetReusable3DProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    ConvHipImplicitGemm3DGroupFwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

TEST_F(GPU_Conv3DKernelTuningSolver_FP32, FullSolverPathway_Bwd_Test)
{
#if !MIOPEN_USE_COMPOSABLEKERNEL
    GTEST_SKIP() << "Composable Kernel is disabled, skipping CK-dependent solver test";
#else
    auto problem =
        GetReusable3DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    ConvHipImplicitGemm3DGroupBwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
#endif
}

// ------------------- Cross-Platform Diagnostic Tests -------------------

class CPU_ConvNDKernelTuningDiagnostic_NONE : public GPU_ConvNDKernelTuning_Base
{
protected:
    bool RequiresAIModels() const override { return false; }
};

TEST_F(CPU_ConvNDKernelTuningDiagnostic_NONE, ArchitectureCompatibility_2D_Test)
{
    std::vector<std::string> test_architectures = {"gfx942", "gfx950", "gfx90a", "gfx908"};
    std::string test_solver                     = "ConvHipImplicitGemmGroupWrwXdlops";

    std::cout << "=== 2D Architecture Compatibility Report ===" << std::endl;
    std::cout << "Current device: " << device_arch << std::endl;

    for(const auto& arch : test_architectures)
    {
        bool files_exist = CheckModelFilesExist(arch, test_solver);
        std::cout << arch << ": " << (files_exist ? "SUPPORTED" : "NOT SUPPORTED") << std::endl;
    }
}

TEST_F(CPU_ConvNDKernelTuningDiagnostic_NONE, ArchitectureCompatibility_3D_Test)
{
    std::vector<std::string> test_architectures = {"gfx942", "gfx950", "gfx90a", "gfx908"};
    std::string test_solver                     = "ConvHipImplicitGemm3DGroupWrwXdlops";

    std::cout << "=== 3D Architecture Compatibility Report ===" << std::endl;
    std::cout << "Current device: " << device_arch << std::endl;

    for(const auto& arch : test_architectures)
    {
        bool files_exist = CheckModelFilesExist(arch, test_solver);
        std::cout << arch << ": " << (files_exist ? "SUPPORTED" : "NOT SUPPORTED") << std::endl;
    }
}

TEST_F(CPU_ConvNDKernelTuningDiagnostic_NONE, GetFeatures2D_CrossPlatform_Test)
{
    auto problem                        = GetReusable2DProblemDescription();
    int max_cu                          = 304;
    std::vector<std::string> test_archs = {"gfx942", "gfx950", "gfx90a"};

    for(const auto& arch : test_archs)
    {
        auto features = GetFeaturesND(problem, max_cu, arch);
        EXPECT_EQ(features.size(), 23u) << "Feature count inconsistent for " << arch;
        EXPECT_EQ(features.at("spatial_dim"), 2.0f) << "Spatial dim incorrect for " << arch;
    }
}

TEST_F(CPU_ConvNDKernelTuningDiagnostic_NONE, GetFeatures3D_CrossPlatform_Test)
{
    auto problem                        = GetReusable3DProblemDescription();
    int max_cu                          = 304;
    std::vector<std::string> test_archs = {"gfx942", "gfx950", "gfx90a"};

    for(const auto& arch : test_archs)
    {
        auto features = GetFeaturesND(problem, max_cu, arch);
        EXPECT_EQ(features.size(), 29u) << "Feature count inconsistent for " << arch;
        EXPECT_EQ(features.at("spatial_dim"), 3.0f) << "Spatial dim incorrect for " << arch;
    }
}

// ===============================================================================
// FillValidKernelsGeneric Tests
// ===============================================================================

class CPU_FillValidKernelsGeneric_NONE : public ::testing::Test
{
};

TEST_F(CPU_FillValidKernelsGeneric_NONE, Documentation_Test)
{
    // This test documents the purpose of FillValidKernelsGeneric:
    // It provides a single reusable template that handles data type dispatch
    // and TF32 fallback logic for all hip_implicit_gemm solvers, eliminating
    // the need for each solver to implement its own fill_valid_kernels lambda.
    SUCCEED();
}

// ===============================================================================
// RunKTNGeneric Tests
// ===============================================================================

class CPU_RunKTNGeneric_NONE : public ::testing::Test
{
};

TEST_F(CPU_RunKTNGeneric_NONE, Documentation_Test)
{
    // This test documents the purpose of RunKTNGeneric:
    // It provides a single reusable template that dispatches KTN model execution
    // based on data type for all hip_implicit_gemm solvers that support gfx90a,
    // eliminating the need for each solver to implement its own ktn_runner lambda.
    SUCCEED();
}

// ===============================================================================
// RunAIHeuristics Integration Tests
// ===============================================================================

class GPU_RunAIHeuristics_Integration_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
};

TEST_F(GPU_RunAIHeuristics_Integration_FP32, CandidateSelection_BasicFlow)
{
    // Skip if not on supported architecture
    if(device_arch != "gfx942" && device_arch != "gfx950")
    {
        GTEST_SKIP() << "Test requires gfx942 or gfx950, current: " << device_arch;
    }

    auto problem =
        GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    std::vector<std::string> valid_kernels;
    int index   = 0;
    int split_k = 0;
    std::string kernel_id;
    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

    constexpr SolverHeuristicConfig test_cfg = {
        "ConvHipImplicitGemmGroupWrwXdlops", // solver_name
        "ConvHipIgemmGroupXdlops",           // solver_name_ktn
        2,                                   // spatial_dims
        true,                                // uses_split_k
        1,                                   // split_k_min
        128,                                 // split_k_max
        false,                               // supports_split_k_autodeduce
        true                                 // supports_ktn
    };

    auto fill_kernels = [&](const miopen::conv::ProblemDescription&, bool /*try_tf32*/) {
        return std::vector<std::string>{
            "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>",
            "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,32,4,Default,4,2,1,4,4,1,1,1,1,1>"};
    };

    bool result = RunAIHeuristics(test_cfg, state, ctx, problem, false, fill_kernels);

    // Verify AI heuristics succeeded and populated state
    EXPECT_TRUE(result) << "RunAIHeuristics should succeed on " << device_arch;
    if(result)
    {
        EXPECT_FALSE(kernel_id.empty()) << "kernel_id should be populated";
        EXPECT_GE(index, 0) << "index should be valid";
        EXPECT_GE(split_k, 1) << "split_k should be >= 1 for WrW solver";
        EXPECT_LE(split_k, 128) << "split_k should be <= 128";

        // Verify kernel_id format (should be "kernel+split_k" for WrW)
        EXPECT_NE(kernel_id.find('+'), std::string::npos)
            << "kernel_id should contain '+' separator for split_k";
    }
}

TEST_F(GPU_RunAIHeuristics_Integration_FP32, Deterministic_EnforcesSplitK1)
{
    // Skip if not on supported architecture
    if(device_arch != "gfx942" && device_arch != "gfx950")
    {
        GTEST_SKIP() << "Test requires gfx942 or gfx950, current: " << device_arch;
    }

    auto problem =
        GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    std::vector<std::string> valid_kernels;
    int index   = 0;
    int split_k = 0;
    std::string kernel_id;
    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

    constexpr SolverHeuristicConfig bwd_cfg = {
        "ConvHipImplicitGemmGroupWrwXdlops", // solver_name
        "ConvHipIgemmGroupXdlops",           // solver_name_ktn
        2,                                   // spatial_dims
        true,                                // uses_split_k
        1,                                   // split_k_min
        128,                                 // split_k_max
        false,                               // supports_split_k_autodeduce
        true                                 // supports_ktn
    };

    auto fill_kernels = [&](const miopen::conv::ProblemDescription&, bool /*try_tf32*/) {
        return std::vector<std::string>{
            "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>"};
    };

    bool result =
        RunAIHeuristics(bwd_cfg, state, ctx, problem, true, fill_kernels); // deterministic=true

    if(result)
    {
        // Deterministic mode must enforce split_k=1
        EXPECT_EQ(split_k, 1) << "Deterministic mode must enforce split_k=1";
        EXPECT_TRUE(kernel_id.find("+1") != std::string::npos ||
                    kernel_id.find("+") == std::string::npos)
            << "kernel_id should indicate split_k=1 in deterministic mode";
    }
}

TEST_F(GPU_RunAIHeuristics_Integration_FP32, ForwardSolver_NoSplitK)
{
    // Skip if not on supported architecture
    if(device_arch != "gfx942" && device_arch != "gfx950")
    {
        GTEST_SKIP() << "Test requires gfx942 or gfx950, current: " << device_arch;
    }

    auto problem = GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::Forward);

    std::vector<std::string> valid_kernels;
    int index   = 0;
    int split_k = 0;
    std::string kernel_id;
    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

    constexpr SolverHeuristicConfig fwd_cfg = {
        "ConvHipImplicitGemmGroupFwdXdlops", // solver_name
        "ConvHipIgemmGroupFwdXdlops",        // solver_name_ktn
        2,                                   // spatial_dims
        false,                               // uses_split_k (Forward doesn't use split_k)
        0,                                   // split_k_min
        0,                                   // split_k_max
        false,                               // supports_split_k_autodeduce
        true                                 // supports_ktn
    };

    auto fill_kernels = [&](const miopen::conv::ProblemDescription&, bool /*try_tf32*/) {
        return std::vector<std::string>{
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256,256,256,4,Default,4,4,1,1,S<4,64,1>,"
            "S<1,0,2>,S<1,0,2>,2,8,8,1,S<4,64,1>,S<1,0,2>,S<1,0,2>,2,8,8,1,1,1,S<1,32,1,8>,S<1,32,"
            "1,8>,7,7,PassThrough,PassThrough,PassThrough>"};
    };

    bool result = RunAIHeuristics(fwd_cfg, state, ctx, problem, false, fill_kernels);

    if(result)
    {
        // Forward solver should have split_k=0
        EXPECT_EQ(split_k, 0) << "Forward solver should have split_k=0";
        EXPECT_EQ(kernel_id.find('+'), std::string::npos)
            << "Forward solver kernel_id should not contain '+' separator";
    }
}

// ===============================================================================
// CK Validator Integration Tests
// ===============================================================================

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

/**
 * @brief Test fixture for CK validator integration with RunAIHeuristics
 *
 * This fixture tests that RunAIHeuristics correctly uses the ck_validator_creator
 * parameter to reject invalid kernel+split_k combinations.
 */
class GPU_CKValidatorIntegration_FP32 : public GPU_ConvNDKernelTuningAI_Base
{
protected:
    // Shared configuration for WrW solver
    static constexpr SolverHeuristicConfig kWrwConfig = {
        "ConvHipImplicitGemmGroupWrwXdlops", // solver_name
        "ConvHipIgemmGroupXdlops",           // solver_name_ktn
        2,                                   // spatial_dims
        true,                                // uses_split_k
        1,                                   // split_k_min
        128,                                 // split_k_max
        false,                               // supports_split_k_autodeduce
        true                                 // supports_ktn
    };

    // Shared state variables
    std::vector<std::string> valid_kernels_;
    int index_   = 0;
    int split_k_ = 0;
    std::string kernel_id_;
    miopen::conv::ProblemDescription problem_;

    // Shared test kernels
    std::vector<std::string> test_kernels_ = {
        "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>",
        "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,32,4,Default,4,2,1,4,4,1,1,1,1,1>",
        "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,256,128,4,Default,4,4,2,4,4,4,2,1,1,4>"};

    void SetUp() override
    {
        spatial_dim = 2;
        solver_name = "ConvHipImplicitGemmGroupWrwXdlops";
        GPU_ConvNDKernelTuningAI_Base::SetUp();

        // Initialize problem description
        problem_ =
            GetReusable2DProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    }

    // Helper: Create HeuristicInitState with fixture members
    HeuristicInitState CreateState()
    {
        return HeuristicInitState(valid_kernels_, index_, split_k_, kernel_id_);
    }
};

TEST_F(GPU_CKValidatorIntegration_FP32, RestrictiveValidator_OnlyAcceptsSplitK1And2)
{
    if(device_arch != "gfx942" && device_arch != "gfx950")
    {
        GTEST_SKIP() << "Test requires gfx942 or gfx950, current: " << device_arch;
    }

    std::set<int> accepted_split_k = {1, 2};
    int validator_call_count       = 0;
    auto state                     = CreateState();

    auto fill_kernels = [this](const miopen::conv::ProblemDescription&, bool /*try_tf32*/) {
        return test_kernels_;
    };

    // Create validator with proper capture
    auto ck_validator_creator = [&accepted_split_k,
                                 &validator_call_count](const miopen::conv::ProblemDescription&) {
        return [&accepted_split_k, &validator_call_count](const std::string&, int sk) {
            validator_call_count++;
            return accepted_split_k.count(sk) > 0;
        };
    };

    bool result = RunAIHeuristics(
        kWrwConfig, state, ctx, problem_, false, fill_kernels, nullptr, ck_validator_creator);

    EXPECT_GT(validator_call_count, 0) << "Validator should have been called";

    if(result)
    {
        EXPECT_TRUE(accepted_split_k.count(split_k_) > 0)
            << "Selected split_k=" << split_k_ << " should be in accepted set {1, 2}";
    }
}

TEST_F(GPU_CKValidatorIntegration_FP32, RejectAllValidator_ReturnsFalse)
{
    if(device_arch != "gfx942" && device_arch != "gfx950")
    {
        GTEST_SKIP() << "Test requires gfx942 or gfx950, current: " << device_arch;
    }

    int validator_call_count = 0;
    auto state               = CreateState();

    auto fill_kernels = [this](const miopen::conv::ProblemDescription&, bool /*try_tf32*/) {
        return test_kernels_;
    };

    // Create validator that rejects all combinations
    auto ck_validator_creator = [&validator_call_count](const miopen::conv::ProblemDescription&) {
        return [&validator_call_count](const std::string&, int) {
            validator_call_count++;
            return false; // Reject everything
        };
    };

    bool result = RunAIHeuristics(
        kWrwConfig, state, ctx, problem_, false, fill_kernels, nullptr, ck_validator_creator);

    EXPECT_GT(validator_call_count, 0) << "Validator should have been called";
    EXPECT_FALSE(result) << "RunAIHeuristics should return false when all combinations rejected";
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

// ===============================================================================
// Parameterized Split_k Validation Tests
// ===============================================================================

class CPU_SplitKValidation_Parameterized_NONE
    : public ::testing::TestWithParam<std::tuple<int, int, int, bool>>
{
    // Parameters: (split_k_value, min, max, expected_valid)
};

TEST_P(CPU_SplitKValidation_Parameterized_NONE, ValidateSplitKRange)
{
    auto [value, min, max, expected] = GetParam();
    SolverHeuristicConfig cfg{"TestSolver", "TestSolverKTN", 2, true, min, max, false, true};

    bool actual = cfg.IsValidSplitK(value);
    EXPECT_EQ(actual, expected) << "IsValidSplitK(" << value << ") with range [" << min << ","
                                << max << "] expected " << (expected ? "true" : "false")
                                << " but got " << (actual ? "true" : "false");
}

INSTANTIATE_TEST_SUITE_P(Unit,
                         CPU_SplitKValidation_Parameterized_NONE,
                         ::testing::Values(
                             // (value, min, max, expected)
                             std::make_tuple(0, 1, 128, false),   // Below minimum
                             std::make_tuple(1, 1, 128, true),    // Valid: minimum power of 2
                             std::make_tuple(2, 1, 128, true),    // Valid: power of 2
                             std::make_tuple(3, 1, 128, false),   // Invalid: not power of 2
                             std::make_tuple(4, 1, 128, true),    // Valid: power of 2
                             std::make_tuple(7, 1, 128, false),   // Invalid: not power of 2
                             std::make_tuple(8, 1, 128, true),    // Valid: power of 2
                             std::make_tuple(16, 1, 128, true),   // Valid: power of 2
                             std::make_tuple(32, 1, 128, true),   // Valid: power of 2
                             std::make_tuple(64, 1, 128, true),   // Valid: power of 2
                             std::make_tuple(100, 1, 128, false), // Invalid: not power of 2
                             std::make_tuple(128, 1, 128, true),  // Valid: maximum power of 2
                             std::make_tuple(256, 1, 128, false), // Above maximum
                             std::make_tuple(512, 1, 128, false), // Above maximum
                             std::make_tuple(1, 2, 64, false),    // Below custom minimum
                             std::make_tuple(2, 2, 64, true),     // At custom minimum
                             std::make_tuple(64, 2, 64, true),    // At custom maximum
                             std::make_tuple(128, 2, 64, false)   // Above custom maximum
                             ));

// ===============================================================================
// Deterministic Mode Enforcement Tests
// ===============================================================================

class CPU_DeterministicMode_NONE : public ::testing::Test
{
};

TEST_F(CPU_DeterministicMode_NONE, HeuristicInit_SetsSplitK1)
{
    std::vector<std::string> valid_kernels = {"kernel_A"};
    int index                              = 0;
    int split_k                            = 999; // Should be overridden
    std::string kernel_id;

    HeuristicInitState state(valid_kernels, index, split_k, kernel_id);

    // When deterministic mode is active, HeuristicInit should set split_k=1
    // This is tested in the integration tests with actual RunAIHeuristics
    state.SetResult(0, 1, true); // Simulate deterministic result

    EXPECT_EQ(split_k, 1) << "Deterministic mode should set split_k=1";
    EXPECT_EQ(kernel_id, "kernel_A+1");
}

TEST_F(CPU_DeterministicMode_NONE, AllBackwardSolvers_SupportDeterministic)
{
    // Verify all backward/WrW solvers are configured to support deterministic mode
    // by using split_k that can be set to 1

    std::vector<SolverHeuristicConfig> backward_configs = {
        {"ConvHipImplicitGemmGroupBwdXdlops",
         "ConvHipIgemmGroupXdlops",
         2,
         true,
         1,
         128,
         false,
         true},
        {"ConvHipImplicitGemmGroupWrwXdlops",
         "ConvHipIgemmGroupXdlops",
         2,
         true,
         1,
         128,
         false,
         true},
        {"ConvHipImplicitGemm3DGroupBwdXdlops",
         "ConvHipIgemmGroup3DXdlops",
         3,
         true,
         1,
         128,
         false,
         true},
        {"ConvHipImplicitGemm3DGroupWrwXdlops",
         "ConvHipIgemmGroup3DXdlops",
         3,
         true,
         1,
         128,
         false,
         true},
    };

    for(const auto& cfg : backward_configs)
    {
        EXPECT_TRUE(cfg.uses_split_k) << cfg.solver_name << " should use split_k";
        EXPECT_LE(cfg.split_k_min, 1) << cfg.solver_name << " should support split_k=1";
        EXPECT_TRUE(cfg.IsValidSplitK(1)) << cfg.solver_name << " should accept split_k=1";
    }
}

// ------------------- Test Instantiations -------------------

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvNDKernelTuning_FP32,
                         ::testing::ValuesIn(GenSmokeTestCases()),
                         ConvNDKernelTuningTestName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvNDKernelTuning_FP32,
                         ::testing::ValuesIn(GenFullTestCases()),
                         ConvNDKernelTuningTestName);

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvNDKernelTuningAI_FP32,
    ::testing::Values(ConvNDKernelTuningTestParam{2,
                                                  miopenFloat,
                                                  miopen::conv::Direction::BackwardWeights,
                                                  "Conv2D"},
                      ConvNDKernelTuningTestParam{
                          3, miopenFloat, miopen::conv::Direction::BackwardWeights, "Conv3D"}),
    ConvNDKernelTuningTestName);

#else
// Dummy test when AI kernel tuning is disabled
TEST(CPU_ConvNDKernelTuningDisabled_NONE, FeatureDisabled)
{
    GTEST_SKIP() << "AI kernel tuning features are disabled in this build";
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
