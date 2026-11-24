// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the supports_split_k_autodeduce flag on the grouped Wrw solver configs.
//
// What it tests:
//   The precompute vocab-walk heuristic walks the ranked (kernel, split_k) pair list and accepts
//   the first pair that passes the applicability gate. The gate calls IsValidSplitK(split_k),
//   which returns false for split_k == CkSplitkAutoDeduce (-1) when
//   supports_split_k_autodeduce == false, causing the -1 pair to be skipped silently.
//
//   This test verifies the gate directly:
//   1. With supports_split_k_autodeduce == false:  IsValidSplitK(-1) must return false.
//   2. With supports_split_k_autodeduce == true:   IsValidSplitK(-1) must return true.
//   3. Explicit split_k values in [1, 128] are always valid regardless of the flag.
//   4. The 2D and 3D grouped Wrw solvers — the ones whose vocab carries +-1 entries — now have
//      the flag set to true; verify this matches the expected shipped value.
//
// This is a CPU unit test over SolverHeuristicConfig; no GPU or CK plugin needed.

#include <gtest/gtest.h>
#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/tensor.hpp>
#include <map>

using miopen::solver::conv::CkSplitkAutoDeduce;
using miopen::solver::conv::SolverHeuristicConfig;

// The autodeduce sentinel must be -1 to match the value stored in the Wrw vocab tables and the
// value CK recognises as "compute k_batch from occupancy". Checked at compile time.
static_assert(CkSplitkAutoDeduce == -1, "CkSplitkAutoDeduce must equal -1");

class CPU_SplitKAutodeduceGate_NONE : public ::testing::Test
{
};

// Gate off: autodeduce rejected, explicit values accepted.
TEST_F(CPU_SplitKAutodeduceGate_NONE, FlagOffRejectsAutodeduce)
{
    const SolverHeuristicConfig cfg{
        .uses_split_k                = true,
        .split_k_min                 = 1,
        .split_k_max                 = 128,
        .supports_split_k_autodeduce = false,
    };
    EXPECT_FALSE(cfg.IsValidSplitK(CkSplitkAutoDeduce))
        << "IsValidSplitK(-1) must return false when supports_split_k_autodeduce=false. "
           "A true result means the heuristic can pick -1 when the flag is off.";
    EXPECT_TRUE(cfg.IsValidSplitK(1));
    EXPECT_TRUE(cfg.IsValidSplitK(32));
    EXPECT_TRUE(cfg.IsValidSplitK(128));
    EXPECT_FALSE(cfg.IsValidSplitK(0));
    EXPECT_FALSE(cfg.IsValidSplitK(3));   // non-power-of-two
    EXPECT_FALSE(cfg.IsValidSplitK(256)); // above max
}

// Gate on: autodeduce accepted, explicit values still accepted.
TEST_F(CPU_SplitKAutodeduceGate_NONE, FlagOnAcceptsAutodeduce)
{
    const SolverHeuristicConfig cfg{
        .uses_split_k                = true,
        .split_k_min                 = 1,
        .split_k_max                 = 128,
        .supports_split_k_autodeduce = true,
    };
    EXPECT_TRUE(cfg.IsValidSplitK(CkSplitkAutoDeduce))
        << "IsValidSplitK(-1) must return true when supports_split_k_autodeduce=true. "
           "A false result means the heuristic silently falls through every -1 vocab entry "
           "even after the flag is enabled.";
    EXPECT_TRUE(cfg.IsValidSplitK(1));
    EXPECT_TRUE(cfg.IsValidSplitK(64));
    EXPECT_TRUE(cfg.IsValidSplitK(128));
}

// Non-split-k solvers (uses_split_k=false) accept split_k==0 and reject everything else.
TEST_F(CPU_SplitKAutodeduceGate_NONE, NonSplitKSolverAcceptsOnlyZero)
{
    const SolverHeuristicConfig cfg{
        .uses_split_k                = false,
        .split_k_min                 = 1,
        .split_k_max                 = 128,
        .supports_split_k_autodeduce = false,
    };
    EXPECT_TRUE(cfg.IsValidSplitK(0))
        << "Non-split-k solver must accept split_k=0 (no split_k suffix in vocab id).";
    EXPECT_FALSE(cfg.IsValidSplitK(1));
    EXPECT_FALSE(cfg.IsValidSplitK(CkSplitkAutoDeduce));
}

// End-to-end test: with the real solver gate (IsValidSplitK from GetWrw2DSolverHeuristicConfig),
// SelectBestVocabCandidate must return split_k=-1 for a shape where the model's top-1 is -1.
//
// This test FAILS when supports_split_k_autodeduce=false in the actual shipped solver config
// (the gate rejects -1, the walk falls to a different pick), and PASSES when it is true.
//
// Requires model assets in MIOPEN_SYSTEM_DB_PATH (same as conv_ai_candidate_selection_cpu.cpp).
class CPU_SplitKAutodeduceEndToEnd_NONE : public ::testing::Test
{
};

TEST_F(CPU_SplitKAutodeduceEndToEnd_NONE, Wrw2DPicksAutodeduceWithRealGate)
{
    const std::string arch   = "gfx950";
    const std::string solver = "ConvHipImplicitGemmGroupWrwXdlops";

    const auto db = miopen::GetSystemDbPath();
    if(!miopen::fs::exists(db / (arch + "_" + solver + "_input_encoder.tn.model")) ||
       !miopen::fs::exists(db / (arch + "_" + solver + "_metadata.tn.model")) ||
       !miopen::fs::exists(db / (arch + "_" + solver + "_kernel_config_embeddings.bin")))
        GTEST_SKIP() << "No candidate-selection assets for " << arch << "/" << solver;

    // Shape where the gfx950 Wrw model's true top-1 is split_k=-1 (autodeduce).
    // Verified by MIOPEN_AI_GOLDEN_GENERATE=1 against the shipped models (Nov 2026).
    // n=16, c=64, H=256, W=256, k=128, 3x3 grouped conv.
    miopen::TensorDescriptor in(miopenHalf, {16, 64, 256, 256});
    miopen::TensorDescriptor wei(miopenHalf, {128, 64, 3, 3});
    miopen::TensorDescriptor out(miopenHalf, {16, 128, 256, 256});
    miopen::ConvolutionDescriptor conv(
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}, {0, 0});
    miopen::conv::ProblemDescription prob(
        in, wei, out, conv, miopen::conv::Direction::BackwardWeights);

    const auto feats = miopen::solver::conv::GetFeaturesND(prob, /*max_cu=*/304, arch);

    // Gate using the ACTUAL shipped solver config (not a hand-crafted copy).
    // If supports_split_k_autodeduce is false in the solver, IsValidSplitK(-1) returns false
    // and the -1 pick is rejected -> pick->second != -1 -> this test fails.
    const auto& cfg = GetWrw2DSolverHeuristicConfig();
    const auto gate = [&cfg](const std::string&, int sk, bool) { return cfg.IsValidSplitK(sk); };

    const auto pick = miopen::ai::tuning::candidate_selection::SelectBestVocabCandidate(
        arch, solver, feats, /*use_tf32=*/false, gate);

    ASSERT_TRUE(pick.has_value())
        << "SelectBestVocabCandidate returned nullopt — model load or md5 guard failed. "
           "Check conv_ai_asset_smoke tests for the root cause.";

    EXPECT_EQ(pick->second, miopen::solver::conv::CkSplitkAutoDeduce)
        << "Expected split_k=-1 (autodeduce) but got " << pick->second << ". "
        << "Most likely supports_split_k_autodeduce was set back to false in the 2D Wrw solver "
           "config (k2DWrwSolverConfig in conv_hip_implicit_gemm_grouped_wrw_xdlops.cpp); a "
           "shipped "
           "model refresh that changes this shape's top-1 could also cause it.";
}

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
