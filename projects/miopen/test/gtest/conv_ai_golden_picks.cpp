// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Golden-output regression test: verifies that the deployed AI heuristic models predict the same
// config strings as a known-good reference run. The expected picks are recorded from an on-device
// run against the shipped model artifacts and must be updated any time the models are retrained.
//
// What this catches that the structural tests (conv_ai_candidate_selection_cpu.cpp,
// conv_ai_asset_smoke.cpp) cannot:
//   - A model retrained to predict different configs (intentional or accidental).
//   - Feature-encoding drift: a change to GetFeaturesND silently changes what the model is asked.
//   - Wrong model loaded for an arch (e.g. gfx942 model loaded for gfx950): inference succeeds but
//     predicts garbage configs.
//   - fdeep/keras version mismatch that makes the model predict different (not throw): the
//     structural tests only check "no throw"; this checks the actual output.
//
// Design:
//   - CPU-only (no GPU execution). SelectBestVocabCandidate + GetFeaturesND make no GPU calls.
//   - Parameterized over (arch, solver, shape) triples covering gfx942, gfx950, and gfx1250.
//   - Shapes are small to keep CPU inference fast.
//   - Uses accept_all (no CK applicability filter) — tests pure model ranking. Golden values MUST
//     be captured via the same accept_all path using MIOPEN_AI_GOLDEN_GENERATE=1 (see below), NOT
//     via MIOpenDriver which applies a real CK applicability walk and may return a lower-ranked
//     pick if the model's #1 is CK-inapplicable on the shape.
//   - Update: MIOPEN_AI_GOLDEN_GENERATE=1 ./test_conv_ai_golden_picks, copy printed lines into
//     kGoldenCases. The gfx942/gfx950 picks in this file were initially captured via MIOpenDriver
//     (provisional) and should be regenerated via MIOPEN_AI_GOLDEN_GENERATE=1 on the first build.
//   - A pick mismatch is a hard failure: the model is not producing the expected output. Either
//     the models were updated (expected: regenerate) or something regressed unexpectedly.

#include <gtest/gtest.h>
#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/tensor.hpp>
#include <string>
#include <vector>

using namespace miopen::ai::tuning::candidate_selection;

namespace {

struct GoldenCase
{
    const char* arch;
    const char* solver;
    int n, c, h, w, k, fy, fx, pad, stride, groups;
    bool is3d;
    int d, fz, dpad, dstride;
    miopenDataType_t dtype;
    miopen::conv::Direction dir;
    const char* expected_pick; // "SKIP" = not yet captured; otherwise must match exactly
};

// clang-format off
// Expected picks captured via MIOPEN_AI_GOLDEN_GENERATE=1 (accept_all path, Nov 2026).
// All 24 cases including gfx1250 and 3D verified on gfx950 hardware.
// Regenerate: MIOPEN_AI_GOLDEN_GENERATE=1 MIOPEN_SYSTEM_DB_PATH=<kdb> ./test_conv_ai_golden_picks
static const GoldenCase kGoldenCases[] = {
    // gfx942 Fwd
    {"gfx942","ConvHipImplicitGemmGroupFwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 64, Filter1x1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>"},
    {"gfx942","ConvHipImplicitGemmGroupFwdXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<64, 16, 16, 128, Filter1x1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1>"},
    {"gfx942","ConvHipImplicitGemmGroupFwdXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Stride1Pad0, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>"},
    // gfx942 Bwd
    {"gfx942","ConvHipImplicitGemmGroupBwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 1, 8, 1, 1, 1>"},
    {"gfx942","ConvHipImplicitGemmGroupBwdXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 32, 64, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 1, 2, 8, 4, 1, 1>"},
    {"gfx942","ConvHipImplicitGemmGroupBwdXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Filter1x1Stride1Pad0, 32, 32, 1, 1, 8, 2, 1, 1>"},
    // gfx942 Wrw
    {"gfx942","ConvHipImplicitGemmGroupWrwXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>"},
    {"gfx942","ConvHipImplicitGemmGroupWrwXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>"},
    {"gfx942","ConvHipImplicitGemmGroupWrwXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 1, 4, 1, 4, 1, 1, 1>"},

    // gfx950 Fwd
    {"gfx950","ConvHipImplicitGemmGroupFwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>"},
    {"gfx950","ConvHipImplicitGemmGroupFwdXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<128, 16, 32, 64, Filter1x1Stride1Pad0, 16, 16, 1, 1, 4, 4, 4, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>"},
    {"gfx950","ConvHipImplicitGemmGroupFwdXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Filter1x1Stride1Pad0, 32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>"},
    // gfx950 Bwd
    {"gfx950","ConvHipImplicitGemmGroupBwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 1, 1, 8, 1, 1, 1>"},
    {"gfx950","ConvHipImplicitGemmGroupBwdXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 64, 16, 16, Filter1x1Stride1Pad0, 16, 16, 1, 1, 16, 1, 1, 1>"},
    {"gfx950","ConvHipImplicitGemmGroupBwdXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3_DirectLoad<256, 256, 32, 64, 8, 8, Filter1x1Stride1Pad0, 32, 32, 2, 1, 8, 8, 1, 1>"},
    // gfx950 Wrw
    {"gfx950","ConvHipImplicitGemmGroupWrwXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>"},
    {"gfx950","ConvHipImplicitGemmGroupWrwXdlops",1,128,28,28,128,1,1,0,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>"},
    {"gfx950","ConvHipImplicitGemmGroupWrwXdlops",4,32,112,112,32,3,3,1,1,2,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 4, 4, 1, 4, 1, 1, 1>"},

    // gfx1250 2D
    {"gfx1250","ConvHipImplicitGemmGroupFwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_WmmaPorted<128, 64, 16, 64, Filter1x1Pad0, 16, 16, 2, 1, 4, 4, 2, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v2>"},
    {"gfx1250","ConvHipImplicitGemmGroupBwdXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<32, 16, 32, 64, Filter1x1Stride1Pad0, 8, 8, 8>"},
    {"gfx1250","ConvHipImplicitGemmGroupWrwXdlops",1,64,56,56,64,3,3,1,1,1,false,0,0,0,0,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3_WmmaPorted<512, 64, 64, 512, Filter1x1Stride1Pad0, 16, 1, 1, 8, 8, 8, 8, 1, 1, 4>"},

    // 3D gfx950
    {"gfx950","ConvHipImplicitGemm3DGroupFwdXdlops",1,32,56,56,32,3,3,1,1,1,true,8,3,1,1,miopenHalf,miopen::conv::Direction::Forward,
     "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 64, 64, 64, Default, 32, 32, 1, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>"},
    {"gfx950","ConvHipImplicitGemm3DGroupBwdXdlops",1,32,56,56,32,3,3,1,1,1,true,8,3,1,1,miopenHalf,miopen::conv::Direction::BackwardData,
     "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 64, 16, 64, 16, 16, Filter1x1Stride1Pad0, 16, 16, 1, 1, 16, 8, 1, 1>"},
    {"gfx950","ConvHipImplicitGemm3DGroupWrwXdlops",1,32,56,56,32,3,3,1,1,1,true,8,3,1,1,miopenHalf,miopen::conv::Direction::BackwardWeights,
     "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256, 64, 64, 8, Filter1x1Stride1Pad0, 8, 1, 1, 1, 4, 4, 4, 1, 1, 4>"},
};
// clang-format on

bool AssetsPresent(const std::string& arch, const std::string& solver)
{
    const auto db = miopen::GetSystemDbPath();
    return miopen::fs::exists(db / (arch + "_" + solver + "_input_encoder.tn.model")) &&
           miopen::fs::exists(db / (arch + "_" + solver + "_metadata.tn.model")) &&
           miopen::fs::exists(db / (arch + "_" + solver + "_kernel_config_embeddings.bin"));
}

miopen::conv::ProblemDescription MakeProblem(const GoldenCase& tc)
{
    if(tc.is3d)
    {
        miopen::TensorDescriptor in(tc.dtype, {tc.n, tc.c, tc.d, tc.h, tc.w});
        miopen::TensorDescriptor wei(tc.dtype, {tc.k, tc.c / tc.groups, tc.fz, tc.fy, tc.fx});
        int od = (tc.d + 2 * tc.dpad - tc.fz) / tc.dstride + 1;
        int oh = (tc.h + 2 * tc.pad - tc.fy) / tc.stride + 1;
        int ow = (tc.w + 2 * tc.pad - tc.fx) / tc.stride + 1;
        miopen::TensorDescriptor out(tc.dtype, {tc.n, tc.k, od, oh, ow});
        miopen::ConvolutionDescriptor conv(3,
                                           miopenConvolution,
                                           miopenPaddingDefault,
                                           {tc.dpad, tc.pad, tc.pad},
                                           {tc.dstride, tc.stride, tc.stride},
                                           {1, 1, 1},
                                           {0, 0, 0});
        return {in, wei, out, conv, tc.dir};
    }
    miopen::TensorDescriptor in(tc.dtype, {tc.n, tc.c, tc.h, tc.w});
    miopen::TensorDescriptor wei(tc.dtype, {tc.k, tc.c / tc.groups, tc.fy, tc.fx});
    int oh = (tc.h + 2 * tc.pad - tc.fy) / tc.stride + 1;
    int ow = (tc.w + 2 * tc.pad - tc.fx) / tc.stride + 1;
    miopen::TensorDescriptor out(tc.dtype, {tc.n, tc.k, oh, ow});
    miopen::ConvolutionDescriptor conv(2,
                                       miopenConvolution,
                                       miopenPaddingDefault,
                                       {tc.pad, tc.pad},
                                       {tc.stride, tc.stride},
                                       {1, 1},
                                       {0, 0});
    return {in, wei, out, conv, tc.dir};
}

std::string CaseName(const ::testing::TestParamInfo<GoldenCase>& info)
{
    const auto& tc = info.param;
    std::string s  = std::string(tc.arch) + "_" + tc.solver + "_n" + std::to_string(tc.n) + "_c" +
                    std::to_string(tc.c) + "_h" + std::to_string(tc.h) + "_k" +
                    std::to_string(tc.k) + "_g" + std::to_string(tc.groups);
    for(char& c : s)
        if(c == ' ' || c == '-')
            c = '_';
    return s;
}

} // namespace

class CPU_ConvAiGoldenPicks_NONE : public ::testing::TestWithParam<GoldenCase>
{
};

// When MIOPEN_AI_GOLDEN_GENERATE=1 the test prints actual picks as C++ initializer lines instead
// of asserting. Use this to regenerate kGoldenCases after a model retrain:
//   MIOPEN_AI_GOLDEN_GENERATE=1 ./test_conv_ai_golden_picks --gtest_also_run_disabled_tests
// Copy the printed lines into the kGoldenCases table above, replacing the old expected_pick
// strings. The generation uses the same accept_all path as the assertion, so the values are
// self-consistent: they reflect pure model ranking with no applicability filter.
static bool GenerateMode()
{
    const char* v = std::getenv("MIOPEN_AI_GOLDEN_GENERATE");
    return v && std::string(v) == "1";
}

TEST_P(CPU_ConvAiGoldenPicks_NONE, PickMatchesGolden)
{
    const auto& tc = GetParam();

    if(!AssetsPresent(tc.arch, tc.solver))
        GTEST_SKIP() << "No assets for " << tc.arch << "/" << tc.solver;

    const auto prob  = MakeProblem(tc);
    const auto feats = miopen::solver::conv::GetFeaturesND(prob, /*max_cu=*/304, tc.arch);

    // accept_all: tests the pure model ranking with no CK applicability filter.
    // Golden values must be captured with the same accept_all path (via MIOPEN_AI_GOLDEN_GENERATE),
    // NOT via MIOpenDriver which applies a real CK applicability walk and may return a different
    // (lower-ranked) pick if the model's #1 is CK-inapplicable on the shape.
    const auto accept_all = [](const std::string&, int, bool) { return true; };
    const auto pick       = SelectBestVocabCandidate(tc.arch, tc.solver, feats, false, accept_all);

    ASSERT_TRUE(pick.has_value())
        << tc.arch << "/" << tc.solver
        << ": SelectBestVocabCandidate returned nullopt — model load or md5 guard failed. "
           "Check conv_ai_asset_smoke tests for the root cause.";

    if(GenerateMode())
    {
        // Print the C++ initializer line; redirect stdout to capture.
        std::cout << "    {\"" << tc.arch << "\",\"" << tc.solver << "\"," << tc.n << "," << tc.c
                  << "," << tc.h << "," << tc.w << "," << tc.k << "," << tc.fy << "," << tc.fx
                  << "," << tc.pad << "," << tc.stride << "," << tc.groups << ","
                  << (tc.is3d ? "true" : "false") << "," << tc.d << "," << tc.fz << "," << tc.dpad
                  << "," << tc.dstride << ","
                  << (tc.dtype == miopenHalf ? "miopenHalf" : "miopenFloat") << ","
                  << (tc.dir == miopen::conv::Direction::Forward
                          ? "miopen::conv::Direction::Forward"
                      : tc.dir == miopen::conv::Direction::BackwardData
                          ? "miopen::conv::Direction::BackwardData"
                          : "miopen::conv::Direction::BackwardWeights")
                  << ",\n     \"" << pick->first << "\"},\n";
        GTEST_SKIP() << "Generate mode: printed pick for " << tc.arch << "/" << tc.solver;
    }

    if(std::string(tc.expected_pick) == "SKIP")
        GTEST_SKIP() << "Golden pick not yet captured for " << tc.arch << "/" << tc.solver
                     << " — set MIOPEN_AI_GOLDEN_GENERATE=1 and run this test to populate.";

    EXPECT_EQ(pick->first, std::string(tc.expected_pick))
        << tc.arch << "/" << tc.solver
        << ": model predicted a different config than the golden reference.\n"
        << "  expected: " << tc.expected_pick << "\n"
        << "  actual:   " << pick->first << "\n"
        << "If the models were retrained, regenerate with: "
           "MIOPEN_AI_GOLDEN_GENERATE=1 ./test_conv_ai_golden_picks";
}

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_ConvAiGoldenPicks_NONE,
                         ::testing::ValuesIn(kGoldenCases),
                         CaseName);

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
