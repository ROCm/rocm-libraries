// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU (no-Handle) coverage for the conv candidate-selection input path and the precompute
// selector, across every shipped arch -- including archs with no local hardware (e.g. gfx1250).
//
// The companion GPU_CandidateSelection_FP32 suite constructs a miopen::Handle in its fixture, so it
// only runs on a machine whose GPU matches the parameter arch; and its parameter list historically
// covered gfx942 only. Two gfx1250 model-artifact defects therefore stayed invisible to CI: a
// config-embedding table stamped with the wrong md5 (rejected at load), and a 2D input encoder
// whose width disagreed with its metadata (fdeep "Invalid inputs shape"). Both are pure CPU checks.
//
// These tests build no Handle and load each arch's models by name via GetSystemDbPath(), so a
// gfx942/gfx950 CI runner validates gfx1250's models without gfx1250 hardware. (The shared gtest
// main still requires some GPU to be present, so the suite runs on any GPU CI runner, not a
// GPU-less host.) They fail closed when the shipped assets are inconsistent, and drive the
// engineered input through GetFeaturesND on a real ProblemDescription so the live feature path
// (including the unconditional dilation_d) is exercised, not a hand-built map.

#include <gtest/gtest.h>
#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace miopen::ai::tuning::candidate_selection;

namespace {

struct ArchSolver
{
    std::string arch;
    std::string solver;
    bool is3d;
    bool uses_split_k;
};

// Every shipped (arch, solver). Combos whose assets are absent are skipped at run time, so adding a
// new arch here is harmless until its models ship. uses_split_k mirrors each solver's
// SolverHeuristicConfig (Fwd and 3D Bwd: no split_k; Wrw and 2D Bwd: split_k).
const std::vector<ArchSolver>& AllArchSolvers()
{
    static const std::vector<ArchSolver> v = [] {
        std::vector<ArchSolver> out;
        for(const auto& arch : {"gfx942", "gfx950", "gfx1250"})
        {
            out.push_back({arch, "ConvHipImplicitGemmGroupFwdXdlops", false, false});
            out.push_back({arch, "ConvHipImplicitGemmGroupBwdXdlops", false, true});
            out.push_back({arch, "ConvHipImplicitGemmGroupWrwXdlops", false, true});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupFwdXdlops", true, false});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupBwdXdlops", true, false});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupWrwXdlops", true, true});
        }
        return out;
    }();
    return v;
}

// A real ProblemDescription per spatial rank, so GetFeaturesND builds the engineered input exactly
// as production does (RunAIHeuristics feeds its output straight to EncodeInputFeatures). stride 1 /
// pad 1 / dilation 1 / 3x3(x3) keeps the output spatial size equal to the input.
miopen::conv::ProblemDescription MakeProblem(bool is3d)
{
    if(is3d)
    {
        miopen::TensorDescriptor in(miopenFloat, std::vector<int>{1, 64, 8, 56, 56});
        miopen::TensorDescriptor wei(miopenFloat, std::vector<int>{64, 64, 3, 3, 3});
        miopen::TensorDescriptor out(miopenFloat, std::vector<int>{1, 64, 8, 56, 56});
        miopen::ConvolutionDescriptor conv(
            3, miopenConvolution, miopenPaddingDefault, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0});
        return miopen::conv::ProblemDescription(
            in, wei, out, conv, miopen::conv::Direction::Forward);
    }
    miopen::TensorDescriptor in(miopenFloat, std::vector<int>{1, 64, 56, 56});
    miopen::TensorDescriptor wei(miopenFloat, std::vector<int>{64, 64, 3, 3});
    miopen::TensorDescriptor out(miopenFloat, std::vector<int>{1, 64, 56, 56});
    miopen::ConvolutionDescriptor conv(
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}, {0, 0});
    return miopen::conv::ProblemDescription(in, wei, out, conv, miopen::conv::Direction::Forward);
}

// The engineered feature map as production builds it (not a hand-coded map), so the dilation_d
// source change and any future GetFeaturesND change are exercised here.
std::map<std::string, float> FeaturesFor(const ArchSolver& as)
{
    return miopen::solver::conv::GetFeaturesND(MakeProblem(as.is3d), /*max_cu=*/304, as.arch);
}

// Split a vocab kernel id the way parse_kernel_id (ai_candidate_selection.cpp) does, and also
// report how many "+<value>" suffixes follow the final '>'. A well-formed id has at most one.
struct ParsedId
{
    std::string base;
    int split_k;
    int num_suffix;
};
ParsedId ParseVocabId(const std::string& kid)
{
    const auto gt   = kid.rfind('>');
    const auto plus = kid.find('+', gt == std::string::npos ? 0 : gt);
    if(plus == std::string::npos)
        return {kid, 1, 0};
    const std::string region = kid.substr(plus + 1);
    const int num_suffix     = 1 + static_cast<int>(std::count(region.begin(), region.end(), '+'));
    int split_k              = 1;
    try
    {
        split_k = std::stoi(region);
    }
    catch(...)
    {
        split_k = 1;
    }
    return {kid.substr(0, plus), split_k, num_suffix};
}

// Precompute assets: input encoder + metadata + the v3 config-embedding table (the config encoder
// is intentionally not shipped on this path).
bool AssetsPresent(const ArchSolver& as)
{
    const auto db = miopen::GetSystemDbPath();
    return miopen::fs::exists(db / (as.arch + "_" + as.solver + "_input_encoder.tn.model")) &&
           miopen::fs::exists(db / (as.arch + "_" + as.solver + "_metadata.tn.model")) &&
           miopen::fs::exists(db / (as.arch + "_" + as.solver + "_kernel_config_embeddings.bin"));
}

std::string NameOf(const ::testing::TestParamInfo<ArchSolver>& info)
{
    return info.param.arch + "_" + info.param.solver;
}

} // namespace

class CPU_CandidateSelectionCpu_NONE : public ::testing::TestWithParam<ArchSolver>
{
};

// The engineered input vector must be exactly the width the input encoder was trained for. A
// metadata/encoder disagreement (e.g. gfx1250 2D folding dilation_d into constants while the
// encoder keeps it active) makes fdeep throw "Invalid inputs shape" here -- caught on CPU.
TEST_P(CPU_CandidateSelectionCpu_NONE, EncodeInputFeaturesMatchesEncoderWidth)
{
    const auto as = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No candidate-selection assets for " << as.arch << "/" << as.solver;

    const auto& model = GetCandidateSelectionModel(as.arch, as.solver);
    std::vector<float> encoded;
    ASSERT_NO_THROW({ encoded = model.EncodeInputFeatures(FeaturesFor(as)); })
        << "EncodeInputFeatures threw for " << as.arch << "/" << as.solver
        << ": engineered input width disagrees with the input encoder.";
    EXPECT_FALSE(encoded.empty());
}

// End to end on the precompute path: with trivial applicability the selector must return a real
// pick for every shipped arch/solver. A nullopt here means the v3 table was rejected (md5 guard),
// the input shape mismatched, or the vocabulary was empty -- all silent non-AI fallbacks in prod.
TEST_P(CPU_CandidateSelectionCpu_NONE, PrecomputeSelectorReturnsPick)
{
    const auto as = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No candidate-selection assets for " << as.arch << "/" << as.solver;

    const auto accept_all = [](const std::string&, int, bool) { return true; };
    const auto pick       = SelectBestVocabCandidate(
        as.arch, as.solver, FeaturesFor(as), /*use_tf32=*/false, accept_all);
    ASSERT_TRUE(pick.has_value()) << "SelectBestVocabCandidate fell back for " << as.arch << "/"
                                  << as.solver
                                  << ": table rejected, input-shape mismatch, or empty vocabulary.";
    EXPECT_FALSE(pick->first.empty());
}

// Structural lint over the whole shipped vocabulary. accept_all above never parses split_k, so it
// cannot see a malformed kernel id: an id carrying two "+<split_k>" suffixes (as the gfx1250
// Wrw/Bwd tables did) makes parse_kernel_id read the first, fixed value and drop the ranked one.
// Assert each id has at most one suffix, and that the parsed split_k matches the solver -- none for
// a non-split-k solver; valid (or the -1 autodeduce sentinel, which stays present but masked) for a
// split-k solver.
TEST_P(CPU_CandidateSelectionCpu_NONE, VocabIdsWellFormed)
{
    const auto as = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No candidate-selection assets for " << as.arch << "/" << as.solver;

    const auto& vocab = GetConfigEmbeddingVocab(as.arch, as.solver);
    ASSERT_FALSE(vocab.empty()) << "Empty vocabulary for " << as.arch << "/" << as.solver;

    // min=1, max=128, no autodeduce match the shipped SolverHeuristicConfigs for these solvers.
    const miopen::solver::conv::SolverHeuristicConfig cfg{.uses_split_k = as.uses_split_k};

    for(const auto& kid : vocab)
    {
        const auto p = ParseVocabId(kid);
        EXPECT_LE(p.num_suffix, 1) << "Kernel id carries more than one split_k suffix: " << kid
                                   << " (" << as.arch << "/" << as.solver << ")";
        if(as.uses_split_k)
        {
            EXPECT_EQ(p.num_suffix, 1) << "split_k solver id is missing its suffix: " << kid;
            EXPECT_TRUE(cfg.IsValidSplitK(p.split_k) ||
                        p.split_k == miopen::solver::conv::CkSplitkAutoDeduce)
                << "Invalid split_k " << p.split_k << " in " << kid << " (" << as.arch << "/"
                << as.solver << ")";
        }
        else
        {
            EXPECT_EQ(p.num_suffix, 0) << "Non-split-k solver id carries a split_k suffix: " << kid;
            EXPECT_TRUE(cfg.IsValidSplitK(0));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_CandidateSelectionCpu_NONE,
                         ::testing::ValuesIn(AllArchSolvers()),
                         NameOf);

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
