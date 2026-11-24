// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU (no-Handle) coverage for the conv candidate-selection path across every shipped arch,
// including archs with no local hardware (gfx1250): the GPU_CandidateSelection_FP32 suite needs a
// matching GPU, so a gfx942/gfx950 runner here validates gfx1250's models by loading them by name
// via GetSystemDbPath(). Catches model-artifact defects on CPU (wrong-md5 table, encoder/metadata
// width mismatch). The shared gtest main still needs some GPU present, so this runs on any GPU
// runner, not a GPU-less host.

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

// The problem's direction must match the solver's: Wrw/Bwd models were trained on
// BackwardWeights/BackwardData shapes, and GetFeaturesND encodes the direction, so a Forward
// problem would not exercise the Bwd/Wrw metadata the target workload uses.
miopen::conv::Direction DirectionOf(const std::string& solver)
{
    if(solver.find("Wrw") != std::string::npos)
        return miopen::conv::Direction::BackwardWeights;
    if(solver.find("Bwd") != std::string::npos)
        return miopen::conv::Direction::BackwardData;
    return miopen::conv::Direction::Forward;
}

// Float precisions every shipped input-encoder metadata one-hots (BF16/FP16/FP32). The target
// workload is BF16; a precision the encoder does not cover would throw in EncodeInputFeatures, so
// covering all three here catches a dtype regression in the metadata. INT8 (Fwd only) uses a
// separate path and is out of scope.
const std::vector<miopenDataType_t>& TestDataTypes()
{
    static const std::vector<miopenDataType_t> v{miopenFloat, miopenHalf, miopenBFloat16};
    return v;
}

// A real ProblemDescription per spatial rank, dtype, direction and dilation, so GetFeaturesND
// builds the engineered input exactly as production does (RunAIHeuristics feeds its output straight
// to EncodeInputFeatures). stride 1 and pad == dilation on a 3-tap filter keep the output spatial
// size equal to the input, so a dilation > 1 case stays valid and exercises the dilation feature.
miopen::conv::ProblemDescription
MakeProblem(bool is3d, miopenDataType_t dtype, miopen::conv::Direction dir, int dilation)
{
    const int p = dilation;
    if(is3d)
    {
        miopen::TensorDescriptor in(dtype, std::vector<int>{1, 64, 8, 56, 56});
        miopen::TensorDescriptor wei(dtype, std::vector<int>{64, 64, 3, 3, 3});
        miopen::TensorDescriptor out(dtype, std::vector<int>{1, 64, 8, 56, 56});
        miopen::ConvolutionDescriptor conv(3,
                                           miopenConvolution,
                                           miopenPaddingDefault,
                                           {p, p, p},
                                           {1, 1, 1},
                                           {dilation, dilation, dilation},
                                           {0, 0, 0});
        return miopen::conv::ProblemDescription(in, wei, out, conv, dir);
    }
    miopen::TensorDescriptor in(dtype, std::vector<int>{1, 64, 56, 56});
    miopen::TensorDescriptor wei(dtype, std::vector<int>{64, 64, 3, 3});
    miopen::TensorDescriptor out(dtype, std::vector<int>{1, 64, 56, 56});
    miopen::ConvolutionDescriptor conv(
        2, miopenConvolution, miopenPaddingDefault, {p, p}, {1, 1}, {dilation, dilation}, {0, 0});
    return miopen::conv::ProblemDescription(in, wei, out, conv, dir);
}

// The engineered feature map as production builds it (not a hand-coded map) for one dtype/dilation,
// so the dilation_d source change and any future GetFeaturesND change are exercised here.
std::map<std::string, float> FeaturesFor(const ArchSolver& as, miopenDataType_t dtype, int dilation)
{
    return miopen::solver::conv::GetFeaturesND(
        MakeProblem(as.is3d, dtype, DirectionOf(as.solver), dilation), /*max_cu=*/304, as.arch);
}

// Mirror parse_kernel_id (ai_candidate_selection.cpp); that lambda is file-local, so keep this copy
// in sync with it. Also report how many "+<value>" suffixes follow the final '>' (well-formed:
// <=1).
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
    for(const auto dtype : TestDataTypes())
    {
        for(const int dilation : {1, 2})
        {
            std::vector<float> encoded;
            ASSERT_NO_THROW({
                encoded = model.EncodeInputFeatures(FeaturesFor(as, dtype, dilation));
            }) << "EncodeInputFeatures threw for "
               << as.arch << "/" << as.solver << " dtype " << dtype << " dilation " << dilation
               << ": engineered input width disagrees with the input encoder.";
            EXPECT_FALSE(encoded.empty());
        }
    }
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
    for(const auto dtype : TestDataTypes())
    {
        for(const int dilation : {1, 2})
        {
            const auto pick = SelectBestVocabCandidate(as.arch,
                                                       as.solver,
                                                       FeaturesFor(as, dtype, dilation),
                                                       /*use_tf32=*/false,
                                                       accept_all);
            ASSERT_TRUE(pick.has_value())
                << "SelectBestVocabCandidate fell back for " << as.arch << "/" << as.solver
                << " dtype " << dtype << " dilation " << dilation
                << ": table rejected, input-shape mismatch, or empty vocabulary.";
            EXPECT_FALSE(pick->first.empty());
        }
    }
}

// Structural lint over the shipped vocabulary: accept_all above never parses split_k, so it misses
// a malformed id. Assert each id has at most one "+<split_k>" suffix (the gfx1250 Wrw/Bwd tables
// had two), and that split_k matches the solver -- none for non-split-k, valid or the -1 autodeduce
// sentinel for split-k.
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

INSTANTIATE_TEST_SUITE_P(Standard,
                         CPU_CandidateSelectionCpu_NONE,
                         ::testing::ValuesIn(AllArchSolvers()),
                         NameOf);

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
