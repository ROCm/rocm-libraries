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

// CPU (no-Handle) coverage for the conv candidate-selection input path and the precompute
// selector, across every shipped arch -- including archs with no local hardware (e.g. gfx1250).
//
// The companion GPU_CandidateSelection_FP32 suite constructs a miopen::Handle in its fixture, so it
// only runs on a machine whose GPU matches the parameter arch; and its parameter list historically
// covered gfx942 only. Two gfx1250 model-artifact defects therefore stayed invisible to CI: a
// config-embedding table stamped with the wrong md5 (rejected at load), and a 2D input encoder
// whose width disagreed with its metadata (fdeep "Invalid inputs shape"). Both are pure CPU checks.
//
// These tests load each arch's models by name via GetSystemDbPath() -- no Handle, no device -- so
// they exercise gfx1250 on any runner and fail closed when the shipped assets are inconsistent.

#include <gtest/gtest.h>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <functional>
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
};

// Every shipped (arch, solver). Combos whose assets are absent are skipped at run time, so adding a
// new arch here is harmless until its models ship.
const std::vector<ArchSolver>& AllArchSolvers()
{
    static const std::vector<ArchSolver> v = [] {
        std::vector<ArchSolver> out;
        for(const auto& arch : {"gfx942", "gfx950", "gfx1250"})
        {
            out.push_back({arch, "ConvHipImplicitGemmGroupFwdXdlops", false});
            out.push_back({arch, "ConvHipImplicitGemmGroupBwdXdlops", false});
            out.push_back({arch, "ConvHipImplicitGemmGroupWrwXdlops", false});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupFwdXdlops", true});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupBwdXdlops", true});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupWrwXdlops", true});
        }
        return out;
    }();
    return v;
}

// A feature map carrying every 2D and 3D key, so it is valid for any solver regardless of which
// params its metadata marks active (2D models ignore the depth keys; 3D models consume them). The
// categorical values are pre-encoded as EngineerCandidateSelectionInputFeatures expects: precision
// is the datatype enum, direction 0 = forward, layouts a valid one-hot index.
std::map<std::string, float> MakeCompleteFeatureMap(bool is3d)
{
    return {
        {"spatial_dim", is3d ? 3.0f : 2.0f},
        {"direction", 0.0f},
        {"batchsize", 2.0f},
        {"in_channels", 64.0f},
        {"out_channels", 64.0f},
        {"in_h", 56.0f},
        {"in_w", 56.0f},
        {"out_h", 56.0f},
        {"out_w", 56.0f},
        {"in_d", 8.0f},
        {"out_d", 8.0f},
        {"fil_h", 3.0f},
        {"fil_w", 3.0f},
        {"fil_d", 1.0f},
        {"pad_h", 1.0f},
        {"pad_w", 1.0f},
        {"pad_d", 0.0f},
        {"conv_stride_h", 1.0f},
        {"conv_stride_w", 1.0f},
        {"conv_stride_d", 1.0f},
        {"dilation_h", 1.0f},
        {"dilation_w", 1.0f},
        {"dilation_d", 1.0f},
        {"group_count", 1.0f},
        {"bias", 0.0f},
        {"in_layout", 1.0f},
        {"fil_layout", 1.0f},
        {"out_layout", 1.0f},
        {"precision", 0.0f},
    };
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
    ASSERT_NO_THROW({ encoded = model.EncodeInputFeatures(MakeCompleteFeatureMap(as.is3d)); })
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
        as.arch, as.solver, MakeCompleteFeatureMap(as.is3d), /*use_tf32=*/false, accept_all);
    ASSERT_TRUE(pick.has_value()) << "SelectBestVocabCandidate fell back for " << as.arch << "/"
                                  << as.solver
                                  << ": table rejected, input-shape mismatch, or empty vocabulary.";
    EXPECT_FALSE(pick->first.empty());
}

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_CandidateSelectionCpu_NONE,
                         ::testing::ValuesIn(AllArchSolvers()),
                         NameOf);
