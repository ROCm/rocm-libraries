/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// CPU-only unit tests for the cross-architecture LGBM solver picker.
// These do not require a GPU: the metadata loader, GPU lookup table, and the
// PickSolverForSpec entry point are all host-side. The fixture replay validates
// the inference path against lgbm_test_vectors.json (the reference
// problem/expected-solver pairs shipped from AutoResearchAllLGBM).
//
// Build: make test_lgbm_picker
// Run:   ./bin/test_lgbm_picker

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_gpu_features.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/tensor.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

using miopen::ai::lgbm::kNumSpecIds;
using miopen::ai::lgbm::kSpecIdNames;
using miopen::ai::lgbm::LgbmMetadata;
using miopen::ai::lgbm::ResolveSpecId;

// ---------------------------------------------------------------------------
// Metadata loader
// ---------------------------------------------------------------------------

class CPU_LgbmMetadata : public ::testing::Test
{
protected:
    const LgbmMetadata& meta = LgbmMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "lgbm_model_meta.json not found in "
                         << miopen::GetSystemDbPath().string();
    }
};

TEST_F(CPU_LgbmMetadata, LoadsSolverVocab)
{
    // v5 ships 52 solver names.
    EXPECT_EQ(meta.Solvers().size(), 52u);
}

TEST_F(CPU_LgbmMetadata, TripleVocabPopulated)
{
    EXPECT_FALSE(meta.TripleVocab().empty());
    // Every solver named in any triple bucket must be a known solver code.
    for(const auto& [key, cands] : meta.TripleVocab())
    {
        EXPECT_FALSE(cands.empty()) << "empty bucket for " << key;
        for(const auto& s : cands)
            EXPECT_GE(meta.SolverCode(s), 0) << "unknown solver " << s << " in bucket " << key;
    }
}

TEST_F(CPU_LgbmMetadata, CategoricalCodesResolve)
{
    // Direction vocab is {"1","2","4"}; data_type includes the four dtypes.
    EXPECT_GE(meta.CategoricalCode("direction", "1"), 0);
    EXPECT_GE(meta.CategoricalCode("direction", "2"), 0);
    EXPECT_GE(meta.CategoricalCode("direction", "4"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "fp16"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "fp32"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "bf16"), 0);

    // Unknown values / columns map to the missing sentinel (-1).
    EXPECT_EQ(meta.CategoricalCode("data_type", "not_a_dtype"), -1);
    EXPECT_EQ(meta.CategoricalCode("not_a_column", "x"), -1);
    EXPECT_EQ(meta.SolverCode("NotARealSolver"), -1);
}

TEST_F(CPU_LgbmMetadata, SolverCodeMatchesVocabIndex)
{
    const auto& solvers = meta.Solvers();
    for(int i = 0; i < static_cast<int>(solvers.size()); ++i)
        EXPECT_EQ(meta.SolverCode(solvers[i]), i);
}

// ---------------------------------------------------------------------------
// GPU lookup table / spec_id resolution
// ---------------------------------------------------------------------------

TEST(CPU_LgbmGpuTable, SpecIdCodeMatchesArrayIndex)
{
    // kGpuTable[i].spec_id_code must equal i so the index doubles as the
    // categorical code fed to the model.
    for(int i = 0; i < static_cast<int>(kNumSpecIds); ++i)
        EXPECT_EQ(miopen::ai::lgbm::kGpuTable[i].spec_id_code, i)
            << "spec " << kSpecIdNames[i];
}

TEST(CPU_LgbmGpuTable, ResolvesKnownSkus)
{
    // gfx942 disambiguates by CU count then VRAM.
    constexpr std::size_t gb = 1024ULL * 1024 * 1024;
    const int mi300a = ResolveSpecId("gfx942", 228, 64 * gb);
    const int mi300x = ResolveSpecId("gfx942", 304, 192 * gb);
    const int mi325x = ResolveSpecId("gfx942", 304, 256 * gb);
    EXPECT_GE(mi300a, 0);
    EXPECT_GE(mi300x, 0);
    EXPECT_GE(mi325x, 0);
    EXPECT_EQ(std::string(kSpecIdNames[mi300a]), "gfx942-mi300a");
    EXPECT_EQ(std::string(kSpecIdNames[mi300x]), "gfx942-mi300x");
    EXPECT_EQ(std::string(kSpecIdNames[mi325x]), "gfx942-mi325x");

    // gfx950 collapses to mi355x (mi350x/mi355x are HIP-indistinguishable).
    const int gfx950 = ResolveSpecId("gfx950", 256, 309ULL * gb);
    EXPECT_GE(gfx950, 0);
    EXPECT_EQ(std::string(kSpecIdNames[gfx950]), "gfx950-mi355x");

    // Single-SKU archs resolve regardless of cu/vram.
    EXPECT_GE(ResolveSpecId("gfx90a", 104, 64 * gb), 0);
    EXPECT_GE(ResolveSpecId("gfx1201", 64, 16 * gb), 0);
}

TEST(CPU_LgbmGpuTable, AbstainsOnUnknownArch)
{
    constexpr std::size_t gb = 1024ULL * 1024 * 1024;
    // Consumer / untrained archs are not in the spec_id vocab.
    EXPECT_LT(ResolveSpecId("gfx1100", 96, 24 * gb), 0);
    EXPECT_LT(ResolveSpecId("gfx908", 120, 32 * gb), 0);
    EXPECT_LT(ResolveSpecId("gfx942", 999, 64 * gb), 0); // bogus CU count
}

// ---------------------------------------------------------------------------
// Fixture replay
// ---------------------------------------------------------------------------

class CPU_LgbmPickerFixture : public ::testing::Test
{
protected:
    nlohmann::json fixture;
    const LgbmMetadata& meta = LgbmMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "LGBM metadata unavailable; picker disabled in this build";

        const auto path = miopen::GetSystemDbPath() / "lgbm_test_vectors.json";
        if(!miopen::fs::exists(path))
            GTEST_SKIP() << "fixture not found: " << path.string();
        std::ifstream ifs(path.string());
        ifs >> fixture;
    }

    static int SpecIdCode(const std::string& spec_id)
    {
        for(int i = 0; i < static_cast<int>(kNumSpecIds); ++i)
            if(std::string(kSpecIdNames[i]) == spec_id)
                return i;
        return -1;
    }

    static double Num(const nlohmann::json& j, const char* key, double fallback)
    {
        if(!j.contains(key) || j.at(key).is_null())
            return fallback;
        const auto& v = j.at(key);
        if(!v.is_number())
            return fallback;
        const double d = v.get<double>();
        return std::isnan(d) ? fallback : d;
    }

    static miopenDataType_t DataType(const std::string& dt)
    {
        if(dt == "fp16")
            return miopenHalf;
        if(dt == "bf16")
            return miopenBFloat16;
        if(dt == "int8")
            return miopenInt8;
        return miopenFloat;
    }

    // Reconstruct a conv::ProblemDescription from a fixture problem_inputs
    // block. The fixture leaves layout/derived fields NaN for some rows; the
    // picker derives those from the tensors regardless, so a concrete layout
    // is always assigned here (matching what MIOpen produces at runtime).
    static miopen::conv::ProblemDescription BuildProblem(const nlohmann::json& p)
    {
        const int sd = static_cast<int>(Num(p, "spatial_dim", 2));
        const int n  = static_cast<int>(Num(p, "n_mini_batch_size", 1));
        const int c  = static_cast<int>(Num(p, "channels", 1));
        const int k  = static_cast<int>(Num(p, "output_channels", 1));
        const int gc = std::max(1, static_cast<int>(Num(p, "groups", 1)));
        const auto dt = DataType(p.value("data_type", "fp32"));

        const int dir_code = static_cast<int>(Num(p, "direction", 1));
        const auto dir     = dir_code == 2   ? miopen::conv::Direction::BackwardData
                             : dir_code == 4 ? miopen::conv::Direction::BackwardWeights
                                             : miopen::conv::Direction::Forward;

        const int h  = static_cast<int>(Num(p, "height", 1));
        const int w  = static_cast<int>(Num(p, "width", 1));
        const int ky = static_cast<int>(Num(p, "filter_height_y", 1));
        const int kx = static_cast<int>(Num(p, "filter_width_x", 1));
        const int ph = static_cast<int>(Num(p, "pad_height", 0));
        const int pw = static_cast<int>(Num(p, "pad_width", 0));
        const int sh = std::max(1, static_cast<int>(Num(p, "stride_height", 1)));
        const int sw = std::max(1, static_cast<int>(Num(p, "stride_width", 1)));
        const int dh = std::max(1, static_cast<int>(Num(p, "dilation_height", 1)));
        const int dw = std::max(1, static_cast<int>(Num(p, "dilation_width", 1)));

        auto conv_out = [](int in, int pad, int dil, int fil, int str) {
            return (in + 2 * pad - dil * (fil - 1) - 1) / str + 1;
        };

        if(sd == 3)
        {
            const int d  = static_cast<int>(Num(p, "depth", 1));
            const int kz = static_cast<int>(Num(p, "filter_depth_z", 1));
            const int pd = static_cast<int>(Num(p, "pad_depth", 0));
            const int sdp = std::max(1, static_cast<int>(Num(p, "stride_depth", 1)));
            const int dd = std::max(1, static_cast<int>(Num(p, "dilation_depth", 1)));
            const int od = conv_out(d, pd, dd, kz, sdp);
            const int oh = conv_out(h, ph, dh, ky, sh);
            const int ow = conv_out(w, pw, dw, kx, sw);

            miopen::TensorDescriptor in(dt, {n, c, d, h, w});
            miopen::TensorDescriptor wei(dt, {k, c / gc, kz, ky, kx});
            miopen::TensorDescriptor out(dt, {n, k, od, oh, ow});
            miopen::ConvolutionDescriptor conv(3,
                                               miopenConvolution,
                                               miopenPaddingDefault,
                                               {pd, ph, pw},
                                               {sdp, sh, sw},
                                               {dd, dh, dw},
                                               {0, 0, 0},
                                               gc,
                                               1.0f);
            const bool fwd = dir == miopen::conv::Direction::Forward;
            return miopen::conv::ProblemDescription(
                fwd ? in : out, wei, fwd ? out : in, conv, dir);
        }

        const int oh = conv_out(h, ph, dh, ky, sh);
        const int ow = conv_out(w, pw, dw, kx, sw);
        miopen::TensorDescriptor in(dt, {n, c, h, w});
        miopen::TensorDescriptor wei(dt, {k, c / gc, ky, kx});
        miopen::TensorDescriptor out(dt, {n, k, oh, ow});
        miopen::ConvolutionDescriptor conv(2,
                                           miopenConvolution,
                                           miopenPaddingDefault,
                                           {ph, pw},
                                           {sh, sw},
                                           {dh, dw},
                                           {0, 0},
                                           gc,
                                           1.0f);
        const bool fwd = dir == miopen::conv::Direction::Forward;
        return miopen::conv::ProblemDescription(fwd ? in : out, wei, fwd ? out : in, conv, dir);
    }
};

// The picker reconstructs problem features from a ProblemDescription, which
// cannot recover every nuance the fixture's perf-DB rows carried (some have a
// NaN layout / missing workload features). So an exact 225/225 match is not
// expected; instead we require a high reproduction rate. A regression in the
// feature ordering, categorical encoding, or GPU table would collapse this far
// below the threshold.
TEST_F(CPU_LgbmPickerFixture, ReproducesReferencePicks)
{
    const auto& vectors = fixture.at("vectors");
    ASSERT_FALSE(vectors.empty());

    int total = 0;
    int match = 0;
    int valid = 0;
    for(const auto& v : vectors)
    {
        const std::string spec_id = v.at("spec_id").get<std::string>();
        const int spec_code       = SpecIdCode(spec_id);
        if(spec_code < 0)
            continue; // spec not in the runtime table (shouldn't happen for fixture)
        ++total;

        const auto problem = BuildProblem(v.at("problem_inputs"));
        const auto picked  = miopen::ai::lgbm::PickSolverForSpec(problem, spec_code);

        // The picker must never return a solver name MIOpen doesn't recognize.
        if(picked.IsValid())
        {
            ++valid;
            const std::string expected = v.at("expected_selected_solver").get<std::string>();
            if(picked.ToString() == expected)
                ++match;
        }
    }

    ASSERT_GT(total, 0);
    // v5 never abstains, so every covered vector should yield a valid pick.
    EXPECT_EQ(valid, total) << "picker abstained on " << (total - valid) << "/" << total
                            << " vectors";

    const double rate = static_cast<double>(match) / total;
    EXPECT_GE(rate, 0.90) << match << "/" << total << " reference picks reproduced (" << rate
                          << ")";
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
