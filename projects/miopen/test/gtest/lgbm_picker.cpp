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

// CPU-only unit tests for the v10 runtime-pure LGBM solver picker.
// No GPU required: the metadata loader and the full-vocab scoring path are
// host-side. The fixture replay validates the scoring + argmax against
// lgbm_test_vectors.json, whose vectors ship the fully-encoded 51-feature row
// and the reference argmax solver.
//
// Build: make test_lgbm_picker
// Run:   ./bin/test_lgbm_picker

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>

#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using miopen::ai::lgbm::kNumFeatures;
using miopen::ai::lgbm::LgbmMetadata;
using miopen::ai::lgbm::ScoreRowArgmaxForTest;

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
    // v10 ships a non-trivial solver vocabulary; the exact count tracks the
    // model, so just assert it is populated and self-consistent.
    EXPECT_GT(meta.Solvers().size(), 1u);
}

TEST_F(CPU_LgbmMetadata, CategoricalCodesResolve)
{
    EXPECT_GE(meta.CategoricalCode("direction", "1"), 0);
    EXPECT_GE(meta.CategoricalCode("direction", "2"), 0);
    EXPECT_GE(meta.CategoricalCode("direction", "4"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "fp16"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "fp32"), 0);
    EXPECT_GE(meta.CategoricalCode("data_type", "bf16"), 0);
    EXPECT_GE(meta.CategoricalCode("gfx_id", "gfx942"), 0);
    EXPECT_GE(meta.CategoricalCode("gfx_id", "gfx950"), 0);

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
};

// The fixture ships the fully-encoded 51-feature row per vector, so the scoring
// + argmax path can be validated exactly (no ProblemDescription round-trip).
// A regression in the feature count, vocab loading, or argmax would break this.
TEST_F(CPU_LgbmPickerFixture, ReproducesReferenceArgmax)
{
    const auto& vectors = fixture.at("vectors");
    ASSERT_FALSE(vectors.empty());

    // Fixture feature_order must match the model the picker loaded.
    ASSERT_EQ(fixture.at("feature_order").size(), static_cast<std::size_t>(kNumFeatures));

    int total = 0;
    int match = 0;
    for(const auto& v : vectors)
    {
        const auto& jrow = v.at("feature_matrix_first_row");
        if(jrow.size() != static_cast<std::size_t>(kNumFeatures))
            continue;

        std::vector<double> row(kNumFeatures);
        for(int i = 0; i < kNumFeatures; ++i)
        {
            const auto& e = jrow[i];
            row[i]        = e.is_null() ? std::numeric_limits<double>::quiet_NaN()
                                        : e.get<double>();
        }

        const std::string picked   = ScoreRowArgmaxForTest(row);
        const std::string expected = v.at("argmax_solver").get<std::string>();
        ++total;
        if(picked == expected)
            ++match;
    }

    ASSERT_GT(total, 0);
    // The fixture provides the exact encoded rows, so this should be a perfect
    // reproduction.
    EXPECT_EQ(match, total) << match << "/" << total << " reference argmaxes reproduced";
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
