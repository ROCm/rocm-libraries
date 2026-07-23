// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only, self-contained unit tests for the cross-arch LGBM solver picker.
// No GPU required and no external reference data: the metadata loader and the
// full-vocab scoring path are host-side. The tests assert repo-owned
// invariants (vocab loads and is self-consistent; the scoring seam picks a
// valid in-range candidate deterministically). They do NOT check picks against
// any externally-computed "expected" values -- the training/export pipeline
// lives outside this repo and the compiled Treelite .so is the source of truth.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>

#include <miopen/db_path.hpp>

#include <vector>

namespace {

using miopen::ai::lgbm::kNumFeatures;
using miopen::ai::lgbm::LgbmMetadata;
using miopen::ai::lgbm::ScoreCandidateMatrixForTest;

// ---------------------------------------------------------------------------
// Metadata loader
// ---------------------------------------------------------------------------

class CPU_LgbmMetadata_NONE : public ::testing::Test
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

TEST_F(CPU_LgbmMetadata_NONE, LoadsSolverVocab)
{
    // The model ships a non-trivial solver vocabulary; the exact count tracks
    // the model, so just assert it is populated.
    EXPECT_GT(meta.Solvers().size(), 1u);
}

TEST_F(CPU_LgbmMetadata_NONE, CategoricalCodesResolve)
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

TEST_F(CPU_LgbmMetadata_NONE, SolverCodeMatchesVocabIndex)
{
    const auto& solvers = meta.Solvers();
    for(int i = 0; i < static_cast<int>(solvers.size()); ++i)
        EXPECT_EQ(meta.SolverCode(solvers[i]), i);
}

// ---------------------------------------------------------------------------
// Scoring seam
// ---------------------------------------------------------------------------

class CPU_LgbmPicker_NONE : public ::testing::Test
{
protected:
    const LgbmMetadata& meta = LgbmMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "LGBM metadata unavailable; picker disabled in this build";
    }
};

// Scoring a candidate matrix returns an in-range argmax deterministically, and
// rejects malformed input. Uses a fixed, arbitrary feature matrix: the test
// asserts the seam's contract (valid index, determinism, error handling), not a
// specific pick -- the .so is the source of truth for the actual scores.
TEST_F(CPU_LgbmPicker_NONE, ScoresCandidateMatrix)
{
    constexpr int n_cands = 4;
    std::vector<std::vector<double>> rows(
        n_cands, std::vector<double>(static_cast<std::size_t>(kNumFeatures), 1.0));
    // Vary the solver_name column (last feature) so the rows are distinct
    // candidates rather than identical duplicates.
    for(int c = 0; c < n_cands; ++c)
        rows[static_cast<std::size_t>(c)].back() = static_cast<double>(c);

    const int argmax = ScoreCandidateMatrixForTest(rows);
    ASSERT_GE(argmax, 0);
    ASSERT_LT(argmax, n_cands);

    // Deterministic: same matrix -> same argmax.
    EXPECT_EQ(argmax, ScoreCandidateMatrixForTest(rows));

    // Malformed input (wrong feature width) is rejected with -1, not a crash.
    std::vector<std::vector<double>> bad(1, std::vector<double>(4, 0.0));
    EXPECT_EQ(ScoreCandidateMatrixForTest(bad), -1);
    EXPECT_EQ(ScoreCandidateMatrixForTest({}), -1);
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
