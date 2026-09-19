// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only unit tests for the cross-arch LGBM solver picker. No GPU required.
// Two kinds of checks:
//   * self-contained invariants (vocab loads and is self-consistent; the
//     scoring seam picks a valid in-range candidate deterministically), and
//   * a golden-vector parity gate: the LightGBM text-model walker (LgbmForest)
//     must reproduce LightGBM's own raw lambdarank scores within a tight
//     tolerance on committed (features -> score) vectors. The golden scores are
//     produced by the LightGBM Python API from the model shipped in the tree.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_forest.hpp>

#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <vector>

namespace {

using miopen::ai::lgbm::kNumFeatures;
using miopen::ai::lgbm::LgbmForest;
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

TEST_F(CPU_LgbmMetadata_NONE, UnseenArchSupportIsConsistent)
{
    // The unseen-arch code is only meaningful when the model was trained for it;
    // when unsupported, the picker abstains on unknown arch. Either way the code
    // is a valid sentinel (default -1 = the model's missing branch).
    if(meta.AllowUnseenArch())
        EXPECT_LT(meta.UnseenArchCode(), meta.CategoricalCode("gfx_id", "gfx942"))
            << "unseen code should be a sentinel below any real vocab code";
    else
        EXPECT_EQ(meta.UnseenArchCode(), -1);
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

// ---------------------------------------------------------------------------
// Golden-vector parity: the forest walker vs LightGBM's own scores
// ---------------------------------------------------------------------------

// The walker (LgbmForest) must reproduce LightGBM's raw lambdarank score for
// every committed (features -> expected) vector within a tight tolerance. Rows
// are 61-feature LgbmEntry inputs: JSON null encodes a missing/NaN feature;
// categoricals are pre-encoded as integer codes. This exercises numeric splits,
// categorical bitset membership, and the missing/default-left path against
// ground truth generated by the LightGBM Python API from the model shipped in
// the tree.
TEST(CPU_LgbmForest_NONE, MatchesGoldenVectors)
{
    const auto& forest = LgbmForest::GetRank();
    if(!forest.IsReady())
        GTEST_SKIP() << "rank model not found in " << miopen::GetSystemDbPath().string();

    const auto gpath = miopen::GetSystemDbPath() / "lgbm_rank_golden.json";
    if(!miopen::fs::exists(gpath))
        GTEST_SKIP() << "golden fixture not installed in " << miopen::GetSystemDbPath().string();
    std::ifstream in(gpath.string());
    ASSERT_TRUE(in.is_open()) << "cannot open golden fixture: " << gpath.string();
    nlohmann::json golden;
    in >> golden;

    const auto& rows     = golden.at("rows");
    const auto& expected = golden.at("expected");
    ASSERT_EQ(rows.size(), expected.size());
    ASSERT_GT(rows.size(), 0u);

    const auto feat_count = golden.at("feature_count").get<std::size_t>();
    double max_abs_err    = 0.0;
    for(std::size_t r = 0; r < rows.size(); ++r)
    {
        const auto& jrow = rows[r];
        ASSERT_EQ(jrow.size(), feat_count) << "row " << r << " wrong width";
        std::vector<LgbmEntry> row(feat_count);
        for(std::size_t i = 0; i < feat_count; ++i)
        {
            if(jrow[i].is_null())
                row[i].missing = -1; // NaN / absent feature
            else
            {
                row[i].missing = 0;
                row[i].fvalue  = jrow[i].get<double>();
            }
        }
        const double got = forest.Score(row.data(), row.size());
        const double exp = expected[r].get<double>();
        max_abs_err      = std::max(max_abs_err, std::abs(got - exp));
        EXPECT_NEAR(got, exp, 1e-6) << "row " << r << " score mismatch";
    }
    MIOPEN_LOG_I2("lgbm golden parity: " << rows.size() << " rows, max abs err " << max_abs_err);
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
