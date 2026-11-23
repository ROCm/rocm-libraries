// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the cross-arch LGBM solver picker. All but two are CPU-only.
// GPU_LgbmPickerStrictArch_FP16 and GPU_LgbmPickerStrictArch_FP32 need a HIP device for their
// MockHandle.
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

#include <miopen/conv/heuristics/lgbm_common.hpp>
#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_forest.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

#include "gtest_common.hpp" // DevDescription, MockHandle

#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using miopen::ai::lgbm::GfxVocabCode;
using miopen::ai::lgbm::kNumFeatures;
using miopen::ai::lgbm::LgbmForest;
using miopen::ai::lgbm::LgbmMetadata;
using miopen::ai::lgbm::NormalizeArchName;
using miopen::ai::lgbm::PickSolverRanked;
using miopen::ai::lgbm::ScoreCandidateMatrixForTest;
using miopen::ai::lgbm::SetNumeric;

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

// gfx1250 parts report their device name as "gfx1250-strict". The model vocab only has the
// base name, so the L1 lookup must treat both the same (and stay strict about anything else).
TEST_F(CPU_LgbmPicker_NONE, StrictArchScoresLikeBaseArch)
{
    // The suffix rule mirrors legacy_ck_common.hpp: everything from the first '-' is dropped.
    EXPECT_EQ(NormalizeArchName("gfx1250-strict"), "gfx1250");
    EXPECT_EQ(NormalizeArchName("gfx1250"), "gfx1250");
    EXPECT_EQ(NormalizeArchName("gfx942-strict"), "gfx942");
    EXPECT_EQ(NormalizeArchName("gfx1250-other"), "gfx1250");
    EXPECT_EQ(NormalizeArchName("gfx9999-strict"), "gfx9999");
    EXPECT_EQ(NormalizeArchName("gfx9999"), "gfx9999");
    EXPECT_EQ(NormalizeArchName(""), "");

    // Same vocab code, and it is the code the vocab itself assigns to the base name.
    const int code = GfxVocabCode("gfx1250");
    ASSERT_GE(code, 0) << "gfx1250 is expected in the shipped rank vocab";
    EXPECT_EQ(code, meta.CategoricalCode("gfx_id", "gfx1250"));
    EXPECT_EQ(GfxVocabCode("gfx1250-strict"), code);
    EXPECT_EQ(GfxVocabCode("gfx942-strict"), GfxVocabCode("gfx942"));
    EXPECT_GE(GfxVocabCode("gfx942"), 0);
    EXPECT_NE(GfxVocabCode("gfx942"), code);

    // An arch outside the vocab still takes the unseen path (-1), with or without a suffix.
    EXPECT_EQ(GfxVocabCode("gfx9999"), -1);
    EXPECT_EQ(GfxVocabCode("gfx9999-strict"), -1);
    EXPECT_EQ(GfxVocabCode("gfx1250x"), -1);
    EXPECT_EQ(GfxVocabCode(""), -1);

    const auto& forest = LgbmForest::GetRank();
    if(!forest.IsReady())
        GTEST_SKIP() << "rank model unavailable";

    // Score every solver in the vocab on two fixed problem rows with the gfx_id column set from
    // GfxVocabCode, the same way PickSolverRanked fills it. gfx1250 and gfx1250-strict must
    // give bit-identical scores; the unseen arch (missing branch) must not, or the comparison
    // would not be sensitive to the gfx_id column at all.
    constexpr int kIdxGfxId      = 53; // lgbm_model_meta.json rank.feature_order
    constexpr int kIdxSolverName = 54;
    static_assert(kIdxSolverName == kNumFeatures - 1);

    const auto score = [&](const std::string& device_name, int solver_code, int variant) {
        std::array<LgbmEntry, kNumFeatures> row{};
        for(int i = 0; i < kNumFeatures; ++i)
            SetNumeric(row[i], 1.0 + static_cast<double>((i * (variant + 1)) % 7));
        const int gfx               = GfxVocabCode(device_name);
        row[kIdxGfxId].missing      = gfx < 0 ? -1 : 0;
        row[kIdxGfxId].fvalue       = gfx < 0 ? 0.0 : static_cast<double>(gfx);
        row[kIdxSolverName].missing = 0;
        row[kIdxSolverName].fvalue  = static_cast<double>(solver_code);
        return forest.Score(row.data(), row.size());
    };

    bool unseen_differs = false;
    for(int variant = 0; variant < 2; ++variant)
    {
        for(const auto& solver : meta.Solvers())
        {
            const int sc          = meta.SolverCode(solver);
            const double base     = score("gfx1250", sc, variant);
            const double strict   = score("gfx1250-strict", sc, variant);
            const double unseen   = score("gfx9999", sc, variant);
            const double unseen_s = score("gfx9999-strict", sc, variant);
            EXPECT_EQ(strict, base) << solver;
            EXPECT_EQ(unseen_s, unseen) << solver;
            unseen_differs = unseen_differs || unseen != base;
        }
    }
    EXPECT_TRUE(unseen_differs) << "gfx_id column has no effect on any score; test is vacuous";
}

// The same check through the production entry point. The test above scores with a hand-built
// row, so it passes even if PickSolverRanked looks the gfx_id up with the raw device name.
// A mock handle reports the device name, everything else comes from the real device.
// One fixture per data type, each with its own control.
template <miopenDataType_t DataType>
class LgbmPickerStrictArch : public CPU_LgbmPicker_NONE
{
protected:
    static std::unique_ptr<MockHandle> MakeHandle(std::string_view name)
    {
        // gfx1250 is a 32 CU, wave32 part in gtest_common.cpp; every name gets the same values.
        return std::make_unique<MockHandle>(DevDescription{name, 32, 32}, false);
    }

    static miopen::conv::ProblemDescription MakeProblem(miopen::conv::Direction dir,
                                                        miopenDataType_t dtype,
                                                        int n,
                                                        int c,
                                                        int k,
                                                        int hw,
                                                        int fil,
                                                        int groups)
    {
        const miopen::TensorDescriptor x(dtype, {n, c, hw, hw});
        const miopen::TensorDescriptor w(dtype, {k, c / groups, fil, fil});
        const miopen::TensorDescriptor y(dtype, {n, k, hw, hw}); // same-size output
        const int pad = fil / 2;
        const miopen::ConvolutionDescriptor conv(
            2, miopenConvolution, miopenPaddingDefault, {pad, pad}, {1, 1}, {1, 1}, {0, 0}, groups);
        return dir == miopen::conv::Direction::Forward
                   ? miopen::conv::ProblemDescription(x, w, y, conv, dir)
                   : miopen::conv::ProblemDescription(y, w, x, conv, dir);
    }

    void CheckPickSolverRankedMatchesBaseArch()
    {
        std::unique_ptr<MockHandle> base, strict, unseen, unseen_strict;
        try
        {
            base          = MakeHandle("gfx1250");
            strict        = MakeHandle("gfx1250-strict");
            unseen        = MakeHandle("gfx9999");
            unseen_strict = MakeHandle("gfx9999-strict");
        }
        catch(const std::exception& e)
        {
            GTEST_SKIP() << "no HIP device for the mock handle: " << e.what();
        }
        ASSERT_EQ(base->GetDeviceName(), "gfx1250");
        ASSERT_EQ(strict->GetDeviceName(), "gfx1250-strict");

        int problems        = 0;
        bool unseen_differs = false;
        using miopen::conv::Direction;
        for(const auto dir :
            {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
        {
            // n, c, k, hw, fil, groups
            for(const auto& s : {std::array<int, 6>{16, 64, 64, 56, 3, 1},
                                 std::array<int, 6>{32, 256, 128, 14, 1, 1},
                                 std::array<int, 6>{2, 64, 64, 28, 3, 4}})
            {
                const auto problem = MakeProblem(dir, DataType, s[0], s[1], s[2], s[3], s[4], s[5]);
                const auto ranked  = PickSolverRanked(problem, *base);
                ASSERT_FALSE(ranked.empty()) << "gfx1250 is in the vocab and must be scored";
                EXPECT_EQ(PickSolverRanked(problem, *strict), ranked);

                const auto ranked_unseen = PickSolverRanked(problem, *unseen);
                EXPECT_EQ(PickSolverRanked(problem, *unseen_strict), ranked_unseen);
                // Unseen arch: scored through the missing branch, or abstains.
                unseen_differs = unseen_differs || ranked_unseen != ranked;
                ++problems;
            }
        }
        ASSERT_EQ(problems, 9);
        EXPECT_TRUE(unseen_differs) << "gfx_id has no effect on any ranking; test is vacuous";
    }
};

using GPU_LgbmPickerStrictArch_FP16 = LgbmPickerStrictArch<miopenHalf>;
using GPU_LgbmPickerStrictArch_FP32 = LgbmPickerStrictArch<miopenFloat>;

TEST_F(GPU_LgbmPickerStrictArch_FP16, PickSolverRankedMatchesBaseArch)
{
    CheckPickSolverRankedMatchesBaseArch();
}

TEST_F(GPU_LgbmPickerStrictArch_FP32, PickSolverRankedMatchesBaseArch)
{
    CheckPickSolverRankedMatchesBaseArch();
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
