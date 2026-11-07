// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only unit tests for the per-solver perf-config picker. No GPU required.
// Three kinds of checks:
//   * a self-contained ranking invariant: ScorePickForTest returns a
//     deterministic permutation of the bucket's real candidate descriptors
//     (nothing invented or dropped), driven off the shipped catalog+metadata,
//   * a golden-vector parity gate: the LightGBM text-model walker (LgbmForest)
//     reproduces LightGBM's own raw scores within a tight tolerance on committed
//     (features -> score) vectors, for each per-solver model, and
//   * the (solver, arch, direction) hold-outs in HeldOutOfPicker.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_pick.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_forest.hpp>
#include <miopen/conv_algo_name.hpp>

#include <miopen/conv/heuristics/lgbm_common.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using miopen::ai::lgbm::LgbmForest;
using miopen::ai::lgbm::NormalizeArchName;
using miopen::ai::lgbm::pcfg::BucketKey;
using miopen::ai::lgbm::pcfg::GfxCode;
using miopen::ai::lgbm::pcfg::HeldOutOfPicker;
using miopen::ai::lgbm::pcfg::kNumBaseProbFeatures;
using miopen::ai::lgbm::pcfg::LgbmPcfgMetadata;
using miopen::ai::lgbm::pcfg::ScorePickForTest;

class CPU_LgbmPcfgPicker_NONE : public ::testing::Test
{
protected:
    const LgbmPcfgMetadata& meta = LgbmPcfgMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "lgbm_pcfg metadata unavailable; picker disabled in this build";
    }
};

// For every solver+bucket in the shipped catalog, scoring the bucket's
// candidates must return a ranking that:
//   - is non-empty,
//   - is a permutation of the bucket's candidate descriptors (every returned
//     entry is a real catalog desc; none invented, none dropped),
//   - is deterministic (same inputs -> same order on a second call).
// This exercises the metadata load, per-solver predict dispatch, feature-row
// assembly, and the stable argsort -- without any external reference.
TEST_F(CPU_LgbmPcfgPicker_NONE, RanksRealCatalogCandidatesDeterministically)
{
    int solvers_checked = 0;
    int buckets_checked = 0;

    for(const auto& solver : meta.SolverNames())
    {
        const auto* model = meta.Find(solver);
        ASSERT_NE(model, nullptr);
        ++solvers_checked;

        // A fixed, arbitrary problem+GPU prefix of the solver's exact length.
        // Values are irrelevant to the invariants under test (we assert
        // structure/determinism, not a specific pick), so any finite vector
        // works; use 1.0 to stay in-range for log-scaled columns.
        const std::vector<double> prefix(static_cast<std::size_t>(model->prob_feat_count), 1.0);

        for(const auto& bucket : model->buckets)
        {
            std::vector<std::string> descs;
            std::vector<std::vector<double>> args;
            std::unordered_set<std::string> desc_set;
            for(const auto& c : bucket.second)
            {
                // The loaded catalog encodes a missing arg as NaN; map it to a
                // finite 0.0 here so the scores (hence the sort order) stay
                // deterministic for this structural invariant.
                std::vector<double> a = c.args;
                for(auto& x : a)
                    if(std::isnan(x))
                        x = 0.0;
                desc_set.insert(c.desc);
                descs.push_back(c.desc);
                args.push_back(std::move(a));
            }
            if(descs.empty())
                continue;

            const auto ranked = ScorePickForTest(solver, prefix, descs, args);
            ASSERT_EQ(ranked.size(), descs.size())
                << solver << " bucket " << bucket.first << ": ranking size mismatch";

            // Every ranked entry is a real catalog descriptor, and the ranking
            // is a permutation (no dupes) of the input set.
            std::unordered_set<std::string> seen;
            for(const auto& r : ranked)
            {
                EXPECT_TRUE(desc_set.count(r) == 1) << solver << " bucket " << bucket.first
                                                    << ": ranked non-catalog desc \"" << r << "\"";
                EXPECT_TRUE(seen.insert(r).second) << solver << " bucket " << bucket.first
                                                   << ": duplicate ranked desc \"" << r << "\"";
            }

            // Deterministic: a second identical call yields the same order.
            const auto ranked2 = ScorePickForTest(solver, prefix, descs, args);
            EXPECT_EQ(ranked, ranked2)
                << solver << " bucket " << bucket.first << ": ranking not deterministic";

            ++buckets_checked;
        }
    }

    ASSERT_GT(solvers_checked, 0) << "no loaded pcfg solver model";
    ASSERT_GT(buckets_checked, 0) << "no non-empty buckets scored";
}

// A gfx_code solver encodes the live gfx_id as its index in the model's shipped
// vocab. Every arch the catalog has buckets for must be in that vocab; a missing
// one would silently be scored as the unknown-arch (-1) category.
TEST_F(CPU_LgbmPcfgPicker_NONE, GfxVocabCoversEveryBucketArch)
{
    int gfx_solvers = 0;
    for(const auto& solver : meta.SolverNames())
    {
        const auto* model = meta.Find(solver);
        ASSERT_NE(model, nullptr);
        if(!model->has_gfx_code)
        {
            EXPECT_TRUE(model->gfx_vocab.empty()) << solver << ": vocab without gfx_code";
            continue;
        }
        ++gfx_solvers;
        const std::unordered_set<std::string> vocab(model->gfx_vocab.begin(),
                                                    model->gfx_vocab.end());
        EXPECT_EQ(vocab.size(), model->gfx_vocab.size()) << solver << ": duplicate gfx_id";
        for(const auto& bucket : model->buckets)
        {
            const auto gfx_id = bucket.first.substr(0, bucket.first.find('|'));
            EXPECT_TRUE(vocab.count(gfx_id) == 1)
                << solver << ": bucket arch " << gfx_id << " missing from gfx_vocab";
        }
    }
    ASSERT_GT(gfx_solvers, 0) << "no gfx_code solver in the bundle";
}

// A0 gfx1250 parts report their device name as "gfx1250-strict". The bucket keys and the
// gfx_code vocab only have the base name, so a gfx1250-strict problem must land in the same
// bucket, get the same gfx_code and therefore the same ranking as on gfx1250; an arch outside
// the catalog must still find no bucket.
TEST_F(CPU_LgbmPcfgPicker_NONE, StrictArchSharesBaseArchBucketAndPick)
{
    using miopen::conv::Direction;
    const std::string solver = "ConvHipImplicitGemm3DGroupFwdXdlops";
    const auto* model        = meta.Find(solver);
    ASSERT_NE(model, nullptr);
    ASSERT_TRUE(model->forest);

    // Same key for every direction and dtype the key encodes.
    for(const auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
    {
        for(const auto dt : {miopenHalf, miopenBFloat16, miopenFloat, miopenInt8})
        {
            const auto base = BucketKey("gfx1250", dir, dt);
            EXPECT_TRUE(base.starts_with("gfx1250|")) << base;
            EXPECT_EQ(BucketKey("gfx1250-strict", dir, dt), base);
            EXPECT_EQ(BucketKey("gfx942-strict", dir, dt), BucketKey("gfx942", dir, dt));
            EXPECT_EQ(BucketKey("gfx9999-strict", dir, dt), BucketKey("gfx9999", dir, dt));
        }
    }
    EXPECT_EQ(BucketKey("gfx1250-strict", Direction::Forward, miopenHalf), "gfx1250|1|fp16");

    // The 3DGroupFwd forward bucket for gfx1250 exists, and is found through the strict name.
    const std::string base_key   = BucketKey("gfx1250", Direction::Forward, miopenHalf);
    const std::string strict_key = BucketKey("gfx1250-strict", Direction::Forward, miopenHalf);
    const auto bit               = model->buckets.find(strict_key);
    ASSERT_NE(bit, model->buckets.end()) << solver << " has no bucket " << strict_key;
    ASSERT_EQ(model->buckets.find(base_key), bit);
    ASSERT_FALSE(bit->second.empty());

    // Unknown arch: no bucket and the missing gfx_code, with or without the suffix.
    for(const char* unknown : {"gfx9999", "gfx9999-strict"})
    {
        EXPECT_EQ(model->buckets.find(BucketKey(unknown, Direction::Forward, miopenHalf)),
                  model->buckets.end());
        EXPECT_EQ(GfxCode(*model, unknown), -1);
    }

    // gfx_code of the strict name is the base name's index in the model's vocab.
    if(model->has_gfx_code)
    {
        const int code = GfxCode(*model, "gfx1250");
        ASSERT_GE(code, 0);
        EXPECT_EQ(model->gfx_vocab[static_cast<std::size_t>(code)], "gfx1250");
        EXPECT_EQ(GfxCode(*model, "gfx1250-strict"), code);
        EXPECT_NE(GfxCode(*model, "gfx942"), code);
    }

    // Rank the bucket's candidates the way PickConfig does: prefix + gfx_code from the
    // device name, then ScorePickForTest. Strict and base must agree, and be a full ranking.
    std::vector<std::string> descs;
    std::vector<std::vector<double>> args;
    for(const auto& c : bit->second)
    {
        std::vector<double> a = c.args;
        for(auto& x : a)
            if(std::isnan(x))
                x = 0.0;
        descs.push_back(c.desc);
        args.push_back(std::move(a));
    }
    const auto rank = [&](const std::string& device_name) {
        std::vector<double> prefix(static_cast<std::size_t>(model->prob_feat_count), 1.0);
        if(model->has_gfx_code)
            prefix.back() = static_cast<double>(GfxCode(*model, device_name));
        return ScorePickForTest(solver, prefix, descs, args);
    };
    const auto ranked_base   = rank("gfx1250");
    const auto ranked_strict = rank("gfx1250-strict");
    ASSERT_EQ(ranked_base.size(), descs.size());
    EXPECT_EQ(ranked_strict, ranked_base);
}

// Golden-vector parity for the per-solver pcfg models: the forest walker must
// reproduce LightGBM's raw score for every committed (features -> expected)
// vector, across all 11 solver models. The fixture includes random+NaN rows so
// the numeric missing_type paths are exercised. Ground truth is produced by the
// LightGBM Python API from the exact models shipped in the tree.
TEST(CPU_LgbmPcfgForest_NONE, MatchesGoldenVectors)
{
    const auto& meta = LgbmPcfgMetadata::Get();
    if(!meta.IsReady())
        GTEST_SKIP() << "lgbm_pcfg metadata unavailable; picker disabled in this build";

    const auto gpath = miopen::GetSystemDbPath() / "lgbm_pcfg_golden.json";
    if(!miopen::fs::exists(gpath))
        GTEST_SKIP() << "pcfg golden fixture not installed in "
                     << miopen::GetSystemDbPath().string();
    nlohmann::json golden;
    std::ifstream(gpath.string()) >> golden;

    const auto& solvers = golden.at("solvers");
    ASSERT_GT(solvers.size(), 0u);

    int solvers_checked = 0;
    double max_abs_err  = 0.0;
    for(auto sit = solvers.begin(); sit != solvers.end(); ++sit)
    {
        const std::string solver = sit.key();
        const auto* model        = meta.Find(solver);
        if(model == nullptr || !model->forest)
            continue; // model not loaded in this build
        const LgbmForest& forest = *model->forest;
        ASSERT_TRUE(forest.IsReady()) << "forest not ready for " << solver;
        ++solvers_checked;

        const auto& block     = sit.value();
        const auto feat_count = block.at("feature_count").get<std::size_t>();
        const auto& rows      = block.at("rows");
        const auto& expected  = block.at("expected");
        ASSERT_EQ(rows.size(), expected.size());

        for(std::size_t r = 0; r < rows.size(); ++r)
        {
            const auto& jrow = rows[r];
            ASSERT_EQ(jrow.size(), feat_count) << solver << " row " << r << " wrong width";
            std::vector<LgbmEntry> row(feat_count);
            for(std::size_t i = 0; i < feat_count; ++i)
            {
                if(jrow[i].is_null())
                    row[i].missing = -1;
                else
                {
                    row[i].missing = 0;
                    row[i].fvalue  = jrow[i].get<double>();
                }
            }
            const double got = forest.Score(row.data(), row.size());
            const double exp = expected[r].get<double>();
            max_abs_err      = std::max(max_abs_err, std::abs(got - exp));
            EXPECT_NEAR(got, exp, 1e-6) << solver << " row " << r << " score mismatch";
        }
    }
    ASSERT_GT(solvers_checked, 0) << "no pcfg model asset was found to check";
    MIOPEN_LOG_I2("lgbm_pcfg golden parity: " << solvers_checked << " solvers, max abs err "
                                              << max_abs_err);
}

// Pins which (solver, arch, direction) combinations the picker holds out although
// lgbm_pcfg.bin has buckets for them. ConvHipConv is held out on gfx1250 in every direction and
// on gfx950 for backward weights only.
TEST(CPU_LgbmPcfgHoldOut_NONE, HeldOutOfPicker)
{
    using miopen::conv::Direction;
    const std::string hipconv = "ConvHipConv";

    EXPECT_FALSE(HeldOutOfPicker(hipconv, "gfx950", Direction::Forward));
    EXPECT_FALSE(HeldOutOfPicker(hipconv, "gfx950", Direction::BackwardData));
    EXPECT_TRUE(HeldOutOfPicker(hipconv, "gfx950", Direction::BackwardWeights));

    for(const auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
    {
        EXPECT_TRUE(HeldOutOfPicker(hipconv, "gfx1250", dir));
        EXPECT_FALSE(HeldOutOfPicker(hipconv, "gfx942", dir));
        EXPECT_FALSE(HeldOutOfPicker("ConvHipImplicitGemmV4R1Fwd", "gfx950", dir));
        EXPECT_FALSE(HeldOutOfPicker("ConvHipImplicitGemmV4R1Fwd", "gfx1250", dir));
    }
}

// The hold-out matches on prefix, so the "-strict" spelling of a held-out arch is held out too,
// with or without normalizing the name first (PickConfig passes the normalized one).
TEST(CPU_LgbmPcfgHoldOut_NONE, HeldOutOfPickerStrictArch)
{
    using miopen::conv::Direction;
    const std::string hipconv = "ConvHipConv";

    for(const auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
    {
        EXPECT_TRUE(HeldOutOfPicker(hipconv, "gfx1250-strict", dir));
        EXPECT_TRUE(HeldOutOfPicker(hipconv, NormalizeArchName("gfx1250-strict"), dir));
        EXPECT_FALSE(HeldOutOfPicker(hipconv, "gfx942-strict", dir));
        EXPECT_FALSE(HeldOutOfPicker("ConvHipImplicitGemm3DGroupFwdXdlops", "gfx1250-strict", dir));
    }
    EXPECT_FALSE(HeldOutOfPicker(hipconv, "gfx950-strict", Direction::Forward));
    EXPECT_TRUE(HeldOutOfPicker(hipconv, "gfx950-strict", Direction::BackwardWeights));
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
