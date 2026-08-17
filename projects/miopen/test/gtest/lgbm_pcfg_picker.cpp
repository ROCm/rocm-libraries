// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only unit tests for the per-solver perf-config picker. No GPU required.
// Two kinds of checks:
//   * a self-contained ranking invariant: ScorePickForTest returns a
//     deterministic permutation of the bucket's real candidate descriptors
//     (nothing invented or dropped), driven off the shipped catalog+metadata, and
//   * a golden-vector parity gate: the LightGBM text-model walker (LgbmForest)
//     reproduces LightGBM's own raw scores within a tight tolerance on committed
//     (features -> score) vectors, for each per-solver model.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_pick.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_forest.hpp>

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
using miopen::ai::lgbm::pcfg::kNumBaseProbFeatures;
using miopen::ai::lgbm::pcfg::LgbmPcfgMetadata;
using miopen::ai::lgbm::pcfg::ScorePickForTest;

class CPU_LgbmPcfgPicker_NONE : public ::testing::Test
{
protected:
    nlohmann::json catalog;
    const LgbmPcfgMetadata& meta = LgbmPcfgMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "lgbm_pcfg metadata unavailable; picker disabled in this build";

        const auto cpath = miopen::GetSystemDbPath() / "lgbm_pcfg_catalog.json";
        if(!miopen::fs::exists(cpath))
            GTEST_SKIP() << "catalog not found in " << miopen::GetSystemDbPath().string();
        std::ifstream(cpath.string()) >> catalog;
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

    for(auto sit = catalog.begin(); sit != catalog.end(); ++sit)
    {
        const std::string solver = sit.key();
        const auto* model        = meta.Find(solver);
        if(model == nullptr)
            continue; // catalog entry without a loaded model
        ++solvers_checked;

        // A fixed, arbitrary problem+GPU prefix of the solver's exact length.
        // Values are irrelevant to the invariants under test (we assert
        // structure/determinism, not a specific pick), so any finite vector
        // works; use 1.0 to stay in-range for log-scaled columns.
        const std::vector<double> prefix(static_cast<std::size_t>(model->prob_feat_count), 1.0);

        for(auto bit = sit.value().at("buckets").begin(); bit != sit.value().at("buckets").end();
            ++bit)
        {
            std::vector<std::string> descs;
            std::vector<std::vector<double>> args;
            std::unordered_set<std::string> desc_set;
            for(const auto& c : bit.value())
            {
                auto d = c.at("desc").get<std::string>();
                std::vector<double> a;
                a.reserve(c.at("args").size());
                for(const auto& x : c.at("args"))
                    a.push_back(x.is_null() ? 0.0 : x.get<double>());
                desc_set.insert(d);
                descs.push_back(std::move(d));
                args.push_back(std::move(a));
            }
            if(descs.empty())
                continue;

            const auto ranked = ScorePickForTest(solver, prefix, descs, args);
            ASSERT_EQ(ranked.size(), descs.size())
                << solver << " bucket " << bit.key() << ": ranking size mismatch";

            // Every ranked entry is a real catalog descriptor, and the ranking
            // is a permutation (no dupes) of the input set.
            std::unordered_set<std::string> seen;
            for(const auto& r : ranked)
            {
                EXPECT_TRUE(desc_set.count(r) == 1) << solver << " bucket " << bit.key()
                                                    << ": ranked non-catalog desc \"" << r << "\"";
                EXPECT_TRUE(seen.insert(r).second) << solver << " bucket " << bit.key()
                                                   << ": duplicate ranked desc \"" << r << "\"";
            }

            // Deterministic: a second identical call yields the same order.
            const auto ranked2 = ScorePickForTest(solver, prefix, descs, args);
            EXPECT_EQ(ranked, ranked2)
                << solver << " bucket " << bit.key() << ": ranking not deterministic";

            ++buckets_checked;
        }
    }

    ASSERT_GT(solvers_checked, 0) << "no catalog solver matched a loaded model";
    ASSERT_GT(buckets_checked, 0) << "no non-empty buckets scored";
}

// Golden-vector parity for the per-solver pcfg models: the forest walker must
// reproduce LightGBM's raw score for every committed (features -> expected)
// vector, across all 11 solver models. The fixture includes random+NaN rows so
// the numeric missing_type paths are exercised. Ground truth is produced by the
// LightGBM Python API from the exact models shipped in the tree.
TEST(CPU_LgbmPcfgForest_NONE, MatchesGoldenVectors)
{
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
        const auto model_path = miopen::GetSystemDbPath() / ("lgbm_pcfg_" + solver + "_model.txt");
        if(!miopen::fs::exists(model_path))
            continue; // model asset not installed in this build
        const LgbmForest forest(model_path.string());
        ASSERT_TRUE(forest.IsReady()) << "failed to load " << model_path.string();
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

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
