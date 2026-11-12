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
#include <cstdint>
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

// Hand-built solver sections for the metadata parser. They go through
// ParseSolverSectionForTest, the same code Find() runs on a bundle section, so a
// malformed or oversized field must make the parse fail (the picker abstains for
// that solver) instead of allocating what the field claims.
class SectionBytes
{
public:
    SectionBytes& U8(unsigned v)
    {
        bytes.push_back(static_cast<char>(v));
        return *this;
    }
    SectionBytes& U16(unsigned v) { return Le(v, 2); }
    SectionBytes& U32(std::uint32_t v) { return Le(v, 4); }
    SectionBytes& I32(std::int32_t v) { return Le(static_cast<std::uint32_t>(v), 4); }
    SectionBytes& Varint(std::uint64_t v)
    {
        for(; v >= 0x80; v >>= 7)
            U8(static_cast<unsigned>(v & 0x7F) | 0x80);
        return U8(static_cast<unsigned>(v));
    }
    SectionBytes& Repeat(char c, std::size_t n)
    {
        bytes.append(n, c);
        return *this;
    }
    SectionBytes& Chars(const std::string& s)
    {
        bytes += s;
        return *this;
    }

    std::string bytes;

private:
    SectionBytes& Le(std::uint64_t v, unsigned n)
    {
        for(unsigned i = 0; i < n; ++i, v >>= 8)
            U8(static_cast<unsigned>(v & 0xFF));
        return *this;
    }
};

// Section header (no gfx_code) followed by the smallest valid FOREST block: one
// tree with no internal nodes and a single leaf.
SectionBytes SectionPrefix(int arg_count, int feat_count)
{
    SectionBytes s;
    s.I32(feat_count).I32(kNumBaseProbFeatures).I32(arg_count).U8(0);
    s.U32(1).U32(0).U8(1).U8(4);   // 1 tree, 0 thresholds, child width 1, f32 leaves
    s.U16(0);                      // 0 internal nodes
    s.Chars(std::string(4, '\0')); // the one leaf value
    s.U16(0).U16(0);               // no categorical bitset / offsets
    return s;
}

bool Parse(const SectionBytes& s, miopen::ai::lgbm::pcfg::SolverModel& model)
{
    return miopen::ai::lgbm::pcfg::ParseSolverSectionForTest(s.bytes.data(), s.bytes.size(), model);
}

// Catalog with no templates, no candidates and no buckets.
void EmptyCatalog(SectionBytes& s) { s.Varint(0).Varint(0).Varint(0).U8(1); }

// A well-formed 2-arg section: four candidates "a0".."a3" (descriptor template
// "a<slot>", the slot taken from arg column 1), and one bucket "k" listing them
// in reverse. Proves the hand-built layout is one the parser accepts, so the
// rejections below are due to the bound and not to a malformed fixture.
TEST(CPU_LgbmPcfgMetadata_NONE, ParsesWellFormedSection)
{
    auto s = SectionPrefix(2, kNumBaseProbFeatures + 2);
    s.Varint(4).Varint(1);                                    // 4 candidates, 1 template
    s.Varint(0).Varint(2).Chars("a\x01");                     // template: no shared prefix, 2 bytes
    s.Varint(4);                                              // candidates using that template
    s.Varint(4).U8(0).Varint(2).Varint(2).Varint(2).U8(0xE4); // col 1: 0,1,2,3
    s.Varint(1).Varint(14);                                   // col 2: constant 7
    s.U8(1);                                                  // slot 0 reads arg column 1
    s.Varint(1);                                              // 1 bucket
    s.Varint(0).Varint(1).Chars("k");                         // its key
    s.U8(2);                                                  // bucket index width
    s.Varint(4).U8(0x1B);                                     // candidates 3,2,1,0

    miopen::ai::lgbm::pcfg::SolverModel model;
    ASSERT_TRUE(Parse(s, model));
    ASSERT_EQ(model.buckets.size(), 1u);
    const auto& cands = model.buckets.at("k");
    ASSERT_EQ(cands.size(), 4u);
    for(std::size_t i = 0; i < cands.size(); ++i)
    {
        const int v = 3 - static_cast<int>(i);
        EXPECT_EQ(cands[i].desc, "a" + std::to_string(v));
        ASSERT_EQ(cands[i].args.size(), 2u);
        EXPECT_EQ(cands[i].args[0], v);
        EXPECT_EQ(cands[i].args[1], 7.0);
    }
}

// Every section of the shipped bundle must still parse, with the schema the
// header check enforces.
TEST_F(CPU_LgbmPcfgPicker_NONE, ShippedSectionsParse)
{
    const auto names = meta.SolverNames();
    ASSERT_FALSE(names.empty());
    for(const auto& name : names)
    {
        const auto* model = meta.Find(name);
        ASSERT_NE(model, nullptr) << name << " section failed to parse";
        EXPECT_EQ(model->prob_feat_count, kNumBaseProbFeatures + (model->has_gfx_code ? 1 : 0))
            << name;
        EXPECT_EQ(model->feat_count, model->prob_feat_count + model->arg_count) << name;
        EXPECT_FALSE(model->buckets.empty()) << name;
    }
}

// A section whose arg_count the format cannot express is rejected from the
// header, before any column is allocated. 4096 is the case from review: 65,536
// candidates and 4,096 constant arg columns take 8 KiB of input but would expand
// to 2 GiB of columns and another 2 GiB of candidate args.
TEST(CPU_LgbmPcfgMetadata_NONE, RejectsOversizedArgCount)
{
    constexpr int kArgs = 4096;
    auto s              = SectionPrefix(kArgs, kNumBaseProbFeatures + kArgs);
    s.Varint(65536).Varint(1).Varint(0).Varint(1).Chars("x").Varint(65536);
    for(int i = 0; i < kArgs; ++i)
        s.Varint(1).Varint(0); // constant-zero column
    s.Varint(0).U8(1);         // no buckets
    ASSERT_LT(s.bytes.size(), 16u * 1024);

    miopen::ai::lgbm::pcfg::SolverModel model;
    EXPECT_FALSE(Parse(s, model));
}

TEST(CPU_LgbmPcfgMetadata_NONE, RejectsInconsistentHeader)
{
    miopen::ai::lgbm::pcfg::SolverModel model;

    // arg_count 2^31-1 (feat_count wraps to match), with an otherwise empty catalog.
    auto huge = SectionPrefix(INT32_MAX, static_cast<int>(INT32_MIN + kNumBaseProbFeatures - 1));
    EmptyCatalog(huge);
    EXPECT_FALSE(Parse(huge, model));

    auto negative = SectionPrefix(-1, kNumBaseProbFeatures - 1);
    EmptyCatalog(negative);
    EXPECT_FALSE(Parse(negative, model));

    // feat_count disagrees with prob_feat_count + arg_count.
    auto mismatch = SectionPrefix(2, kNumBaseProbFeatures + 3);
    EmptyCatalog(mismatch);
    EXPECT_FALSE(Parse(mismatch, model));
}

// The largest arg_count the format allows (255) still must not decode past the
// section's memory budget: 65,536 candidates by 255 constant columns is 133 MB of
// int64 columns plus 133 MB of double args, from 510 bytes of columns.
TEST(CPU_LgbmPcfgMetadata_NONE, RejectsColumnsOverDecodedBudget)
{
    constexpr int kArgs = 255;
    auto s              = SectionPrefix(kArgs, kNumBaseProbFeatures + kArgs);
    s.Varint(65536).Varint(1).Varint(0).Varint(1).Chars("x").Varint(65536);
    for(int i = 0; i < kArgs; ++i)
        s.Varint(1).Varint(0);
    s.Varint(0).U8(1);

    miopen::ai::lgbm::pcfg::SolverModel model;
    EXPECT_FALSE(Parse(s, model));
}

// A string stored as "share the previous string's whole prefix, add nothing"
// costs two bytes however long the prefix is. 2,000 templates of 100,000 bytes
// are 200 MB decoded from about 104 KB of input.
TEST(CPU_LgbmPcfgMetadata_NONE, RejectsSharedPrefixStringsOverDecodedBudget)
{
    constexpr std::size_t kLen       = 100000;
    constexpr std::uint64_t kStrings = 2000;
    auto s                           = SectionPrefix(0, kNumBaseProbFeatures);
    s.Varint(kStrings).Varint(kStrings);
    s.Varint(0).Varint(kLen).Repeat('a', kLen);
    for(std::uint64_t i = 1; i < kStrings; ++i)
        s.Varint(kLen).Varint(0);
    for(std::uint64_t i = 0; i < kStrings; ++i)
        s.Varint(1); // one candidate per template
    s.Varint(0).U8(1);

    miopen::ai::lgbm::pcfg::SolverModel model;
    EXPECT_FALSE(Parse(s, model));
}

// 200 bucket entries that all copy one 1 MiB descriptor are 200 MB, from a
// 25-byte index.
TEST(CPU_LgbmPcfgMetadata_NONE, RejectsBucketEntriesOverDecodedBudget)
{
    constexpr std::size_t kLen = 1u << 20;
    auto s                     = SectionPrefix(0, kNumBaseProbFeatures);
    s.Varint(1).Varint(1).Varint(0).Varint(kLen).Repeat('a', kLen).Varint(1);
    s.Varint(1).Varint(0).Varint(1).Chars("k"); // one bucket "k"
    s.U8(1).Varint(200).Repeat('\0', 25);       // 200 one-bit indices, all 0

    miopen::ai::lgbm::pcfg::SolverModel model;
    EXPECT_FALSE(Parse(s, model));
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
