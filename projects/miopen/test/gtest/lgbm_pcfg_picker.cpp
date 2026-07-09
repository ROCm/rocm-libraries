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

// CPU-only unit tests for the per-solver perf-config picker. No GPU required:
// the test rebuilds the 27-feature problem+GPU prefix from each vector's raw
// inputs (mirroring model_fields.build_X), looks up the bucket's candidate
// configs from the shipped catalog, and asserts the scoring + argmax via
// ScorePickForTest reproduces the exported expected descriptor.
//
// This validates the C++ feature math (log1p/derived), the per-solver predict
// dispatch, and the argmax — exactly the parity the exporter guaranteed in
// Python. It does NOT exercise the live ProblemDescription getters (that needs a
// GPU + real conv problem); see the plan's GPU-validation item.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_pick.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>

#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using miopen::ai::lgbm::pcfg::kNumBaseProbFeatures;
using miopen::ai::lgbm::pcfg::LgbmPcfgMetadata;
using miopen::ai::lgbm::pcfg::ScorePickForTest;

inline double Log1pAbs(double v) { return std::log1p(std::fabs(v)); }

// Fixed gfx_code vocabulary, matching GfxCode in lgbm_pcfg_pick.cpp /
// model_fields.build_X gfx_order. Unknown -> -1.
int GfxCode(const std::string& gfx_id)
{
    static const std::array<const char*, 10> kGfxOrder = {"gfx906",
                                                          "gfx90a",
                                                          "gfx942",
                                                          "gfx950",
                                                          "gfx1100",
                                                          "gfx1101",
                                                          "gfx1102",
                                                          "gfx1105",
                                                          "gfx1151",
                                                          "gfx1201"};
    for(int i = 0; i < static_cast<int>(kGfxOrder.size()); ++i)
        if(gfx_id == kGfxOrder[static_cast<std::size_t>(i)])
            return i;
    return -1;
}

// Rebuild the prefix from a vector's raw problem_inputs + gpu_inputs, matching
// FillProblemPrefix in lgbm_pcfg_pick.cpp (and model_fields.build_X). When
// with_gfx_code, append the trailing gfx_code derived from llvm_target.
std::vector<double> BuildPrefix(const nlohmann::json& pin,
                                const nlohmann::json& gin,
                                const std::string& llvm_target,
                                bool with_gfx_code)
{
    auto P = [&](const char* k) { return pin.at(k).get<double>(); };

    const double channels = P("channels");
    const double height   = P("height");
    const double width    = P("width");
    const double out_ch   = P("output_channels");
    const double fil_y    = P("filter_height_y");
    const double fil_x    = P("filter_width_x");
    const double pad_h    = P("pad_height");
    const double pad_w    = P("pad_width");
    const double stride_h = P("stride_height");
    const double stride_w = P("stride_width");
    const double dil_h    = P("dilation_height");
    const double dil_w    = P("dilation_width");
    const double groups   = P("groups");
    const double batch    = P("n_mini_batch_size");
    const double dir      = P("direction");

    const std::string dt = pin.at("data_type").get<std::string>();
    const double dtype_code =
        dt == "fp32" ? 0.0 : dt == "fp16" ? 1.0 : dt == "bf16" ? 2.0 : dt == "int8" ? 3.0 : -1.0;

    const double g       = groups < 1.0 ? 1.0 : groups;
    const double cpg     = channels / g;
    const double opg     = out_ch / g;
    const double farea   = fil_y * fil_x;
    const double spatial = height * width;
    const double bxs     = batch * height * width;

    std::vector<double> f;
    f.reserve(kNumBaseProbFeatures + 1);
    f.push_back(Log1pAbs(channels));
    f.push_back(Log1pAbs(height));
    f.push_back(Log1pAbs(width));
    f.push_back(Log1pAbs(out_ch));
    f.push_back(Log1pAbs(fil_y));
    f.push_back(Log1pAbs(fil_x));
    f.push_back(Log1pAbs(pad_h));
    f.push_back(Log1pAbs(pad_w));
    f.push_back(Log1pAbs(stride_h));
    f.push_back(Log1pAbs(stride_w));
    f.push_back(Log1pAbs(dil_h));
    f.push_back(Log1pAbs(dil_w));
    f.push_back(Log1pAbs(groups));
    f.push_back(Log1pAbs(batch));
    f.push_back(Log1pAbs(cpg));
    f.push_back(Log1pAbs(opg));
    f.push_back(Log1pAbs(farea));
    f.push_back(Log1pAbs(spatial));
    f.push_back(Log1pAbs(bxs));
    f.push_back(gin.at("cu_count").get<double>());
    f.push_back(gin.at("wave_size").get<double>());
    f.push_back(gin.at("lds_size_per_workgroup_kb").get<double>());
    f.push_back(gin.at("l2_cache_total_kb").get<double>());
    f.push_back(gin.at("boost_clock_mhz").get<double>());
    f.push_back(gin.at("vram_bytes").get<double>());
    f.push_back(dir);
    f.push_back(dtype_code);
    if(with_gfx_code)
        f.push_back(static_cast<double>(GfxCode(llvm_target)));
    return f;
}

class CPU_LgbmPcfgPickerFixture : public ::testing::Test
{
protected:
    nlohmann::json vectors_by_solver;
    nlohmann::json catalog;
    const LgbmPcfgMetadata& meta = LgbmPcfgMetadata::Get();

    void SetUp() override
    {
        if(!meta.IsReady())
            GTEST_SKIP() << "lgbm_pcfg metadata unavailable; picker disabled in this build";

        const auto vpath = miopen::GetSystemDbPath() / "lgbm_pcfg_test_vectors.json";
        const auto cpath = miopen::GetSystemDbPath() / "lgbm_pcfg_catalog.json";
        if(!miopen::fs::exists(vpath) || !miopen::fs::exists(cpath))
            GTEST_SKIP() << "fixture/catalog not found in " << miopen::GetSystemDbPath().string();
        std::ifstream(vpath.string()) >> vectors_by_solver;
        std::ifstream(cpath.string()) >> catalog;
    }
};

TEST_F(CPU_LgbmPcfgPickerFixture, ReproducesExportedRanking)
{
    int total      = 0;
    int top1_match = 0;
    int rank_match = 0;

    for(auto sit = vectors_by_solver.begin(); sit != vectors_by_solver.end(); ++sit)
    {
        const std::string solver = sit.key();
        const auto cat_it        = catalog.find(solver);
        if(cat_it == catalog.end())
            continue;
        const auto& sbuckets = cat_it->at("buckets");

        // Whether this solver's prefix carries the trailing gfx_code (the loaded
        // model knows; BuildPrefix must match its length exactly).
        const auto* model        = meta.Find(solver);
        const bool with_gfx_code = (model != nullptr) && model->has_gfx_code;

        for(const auto& v : sit.value())
        {
            const std::string bucket = v.at("bucket").get<std::string>();
            const auto bit           = sbuckets.find(bucket);
            if(bit == sbuckets.end())
                continue;

            std::vector<std::string> descs;
            std::vector<std::vector<double>> args;
            for(const auto& c : bit.value())
            {
                descs.push_back(c.at("desc").get<std::string>());
                std::vector<double> a;
                for(const auto& x : c.at("args"))
                    a.push_back(x.is_null() ? std::numeric_limits<double>::quiet_NaN()
                                            : x.get<double>());
                args.push_back(std::move(a));
            }

            const std::string llvm_target = v.at("llvm_target").get<std::string>();
            const auto prefix =
                BuildPrefix(v.at("problem_inputs"), v.at("gpu_inputs"), llvm_target, with_gfx_code);
            const auto ranked = ScorePickForTest(solver, prefix, descs, args);
            ++total;
            if(ranked.empty())
                continue;

            // Top-1 must equal expected_desc (the argmax contract).
            if(ranked.front() == v.at("expected_desc").get<std::string>())
                ++top1_match;

            // The C ranking's head must reproduce expected_ranked exactly (the
            // first-valid walk consumes this order). expected_ranked is the top-N
            // by score; compare element-wise up to its length.
            bool rank_ok = true;
            if(v.contains("expected_ranked"))
            {
                const auto& exp = v.at("expected_ranked");
                for(std::size_t i = 0; i < exp.size(); ++i)
                {
                    if(i >= ranked.size() || ranked[i] != exp[i].get<std::string>())
                    {
                        rank_ok = false;
                        break;
                    }
                }
            }
            if(rank_ok)
                ++rank_match;
        }
    }

    ASSERT_GT(total, 0);
    EXPECT_EQ(top1_match, total) << top1_match << "/" << total << " top-1 picks reproduced";
    EXPECT_EQ(rank_match, total) << rank_match << "/" << total << " ranked orders reproduced";
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
