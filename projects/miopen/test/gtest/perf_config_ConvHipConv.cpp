// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

#include "get_handle.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#endif

using Config   = miopen::solver::conv::PerformanceConfigConvHipConv;
using TestCase = miopen::unit_tests::ConvTestCase;

// A kernel label contains commas, which field-wise perf-config serialization would split on.
TEST(CPU_PerfConfig_ConvHipConv_NONE, SerializeRoundTrip)
{
    Config stored;
    stored.descriptor = "direct[tile_size_k=256,tile_size_n=1,tile_size_h=16,tile_size_w=16]";

    Config loaded;
    ASSERT_TRUE(loaded.Deserialize(stored.ToString()));
    EXPECT_EQ(loaded.descriptor, stored.descriptor);
}

// A perf-db record selects its config only under the hipconv minor version that wrote it.
TEST(GPU_PerfConfig_ConvHipConv_FP16, VersionStamp)
{
    if(!IsTestSupportedByDevice(Gpu::gfx950 | Gpu::gfx125X))
        GTEST_SKIP();

    constexpr auto type   = miopenHalf;
    constexpr auto layout = miopenTensorNHWC;
    // clang-format off
    const auto test_case = TestCase{{type, layout, {7, 64, 8, 8}}, {type, layout, {128, 64, 3, 3}}, type, {{1, 1}, {1, 1}, {1, 1}}};
    // clang-format on
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::Forward);

    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const auto solver = miopen::solver::conv::ConvHipConv{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));

    // The index a stored record selects, or -1 if it is rejected.
    const auto load = [&](const std::string& stored) {
        Config config;
        config.Deserialize(stored);
        return solver.IsValidPerformanceConfig(ctx, problem, config) ? config.index : -1;
    };

    // IsValidPerformanceConfig sets config.descriptor from config.index.
    int configs = 0;
    for(;; ++configs)
    {
        Config config;
        config.index = configs;
        if(!solver.IsValidPerformanceConfig(ctx, problem, config))
            break;
        const auto record = config.ToString();
        const auto colon  = record.find(':');
        ASSERT_NE(colon, std::string::npos) << record;
        const auto version = record.substr(0, colon);
        EXPECT_NE(version, "unknown");
        const auto body = record.substr(colon + 1);

        EXPECT_EQ(load(record), configs) << record;
        EXPECT_EQ(load("v999.999:" + body), -1) << record;
        EXPECT_EQ(load(version + "0:" + body), -1) << record;
        EXPECT_EQ(load("unknown:" + body), -1) << record;
        EXPECT_EQ(load(body), -1) << record;
    }
    // Config 0 alone would not catch a record resolving to the default config.
    EXPECT_GE(configs, 2);
}

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
// The perf-config picker's ConvHipConv candidates for this device carry this build's config
// version. IsValidPerformanceConfig rejects a record of any other version, so a model left on
// an older version after a hipconv config bump would run the default config on every problem.
TEST(GPU_PerfConfig_ConvHipConv_FP16, PickerCatalogVersion)
{
    if(!IsTestSupportedByDevice(Gpu::gfx950 | Gpu::gfx125X))
        GTEST_SKIP();

    const auto& meta  = miopen::ai::lgbm::pcfg::LgbmPcfgMetadata::Get();
    const auto* model = meta.IsReady() ? meta.Find("ConvHipConv") : nullptr;
    if(model == nullptr)
        GTEST_SKIP() << "no ConvHipConv perf-config model installed";

    constexpr auto type   = miopenHalf;
    constexpr auto layout = miopenTensorNHWC;
    // clang-format off
    const auto test_case = TestCase{{type, layout, {7, 64, 8, 8}}, {type, layout, {128, 64, 3, 3}}, type, {{1, 1}, {1, 1}, {1, 1}}};
    // clang-format on
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::Forward);

    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const auto record =
        miopen::solver::conv::ConvHipConv{}.GetDefaultPerformanceConfig(ctx, problem).ToString();
    const auto colon = record.find(':');
    ASSERT_NE(colon, std::string::npos) << record;
    const auto version = record.substr(0, colon + 1);

    const auto gfx_id = handle.GetDeviceName();
    int candidates    = 0;
    for(const auto& [key, bucket] : model->buckets)
    {
        if(key.substr(0, key.find('|')) != gfx_id)
            continue;
        for(const auto& candidate : bucket)
        {
            // An empty desc stands for the solver's default config.
            if(candidate.desc.empty())
                continue;
            EXPECT_TRUE(candidate.desc.starts_with(version)) << key << ": " << candidate.desc;
            ++candidates;
        }
    }
    if(candidates == 0)
        GTEST_SKIP() << "no ConvHipConv picker candidates for " << gfx_id;
}
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#endif // MIOPEN_USE_HIPCONV
