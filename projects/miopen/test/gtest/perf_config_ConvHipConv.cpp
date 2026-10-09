// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

#include "get_handle.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

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

#endif // MIOPEN_USE_HIPCONV
