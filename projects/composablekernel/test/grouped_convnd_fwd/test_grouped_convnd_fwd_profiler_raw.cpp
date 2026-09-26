// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"

TEST(GroupedConvProfilerRawInvocation, ListsAndRunsOneExactCandidate)
{
    using namespace ck::tensor_layout::convolution;
    using F16         = ck::half_t;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    const ck::utils::conv::ConvParam param{
        2, 32, 64, 4, 4, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    auto profile = [&](ck::index_t selected, bool list) {
        return ck::profiler::profile_grouped_conv_fwd_impl<2, NHWGC, GKYXC, NHWGK, F16, F16, F16>(
            0, 1, false, false, param, PassThrough{}, selected, list, true);
    };

    testing::internal::CaptureStdout();
    const bool listed    = profile(-1, true);
    const auto list_text = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(listed);

    std::vector<std::string> candidates;
    std::istringstream lines(list_text);
    for(std::string line; std::getline(lines, line);)
    {
        const auto prefix = "[" + std::to_string(candidates.size()) + "] ";
        if(line.rfind("[", 0) == 0)
        {
            ASSERT_EQ(line.rfind(prefix, 0), 0U) << line;
            const auto end = line.find(" (requested_split=");
            ASSERT_NE(end, std::string::npos) << line;
            candidates.push_back(line.substr(prefix.size(), end - prefix.size()));
        }
    }
    ASSERT_FALSE(candidates.empty());
    EXPECT_NE(list_text.find("Total: " + std::to_string(candidates.size()) + " valid instances"),
              std::string::npos);

    testing::internal::CaptureStdout();
    const bool ran      = profile(0, false);
    const auto run_text = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(ran);
    const auto raw_prefix = std::string("Raw invocation: ");
    const auto raw_begin  = run_text.find(raw_prefix);
    ASSERT_NE(raw_begin, std::string::npos);
    const auto raw_end = run_text.find('\n', raw_begin);
    const auto record  = run_text.substr(raw_begin, raw_end - raw_begin);
    const float raw_ms = std::stof(record.substr(raw_prefix.size()));
    EXPECT_GT(raw_ms, 0);
    EXPECT_TRUE(std::isfinite(raw_ms));
    EXPECT_NE(record.find("ms, instance 0, requested_split=1, effective_split=1, "
                          "policy=hot-reuse, repeats=50, " +
                          candidates.front()),
              std::string::npos);
    EXPECT_EQ(run_text.find(raw_prefix, raw_end), std::string::npos);
    EXPECT_EQ(run_text.find("Perf:"), std::string::npos);
    EXPECT_EQ(run_text.find("Best configuration parameters:"), std::string::npos);
}
