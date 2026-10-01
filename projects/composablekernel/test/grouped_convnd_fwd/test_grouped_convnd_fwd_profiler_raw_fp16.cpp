// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"

#ifdef CK_ENABLE_FP16
TEST(GroupedConvProfilerRawInvocation, ListsUniqueCandidatesAndRunsLast)
{
    using namespace ck::tensor_layout::convolution;
    using F16         = ck::half_t;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using DeviceOp    = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                                      NHWGC,
                                                                                      GKYXC,
                                                                                      ck::Tuple<>,
                                                                                      NHWGK,
                                                                                      F16,
                                                                                      F16,
                                                                                      ck::Tuple<>,
                                                                                      F16,
                                                                                      PassThrough,
                                                                                      PassThrough,
                                                                                      PassThrough>;

    const ck::utils::conv::ConvParam param{
        2, 1, 1, 64, 64, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    auto profile = [&](ck::index_t selected, bool list) {
        return ck::profiler::profile_grouped_conv_fwd_impl<2, NHWGC, GKYXC, NHWGK, F16, F16, F16>(
            0, 1, false, false, param, PassThrough{}, selected, list, true);
    };

    const auto input_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<NHWGC>(param);
    const auto weight_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<GKYXC>(param);
    const auto output_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<NHWGK>(param);
    auto dimensions = [](const auto& values) {
        std::array<ck::index_t, 5> result{};
        ck::ranges::copy(values, result.begin());
        return result;
    };
    const std::array<ck::index_t, 2> stride{1, 1};
    const std::array<ck::index_t, 2> dilation{1, 1};
    const std::array<ck::index_t, 2> pad{1, 1};
    const auto ops = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();
    std::size_t supported_count = 0;
    for(const auto& op : ops)
    {
        auto argument = op->MakeArgumentPointer(nullptr,
                                                nullptr,
                                                {},
                                                nullptr,
                                                dimensions(input_desc.GetLengths()),
                                                dimensions(input_desc.GetStrides()),
                                                dimensions(weight_desc.GetLengths()),
                                                dimensions(weight_desc.GetStrides()),
                                                {},
                                                {},
                                                dimensions(output_desc.GetLengths()),
                                                dimensions(output_desc.GetStrides()),
                                                stride,
                                                dilation,
                                                pad,
                                                pad,
                                                PassThrough{},
                                                PassThrough{},
                                                PassThrough{});
        // Support probing needs a non-null workspace, but never launches kernels.
        int workspace_dummy = 0;
        if(op->GetWorkSpaceSize(argument.get()) != 0)
            op->SetWorkSpacePointer(argument.get(), &workspace_dummy);
        if(op->IsSupportedArgument(argument.get()))
            ++supported_count;
    }

    testing::internal::CaptureStdout();
    const bool listed    = profile(-1, true);
    const auto list_text = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(listed);

    std::vector<std::string> candidates;
    std::istringstream lines(list_text);
    for(std::string line; std::getline(lines, line);)
    {
        if(line.rfind("[", 0) == 0)
        {
            const auto prefix = "[" + std::to_string(candidates.size()) + "] ";
            ASSERT_EQ(line.rfind(prefix, 0), 0U) << line;
            const auto name = line.substr(prefix.size());
            candidates.push_back(name);
        }
    }
    // Distinct instances may share a type string, so the count (not name uniqueness)
    // detects a re-listed first candidate.
    ASSERT_EQ(candidates.size(), supported_count);
    ASSERT_GT(candidates.size(), 1U) << "The regression requires a nonzero selection index";

    const auto selected = static_cast<ck::index_t>(candidates.size() - 1);
    testing::internal::CaptureStdout();
    const bool ran      = profile(selected, false);
    const auto run_text = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(ran);
    const std::string raw_prefix = "Raw invocation: ";
    const auto raw_begin         = run_text.find(raw_prefix);
    ASSERT_NE(raw_begin, std::string::npos);
    const auto raw_end = run_text.find('\n', raw_begin);
    const auto record  = run_text.substr(raw_begin, raw_end - raw_begin);
    const float raw_ms = std::stof(record.substr(raw_prefix.size()));
    EXPECT_GT(raw_ms, 0);
    EXPECT_TRUE(std::isfinite(raw_ms));

    const std::string instance_prefix = "instance ";
    const auto instance_begin         = record.find(instance_prefix);
    ASSERT_NE(instance_begin, std::string::npos);
    EXPECT_EQ(std::stoi(record.substr(instance_begin + instance_prefix.size())), selected);
    const auto& expected_name = candidates.back();
    ASSERT_GE(record.size(), expected_name.size());
    EXPECT_EQ(record.substr(record.size() - expected_name.size()), expected_name);
    EXPECT_EQ(run_text.find(raw_prefix, raw_end), std::string::npos);
    EXPECT_EQ(run_text.find("Perf:"), std::string::npos);
    EXPECT_EQ(run_text.find("Best configuration parameters:"), std::string::npos);
}
#endif
