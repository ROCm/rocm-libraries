// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "ProgramOptions.hpp"
#include "hipblaslt_bench_options.hpp"

using namespace TensileLite::Client;

namespace
{
    po::variables_map parsePolicy(std::initializer_list<char const*> values)
    {
        po::options_description options("Hybrid assignment policy");
        options.add_options()
            ("hybrid-assignment-policy", po::value<std::vector<std::string>>(), "Canonical policy")
            ("streamk-hybrid-mode", po::value<std::vector<int>>(), "Legacy alias");
        std::vector<char const*> args{"test"};
        args.insert(args.end(), values);
        return po::parse_command_line(static_cast<int>(args.size()), args.data(), options);
    }
}

TEST(HybridAssignmentPolicyTest, OmittedPolicyPreservesDefaultEncoding)
{
    EXPECT_EQ(resolveHybridAssignmentPolicies(parsePolicy({})), std::vector<int>{0});
}

TEST(HybridAssignmentPolicyTest, CanonicalAndLegacyValuesAgree)
{
    for(auto const& pair : {std::pair{"Default", "0"}, std::pair{"DynamicWorkQueue", "1"}, std::pair{"Auto", "2"}})
    {
        auto canonical = resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", pair.first}));
        auto legacy = resolveHybridAssignmentPolicies(parsePolicy({"--streamk-hybrid-mode", pair.second}));
        EXPECT_EQ(canonical, legacy);
        EXPECT_EQ(resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", pair.first,
                                                               "--streamk-hybrid-mode", pair.second})), canonical);
    }
}

TEST(HybridAssignmentPolicyTest, SweepOrderIsPreserved)
{
    auto values = resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", "Default,DynamicWorkQueue,Auto"}));
    EXPECT_EQ(values, (std::vector<int>{0, 1, 2}));
}

TEST(HybridAssignmentPolicyTest, ConflictingExplicitAliasesAreRejected)
{
    EXPECT_THROW(resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", "Default",
                                                              "--streamk-hybrid-mode", "1"})), std::invalid_argument);
    EXPECT_THROW(resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", "Auto",
                                                              "--streamk-hybrid-mode", "0"})), std::invalid_argument);
}

TEST(HybridAssignmentPolicyTest, InvalidValuesAreRejected)
{
    EXPECT_THROW(resolveHybridAssignmentPolicies(parsePolicy({"--hybrid-assignment-policy", "StaticGrid"})), std::invalid_argument);
    EXPECT_THROW(resolveHybridAssignmentPolicies(parsePolicy({"--streamk-hybrid-mode", "3"})), std::invalid_argument);
}

TEST(HybridAssignmentPolicyTest, BenchAliasesPreserveUnsetAndLegacyValues)
{
    using hipblaslt_bench_options::resolve_hybrid_assignment_policy;
    EXPECT_EQ(resolve_hybrid_assignment_policy("", ""), -1);
    EXPECT_EQ(resolve_hybrid_assignment_policy("Default", ""), 0);
    EXPECT_EQ(resolve_hybrid_assignment_policy("DynamicWorkQueue", "ON"), 1);
    EXPECT_EQ(resolve_hybrid_assignment_policy("Auto", "2"), 2);
    EXPECT_EQ(resolve_hybrid_assignment_policy("", "Off"), 0);
    EXPECT_EQ(resolve_hybrid_assignment_policy("", "auto"), 2);
    EXPECT_THROW(resolve_hybrid_assignment_policy("Default", "on"), std::invalid_argument);
    EXPECT_THROW(resolve_hybrid_assignment_policy("StaticGrid", ""), std::invalid_argument);
}
