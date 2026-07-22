// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the client string<->enum parsers declared in
// hipblaslt_datatype2string.hpp. They never touch the GPU, so they can run in a
// CI lane without a device and give fast feedback on the parser/mapping code
// that previously could only be exercised through a full matmul run.
//
// Motivated by the PR #6514 review (davidd-amd), observation #2 ("Do we have a
// plan to start adding unit tests for functions that we are changing that can be
// unit tested?"). Follow-up tracked as AIHPBLAS-3550.

#include "hipblaslt_datatype2string.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace
{
    // Every init string the client accepts today, paired with its enum value.
    // Keep this in sync with string2hipblaslt_initialization().
    const std::vector<std::pair<std::string, hipblaslt_initialization>>& known_init_modes()
    {
        static const std::vector<std::pair<std::string, hipblaslt_initialization>> modes = {
            {"rand_int", hipblaslt_initialization::rand_int},
            {"trig_float", hipblaslt_initialization::trig_float},
            {"hpl", hipblaslt_initialization::hpl},
            {"special", hipblaslt_initialization::special},
            {"zero", hipblaslt_initialization::zero},
            {"norm_dist", hipblaslt_initialization::norm_dist},
            {"uniform_01", hipblaslt_initialization::uniform_01},
            {"integer_exact", hipblaslt_initialization::integer_exact},
            {"fp16_accumulator_probe", hipblaslt_initialization::fp16_accumulator_probe},
            {"inf", hipblaslt_initialization::inf},
            {"neg_zero", hipblaslt_initialization::neg_zero},
            {"neg_inf", hipblaslt_initialization::neg_inf},
            {"nan", hipblaslt_initialization::nan},
            {"norm_dist_one_special", hipblaslt_initialization::norm_dist_one_special},
            {"uniform_low_precision", hipblaslt_initialization::uniform_low_precision},
        };
        return modes;
    }
}

TEST(ClientInitParser, KnownStringsMapToExpectedEnum)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization(name)),
                  static_cast<int>(expected))
            << "string2hipblaslt_initialization(\"" << name << "\")";
    }
}

TEST(ClientInitParser, RoundTripsThroughInitialization2String)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_STREQ(hipblaslt_initialization2string(string2hipblaslt_initialization(name)),
                     name.c_str());
    }
}

// Current (develop) behavior: an unrecognized init string silently maps to the
// integer sentinel 0, which is NOT a valid hipblaslt_initialization enumerator
// (the enum starts at 111). This is exactly the footgun called out in the PR
// #6514 review, observation #3. AIHPBLAS-3551 replaces this with a
// std::optional-returning, noexcept parser, at which point this expectation is
// meant to flip.
TEST(ClientInitParser, UnknownStringMapsToZeroSentinel_PreAIHPBLAS3551)
{
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("not_a_real_init_mode")), 0);
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("")), 0);
}

TEST(ClientActivationParser, KnownStringsMapAndRoundTrip)
{
    const std::vector<std::pair<std::string, hipblaslt_activation_type>> known = {
        {"none", hipblaslt_activation_type::none},
        {"gelu", hipblaslt_activation_type::gelu},
        {"relu", hipblaslt_activation_type::relu},
        {"swish", hipblaslt_activation_type::swish},
        {"clamp", hipblaslt_activation_type::clamp},
        {"sigmoid", hipblaslt_activation_type::sigmoid},
    };
    for(const auto& [name, expected] : known)
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_activation_type(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_activation_type_to_string(expected), name.c_str());
    }
}

// none == 0 is a valid activation, so the unknown sentinel here is -1, not 0.
TEST(ClientActivationParser, UnknownStringMapsToNegativeSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type("not_an_activation")), -1);
}

TEST(ClientBiasSourceParser, KnownStringsMapAndRoundTrip)
{
    const std::vector<std::pair<std::string, hipblaslt_bias_source>> known = {
        {"a", hipblaslt_bias_source::a},
        {"b", hipblaslt_bias_source::b},
        {"d", hipblaslt_bias_source::d},
    };
    for(const auto& [name, expected] : known)
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_bias_source(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_bias_source_to_string(expected), name.c_str());
    }
}

// bias_source values start at 1, so 0 is the unknown sentinel.
TEST(ClientBiasSourceParser, UnknownStringMapsToZeroSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source("not_a_bias_source")), 0);
}
