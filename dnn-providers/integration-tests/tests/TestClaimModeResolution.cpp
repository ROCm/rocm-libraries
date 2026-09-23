// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// resolveClaimMode() has four boolean inputs, so the whole input space is sixteen
// rows and every one is pinned here. A wrong answer in this function does not crash
// anything -- it quietly runs a lane in WARN that was meant to ENFORCE -- so no row
// is left to be inferred from its neighbours.

#include <gtest/gtest.h>

#include <array>
#include <bitset>
#include <cstddef>
#include <string>

#include "harness/bundle/HarnessPolicy.hpp"

using hipdnn_integration_tests::bundle::ClaimMode;
using hipdnn_integration_tests::bundle::ClaimModeRequest;
using hipdnn_integration_tests::bundle::resolveClaimMode;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// Each refusal names the flag that makes it a refusal and no other refusal does, so
// a substring identifies which of the three fired without pinning the wording.
constexpr const char* REFUSED_CONFLICT = "--no-enforce-support-claims";
constexpr const char* WRITE_CONFLICT = "--write-support-claims";
constexpr const char* NO_ENGINE = "--test-engine";

struct Row
{
    bool enforceAsked;
    bool enforceRefused;
    bool writing;
    bool hasEngine;
    ClaimMode mode; // ignored when an error is expected
    const char* error; // nullptr: the run must start
};

// clang-format off
constexpr std::array<Row, 16> TRUTH_TABLE = {{
    // asked  refused writing engine  mode               error
    {false,  false,  false,  false,  ClaimMode::WARN,    nullptr},
    {false,  false,  false,  true,   ClaimMode::ENFORCE, nullptr}, // the CI lane, default on
    {false,  false,  true,   false,  ClaimMode::WARN,    nullptr},
    {false,  false,  true,   true,   ClaimMode::WARN,    nullptr},
    {false,  true,   false,  false,  ClaimMode::WARN,    nullptr},
    {false,  true,   false,  true,   ClaimMode::WARN,    nullptr},
    {false,  true,   true,   false,  ClaimMode::WARN,    nullptr},
    {false,  true,   true,   true,   ClaimMode::WARN,    nullptr},
    {true,   false,  false,  false,  ClaimMode::WARN,    NO_ENGINE},
    {true,   false,  false,  true,   ClaimMode::ENFORCE, nullptr}, // the CI lane, typed
    {true,   false,  true,   false,  ClaimMode::WARN,    WRITE_CONFLICT},
    {true,   false,  true,   true,   ClaimMode::WARN,    WRITE_CONFLICT},
    {true,   true,   false,  false,  ClaimMode::WARN,    REFUSED_CONFLICT},
    {true,   true,   false,  true,   ClaimMode::WARN,    REFUSED_CONFLICT},
    {true,   true,   true,   false,  ClaimMode::WARN,    REFUSED_CONFLICT},
    {true,   true,   true,   true,   ClaimMode::WARN,    REFUSED_CONFLICT},
}};
// clang-format on

std::string describe(const Row& row)
{
    return std::string("asked=") + (row.enforceAsked ? "1" : "0")
           + " refused=" + (row.enforceRefused ? "1" : "0")
           + " writing=" + (row.writing ? "1" : "0") + " engine=" + (row.hasEngine ? "1" : "0");
}

std::size_t inputIndex(const Row& row)
{
    return (row.enforceAsked ? 8U : 0U) | (row.enforceRefused ? 4U : 0U) | (row.writing ? 2U : 0U)
           | (row.hasEngine ? 1U : 0U);
}

} // namespace

// Sixteen rows is not enough on its own -- std::array zero-fills a missing row, and a
// duplicated row fills the count too. Sixteen distinct inputs is every input.
TEST(TestClaimModeResolution, TruthTableCoversEveryInputCombination)
{
    std::bitset<16> seen;
    for(const Row& row : TRUTH_TABLE)
    {
        EXPECT_FALSE(seen.test(inputIndex(row))) << "duplicate row: " << describe(row);
        seen.set(inputIndex(row));
    }
    EXPECT_TRUE(seen.all());
}

TEST(TestClaimModeResolution, EveryInputCombinationResolvesAsTabled)
{
    for(const Row& row : TRUTH_TABLE)
    {
        SCOPED_TRACE(describe(row));

        ClaimModeRequest request;
        request.enforceAsked = row.enforceAsked;
        request.enforceRefused = row.enforceRefused;
        request.writing = row.writing;
        request.hasEngine = row.hasEngine;

        const auto resolved = resolveClaimMode(request);

        if(row.error == nullptr)
        {
            EXPECT_FALSE(resolved.error.has_value()) << resolved.error.value_or("");
            EXPECT_EQ(resolved.mode, row.mode);
        }
        else
        {
            ASSERT_TRUE(resolved.error.has_value());
            const std::string error = resolved.error.value_or("");
            EXPECT_NE(error.find(row.error), std::string::npos) << error;
        }
    }
}

// The substring check above only identifies a refusal if no other refusal's message
// also names that flag; this is what keeps it from passing on the wrong one.
TEST(TestClaimModeResolution, EachRefusalNamesOnlyItsOwnFlag)
{
    ClaimModeRequest refused;
    refused.enforceAsked = true;
    refused.enforceRefused = true;
    refused.hasEngine = true;

    ClaimModeRequest writing;
    writing.enforceAsked = true;
    writing.writing = true;
    writing.hasEngine = true;

    ClaimModeRequest noEngine;
    noEngine.enforceAsked = true;

    const std::string refusedError = resolveClaimMode(refused).error.value_or("");
    const std::string writingError = resolveClaimMode(writing).error.value_or("");
    const std::string noEngineError = resolveClaimMode(noEngine).error.value_or("");
    ASSERT_FALSE(refusedError.empty());
    ASSERT_FALSE(writingError.empty());
    ASSERT_FALSE(noEngineError.empty());

    EXPECT_EQ(writingError.find(REFUSED_CONFLICT), std::string::npos);
    EXPECT_EQ(noEngineError.find(REFUSED_CONFLICT), std::string::npos);
    EXPECT_EQ(refusedError.find(WRITE_CONFLICT), std::string::npos);
    EXPECT_EQ(noEngineError.find(WRITE_CONFLICT), std::string::npos);
    EXPECT_EQ(refusedError.find(NO_ENGINE), std::string::npos);
    EXPECT_EQ(writingError.find(NO_ENGINE), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
