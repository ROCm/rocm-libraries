// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// resolveClaimMode() has three inputs -- the enforce flag (not typed, false, true) and
// two booleans -- so the whole input space is twelve rows and every one is pinned
// here. A wrong answer in this function does not crash anything -- it quietly runs a
// lane in WARN that was meant to ENFORCE -- so no row is left to be inferred from its
// neighbours.

#include <gtest/gtest.h>

#include <array>
#include <bitset>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

#include "harness/bundle/HarnessPolicy.hpp"

using hipdnn_integration_tests::bundle::ClaimMode;
using hipdnn_integration_tests::bundle::ClaimModeRequest;
using hipdnn_integration_tests::bundle::parseEnforceClaimsValue;
using hipdnn_integration_tests::bundle::resolveClaimMode;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// Each refusal names the flag that makes it a refusal and the other does not, so a
// substring identifies which one fired without pinning the wording.
constexpr const char* WRITE_CONFLICT = "--write-support-claims";
constexpr const char* NO_ENGINE = "--test-engine";

struct Row
{
    std::optional<bool> enforce;
    bool writing;
    bool hasEngine;
    ClaimMode mode; // ignored when an error is expected
    const char* error; // nullptr: the run must start
};

// clang-format off
constexpr std::array<Row, 12> TRUTH_TABLE = {{
    // enforce       writing engine  mode               error
    {std::nullopt,  false,  false,  ClaimMode::WARN,    nullptr},
    {std::nullopt,  false,  true,   ClaimMode::ENFORCE, nullptr}, // the CI lane, default on
    {std::nullopt,  true,   false,  ClaimMode::WARN,    nullptr},
    {std::nullopt,  true,   true,   ClaimMode::WARN,    nullptr},
    {false,         false,  false,  ClaimMode::WARN,    nullptr},
    {false,         false,  true,   ClaimMode::WARN,    nullptr}, // the opt-out
    {false,         true,   false,  ClaimMode::WARN,    nullptr},
    {false,         true,   true,   ClaimMode::WARN,    nullptr},
    {true,          false,  false,  ClaimMode::WARN,    NO_ENGINE},
    {true,          false,  true,   ClaimMode::ENFORCE, nullptr}, // the CI lane, typed
    {true,          true,   false,  ClaimMode::WARN,    WRITE_CONFLICT},
    {true,          true,   true,   ClaimMode::WARN,    WRITE_CONFLICT},
}};
// clang-format on

// 0 not typed, 1 typed false, 2 typed true.
std::size_t enforceIndex(const Row& row)
{
    if(!row.enforce.has_value())
    {
        return 0U;
    }
    return *row.enforce ? 2U : 1U;
}

std::string describe(const Row& row)
{
    constexpr std::array<const char*, 3> ENFORCE_NAMES = {"unset", "false", "true"};
    return std::string("enforce=") + ENFORCE_NAMES.at(enforceIndex(row))
           + " writing=" + (row.writing ? "1" : "0") + " engine=" + (row.hasEngine ? "1" : "0");
}

std::size_t inputIndex(const Row& row)
{
    return (enforceIndex(row) * 4U) + (row.writing ? 2U : 0U) + (row.hasEngine ? 1U : 0U);
}

} // namespace

// Twelve rows is not enough on its own -- a duplicated row fills the count as well as
// a distinct one does. Twelve distinct inputs is every input.
TEST(TestClaimModeResolution, TruthTableCoversEveryInputCombination)
{
    std::bitset<12> seen;
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
        request.enforce = row.enforce;
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

// The substring check above only identifies a refusal if the other refusal's message
// does not also name that flag; this is what keeps it from passing on the wrong one.
TEST(TestClaimModeResolution, EachRefusalNamesOnlyItsOwnFlag)
{
    ClaimModeRequest writing;
    writing.enforce = true;
    writing.writing = true;
    writing.hasEngine = true;

    ClaimModeRequest noEngine;
    noEngine.enforce = true;

    const std::string writingError = resolveClaimMode(writing).error.value_or("");
    const std::string noEngineError = resolveClaimMode(noEngine).error.value_or("");
    ASSERT_FALSE(writingError.empty());
    ASSERT_FALSE(noEngineError.empty());

    EXPECT_EQ(noEngineError.find(WRITE_CONFLICT), std::string::npos);
    EXPECT_EQ(writingError.find(NO_ENGINE), std::string::npos);
}

TEST(TestClaimModeResolution, FlagValueAcceptsTrueAndFalse)
{
    EXPECT_TRUE(parseEnforceClaimsValue("true"));
    EXPECT_FALSE(parseEnforceClaimsValue("false"));
}

// A value that is not recognised must stop the run, not fall back to the default:
// the default is "on", so falling back is how a typo'd opt-out silently enforces.
TEST(TestClaimModeResolution, FlagValueRejectsAnythingElse)
{
    for(const char* value : {"", "TRUE", "False", "1", "0", "yes", "no", "off", "tru"})
    {
        SCOPED_TRACE(value);
        EXPECT_THROW(parseEnforceClaimsValue(value), std::invalid_argument);
    }
}

// NOLINTEND(readability-identifier-naming)
