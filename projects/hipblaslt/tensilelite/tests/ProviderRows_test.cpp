// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ProviderRows.hpp>

using namespace TensileLite;

// Direct calls, so the TensileLite coverage lane (tensilelite-tests) executes
// every row of ProviderRows.cpp. The hipBLASLt ProviderRows gtest links the
// same code into a different binary.

TEST(ProviderRows, skipsTheNamedPredicateFamilies)
{
    EXPECT_FALSE(skipsProviderRow(ProviderRows::All, "EqualityMatching"));
    EXPECT_FALSE(skipsProviderRow(ProviderRows::All, "RangeMatching"));

    EXPECT_TRUE(skipsProviderRow(ProviderRows::EqualityOnly, "RangeMatching"));
    EXPECT_TRUE(skipsProviderRow(ProviderRows::EqualityOnly, "PredictionMatching"));
    EXPECT_TRUE(skipsProviderRow(ProviderRows::EqualityOnly, "GridBasedMatching"));
    EXPECT_TRUE(skipsProviderRow(ProviderRows::EqualityOnly, "FreeSizeMatching"));
    EXPECT_TRUE(skipsProviderRow(ProviderRows::EqualityOnly, "ExperimentalMLP"));
    EXPECT_FALSE(skipsProviderRow(ProviderRows::EqualityOnly, "EqualityMatching"));

    EXPECT_TRUE(skipsProviderRow(ProviderRows::ExceptEquality, "EqualityMatching"));
    EXPECT_FALSE(skipsProviderRow(ProviderRows::ExceptEquality, "RangeMatching"));
    EXPECT_FALSE(skipsProviderRow(ProviderRows::ExceptEquality, "ExperimentalMLP"));
}

TEST(ProviderRows, scopeRestoresThePreviousRows)
{
    EXPECT_EQ(currentProviderRows(), ProviderRows::All);
    {
        ProviderRowsScope outer(ProviderRows::EqualityOnly);
        EXPECT_EQ(currentProviderRows(), ProviderRows::EqualityOnly);
        {
            ProviderRowsScope inner(ProviderRows::ExceptEquality);
            EXPECT_EQ(currentProviderRows(), ProviderRows::ExceptEquality);
        }
        EXPECT_EQ(currentProviderRows(), ProviderRows::EqualityOnly);
    }
    EXPECT_EQ(currentProviderRows(), ProviderRows::All);
}
