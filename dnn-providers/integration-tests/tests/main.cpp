// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    // InitGoogleMock also initializes GTest, and additionally installs the
    // leaked-mock detector — without it a mock that outlives its expectations
    // fails silently.
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
