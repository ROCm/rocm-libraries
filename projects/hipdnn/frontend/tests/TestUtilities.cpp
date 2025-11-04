// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::test_utilities;

TEST(TestUtilities, FindCommonShapeValid)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 1}, {1, 1, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, FindCommonShapeEmptyInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeIncompatibleShapes)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestUtilities, FindCommonShapeSingleInput)
{
    std::vector<std::vector<int64_t>> inputShapes = {{1, 2, 3}};
    std::vector<int64_t> commonShape;

    auto error = findCommonShape(inputShapes, commonShape);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(commonShape, (std::vector<int64_t>{1, 2, 3}));
}

TEST(TestUtilities, InitializeFrontendLoggingReturnsCorrectly)
{
    ScopedEnvironmentVariableSetter guard("HIPDNN_LOG_LEVEL", "info");

    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(nullptr), -1);
    EXPECT_EQ(hipdnn_frontend::initializeFrontendLogging(), 0);
}
