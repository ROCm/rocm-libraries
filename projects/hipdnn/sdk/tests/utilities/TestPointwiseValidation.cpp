// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestPointwiseValidation, UnaryModesClassifiedCorrectly)
{
    // Test known unary operations
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::RELU_FWD));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::SIGMOID_FWD));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::TANH_FWD));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::ABS));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::NEG));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::EXP));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::LOG));
    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::SQRT));

    // Test that binary operations are NOT unary
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::ADD));
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::SUB));
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::MUL));
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::RELU_BWD));
}

TEST(TestPointwiseValidation, BinaryModesClassifiedCorrectly)
{
    // Test known binary operations
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::ADD));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::SUB));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::MUL));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::DIV));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::MAX_OP));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::MIN_OP));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::RELU_BWD));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::SIGMOID_BWD));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::TANH_BWD));

    // Test that unary operations are NOT binary
    EXPECT_FALSE(isBinaryPointwiseMode(PointwiseMode::RELU_FWD));
    EXPECT_FALSE(isBinaryPointwiseMode(PointwiseMode::SIGMOID_FWD));
    EXPECT_FALSE(isBinaryPointwiseMode(PointwiseMode::TANH_FWD));
}

TEST(TestPointwiseValidation, TernaryModesClassifiedCorrectly)
{
    // Test known ternary operations
    EXPECT_TRUE(isTernaryPointwiseMode(PointwiseMode::BINARY_SELECT));

    // Test that unary and binary operations are NOT ternary
    EXPECT_FALSE(isTernaryPointwiseMode(PointwiseMode::RELU_FWD));
    EXPECT_FALSE(isTernaryPointwiseMode(PointwiseMode::ADD));
}

TEST(TestPointwiseValidation, MutualExclusivityOfCategories)
{
    // Iterate through all pointwise modes and verify mutual exclusivity
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        bool isUnary = isUnaryPointwiseMode(mode);
        bool isBinary = isBinaryPointwiseMode(mode);
        bool isTernary = isTernaryPointwiseMode(mode);

        // Count how many categories this mode belongs to
        int categoryCount = (isUnary ? 1 : 0) + (isBinary ? 1 : 0) + (isTernary ? 1 : 0);

        // Each mode should belong to exactly one category
        EXPECT_LE(categoryCount, 1)
            << "Mode " << static_cast<int>(mode) << " belongs to multiple categories";
    }
}

TEST(TestPointwiseValidation, ImplementedUnaryModesAreSubset)
{
    // All implemented unary modes must be classified as unary
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        if(isImplementedUnaryPointwiseMode(mode))
        {
            EXPECT_TRUE(isUnaryPointwiseMode(mode))
                << "Implemented unary mode " << static_cast<int>(mode)
                << " should be classified as unary";
        }
    }
}

TEST(TestPointwiseValidation, ImplementedBinaryModesAreSubset)
{
    // All implemented binary modes must be classified as binary
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        if(isImplementedBinaryPointwiseMode(mode))
        {
            EXPECT_TRUE(isBinaryPointwiseMode(mode))
                << "Implemented binary mode " << static_cast<int>(mode)
                << " should be classified as binary";
        }
    }
}

TEST(TestPointwiseValidation, ImplementedTernaryModesAreSubset)
{
    // All implemented ternary modes must be classified as ternary
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        if(isImplementedTernaryPointwiseMode(mode))
        {
            EXPECT_TRUE(isTernaryPointwiseMode(mode))
                << "Implemented ternary mode " << static_cast<int>(mode)
                << " should be classified as ternary";
        }
    }
}

TEST(TestPointwiseValidation, KnownImplementedUnaryOperations)
{
    // Define which unary operations are currently implemented
    std::set<PointwiseMode> expectedImplementedUnary = {PointwiseMode::RELU_FWD,
                                                        PointwiseMode::SIGMOID_FWD,
                                                        PointwiseMode::TANH_FWD,
                                                        PointwiseMode::ABS,
                                                        PointwiseMode::NEG};

    // Check all unary modes
    for(size_t i = 0; i < POINTWISE_MODE_COUNT; ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        // Only check modes that are classified as unary
        if(!isUnaryPointwiseMode(mode))
        {
            continue;
        }

        bool isExpectedImplemented = expectedImplementedUnary.count(mode) > 0;
        bool isActuallyImplemented = isImplementedUnaryPointwiseMode(mode);

        EXPECT_EQ(isActuallyImplemented, isExpectedImplemented)
            << "Mode " << static_cast<int>(mode) << " implementation status mismatch";
    }
}

TEST(TestPointwiseValidation, KnownImplementedBinaryOperations)
{
    // Define which binary operations are currently implemented
    std::set<PointwiseMode> expectedImplementedBinary = {PointwiseMode::ADD,
                                                         PointwiseMode::SUB,
                                                         PointwiseMode::MUL,
                                                         PointwiseMode::RELU_BWD,
                                                         PointwiseMode::SIGMOID_BWD,
                                                         PointwiseMode::TANH_BWD};

    // Check all binary modes
    for(size_t i = 0; i < POINTWISE_MODE_COUNT; ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        // Only check modes that are classified as binary
        if(!isBinaryPointwiseMode(mode))
        {
            continue;
        }

        bool isExpectedImplemented = expectedImplementedBinary.count(mode) > 0;
        bool isActuallyImplemented = isImplementedBinaryPointwiseMode(mode);

        EXPECT_EQ(isActuallyImplemented, isExpectedImplemented)
            << "Mode " << static_cast<int>(mode) << " implementation status mismatch";
    }
}

TEST(TestPointwiseValidation, BitsetConsistencyUnary)
{
    // Verify bitset-based helpers match individual function results
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        bool isUnaryFromFunction = isUnaryPointwiseMode(mode);
        bool isImplementedFromFunction = isImplementedUnaryPointwiseMode(mode);

        // If implemented, must be classified as unary
        if(isImplementedFromFunction)
        {
            EXPECT_TRUE(isUnaryFromFunction) << "Mode " << static_cast<int>(mode)
                                             << " is implemented but not classified as unary";
        }
    }
}

TEST(TestPointwiseValidation, BitsetConsistencyBinary)
{
    // Verify bitset-based helpers match individual function results
    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        bool isBinaryFromFunction = isBinaryPointwiseMode(mode);
        bool isImplementedFromFunction = isImplementedBinaryPointwiseMode(mode);

        // If implemented, must be classified as binary
        if(isImplementedFromFunction)
        {
            EXPECT_TRUE(isBinaryFromFunction) << "Mode " << static_cast<int>(mode)
                                              << " is implemented but not classified as binary";
        }
    }
}

TEST(TestPointwiseValidation, AllModesAccountedFor)
{
    // UNSET is a special case - it should NOT be classified
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::UNSET));
    EXPECT_FALSE(isBinaryPointwiseMode(PointwiseMode::UNSET));
    EXPECT_FALSE(isTernaryPointwiseMode(PointwiseMode::UNSET));

    // All other modes MUST be classified as unary, binary, or ternary
    for(size_t i = 1; i < POINTWISE_MODE_COUNT; ++i) // Start from 1 to skip UNSET
    {
        auto mode = static_cast<PointwiseMode>(i);

        bool isClassified = isUnaryPointwiseMode(mode) || isBinaryPointwiseMode(mode)
                            || isTernaryPointwiseMode(mode);

        EXPECT_TRUE(isClassified) << "Mode " << static_cast<int>(mode)
                                  << " must be classified as unary, binary, or ternary";
    }
}
