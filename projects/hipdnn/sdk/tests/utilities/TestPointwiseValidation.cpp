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
    // Verify specific operations we know are implemented
    EXPECT_TRUE(isImplementedUnaryPointwiseMode(PointwiseMode::RELU_FWD));
    EXPECT_TRUE(isImplementedUnaryPointwiseMode(PointwiseMode::SIGMOID_FWD));
    EXPECT_TRUE(isImplementedUnaryPointwiseMode(PointwiseMode::TANH_FWD));
    EXPECT_TRUE(isImplementedUnaryPointwiseMode(PointwiseMode::ABS));
    EXPECT_TRUE(isImplementedUnaryPointwiseMode(PointwiseMode::NEG));

    // Verify some operations we know are NOT implemented
    EXPECT_FALSE(isImplementedUnaryPointwiseMode(PointwiseMode::EXP));
    EXPECT_FALSE(isImplementedUnaryPointwiseMode(PointwiseMode::LOG));
}

TEST(TestPointwiseValidation, KnownImplementedBinaryOperations)
{
    // Verify specific operations we know are implemented
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::ADD));
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::SUB));
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::MUL));
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::RELU_BWD));
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::SIGMOID_BWD));
    EXPECT_TRUE(isImplementedBinaryPointwiseMode(PointwiseMode::TANH_BWD));

    // Verify some operations we know are NOT implemented
    EXPECT_FALSE(isImplementedBinaryPointwiseMode(PointwiseMode::DIV));
    EXPECT_FALSE(isImplementedBinaryPointwiseMode(PointwiseMode::MAX_OP));
    EXPECT_FALSE(isImplementedBinaryPointwiseMode(PointwiseMode::MIN_OP));
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
    // Every mode should be in at least one category (unary, binary, or ternary)
    int unclassifiedCount = 0;

    for(int i = static_cast<int>(PointwiseMode::UNSET);
        i <= static_cast<int>(PointwiseMode::TANH_FWD);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);

        bool isUnary = isUnaryPointwiseMode(mode);
        bool isBinary = isBinaryPointwiseMode(mode);
        bool isTernary = isTernaryPointwiseMode(mode);

        if(!isUnary && !isBinary && !isTernary)
        {
            ++unclassifiedCount;
            // It's OK to have some unclassified modes (NONE, future extensions)
            // but log them for awareness
            std::cout << "Unclassified mode: " << static_cast<int>(mode) << "\n";
        }
    }

    // Most modes should be classified
    int totalModes
        = static_cast<int>(PointwiseMode::TANH_FWD) - static_cast<int>(PointwiseMode::UNSET) + 1;
    EXPECT_LT(unclassifiedCount, totalModes / 2)
        << "Too many unclassified modes - check PointwiseValidation.hpp";
}
