// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/datatype_utils.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace rocm_ck;

// ============================================================================
// data_type_bits
// ============================================================================

TEST(DataType, ReportsCorrectBitsForFloatingPoint)
{
    EXPECT_EQ(data_type_bits(DataType::FP64), 64);
    EXPECT_EQ(data_type_bits(DataType::FP32), 32);
    EXPECT_EQ(data_type_bits(DataType::FP16), 16);
    EXPECT_EQ(data_type_bits(DataType::BF16), 16);
}

TEST(DataType, ReportsCorrectBitsForFP8Formats)
{
    EXPECT_EQ(data_type_bits(DataType::FP8_FNUZ), 8);
    EXPECT_EQ(data_type_bits(DataType::BF8_FNUZ), 8);
    EXPECT_EQ(data_type_bits(DataType::FP8_OCP), 8);
    EXPECT_EQ(data_type_bits(DataType::BF8_OCP), 8);
}

TEST(DataType, ReportsCorrectBitsForIntegers)
{
    EXPECT_EQ(data_type_bits(DataType::I4), 4);
    EXPECT_EQ(data_type_bits(DataType::I8), 8);
    EXPECT_EQ(data_type_bits(DataType::I16), 16);
    EXPECT_EQ(data_type_bits(DataType::I32), 32);
    EXPECT_EQ(data_type_bits(DataType::I64), 64);
    EXPECT_EQ(data_type_bits(DataType::U8), 8);
    EXPECT_EQ(data_type_bits(DataType::U16), 16);
    EXPECT_EQ(data_type_bits(DataType::U32), 32);
    EXPECT_EQ(data_type_bits(DataType::U64), 64);
}

// ============================================================================
// data_type_name
// ============================================================================

TEST(DataType, MapsEveryVariantToAValidName)
{
    // Every DataType should have a non-"???" name
    constexpr DataType all_types[] = {DataType::FP64,
                                      DataType::FP32,
                                      DataType::FP16,
                                      DataType::BF16,
                                      DataType::FP8_FNUZ,
                                      DataType::BF8_FNUZ,
                                      DataType::FP8_OCP,
                                      DataType::BF8_OCP,
                                      DataType::I4,
                                      DataType::I8,
                                      DataType::I16,
                                      DataType::I32,
                                      DataType::I64,
                                      DataType::U8,
                                      DataType::U16,
                                      DataType::U32,
                                      DataType::U64};

    for(DataType dt : all_types)
    {
        std::string_view name = data_type_name(dt);
        EXPECT_NE(name, "???") << "DataType with bits=" << data_type_bits(dt) << " has no name";
        EXPECT_FALSE(name.empty());
    }
}

TEST(DataType, MapsVariantsToExpectedStrings)
{
    EXPECT_STREQ(data_type_name(DataType::FP32), "FP32");
    EXPECT_STREQ(data_type_name(DataType::FP16), "FP16");
    EXPECT_STREQ(data_type_name(DataType::BF16), "BF16");
    EXPECT_STREQ(data_type_name(DataType::FP8_FNUZ), "FP8_FNUZ");
    EXPECT_STREQ(data_type_name(DataType::I4), "I4");
}

// ============================================================================
// constexpr validation
// ============================================================================

TEST(DataType, EvaluatesBitsAndNameAtCompileTime)
{
    // Verify these functions work at compile time
    constexpr int fp32_bits = data_type_bits(DataType::FP32);
    EXPECT_EQ(fp32_bits, 32);

    constexpr const char* fp32_name = data_type_name(DataType::FP32);
    EXPECT_STREQ(fp32_name, "FP32");
}
