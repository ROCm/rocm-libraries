/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <tuple>

#include <gtest/gtest.h>

#include <Tensile/DataTypes.hpp>

template <typename Tuple>
struct TypedDataTypesTest : public ::testing::Test
{
    using DataType = typename std::tuple_element<0, Tuple>::type;
};

// Due to a bug (could be in the compiler, in a hip runtime header, or in
// gtest), this fails to link when TensileLite::Half is used by itself.  If we wrap
// this in a std::tuple, then it works correctly.
using InputTypes = ::testing::Types<std::tuple<float>,
                                    std::tuple<double>,
                                    std::tuple<TensileLite::Half>,
                                    std::tuple<TensileLite::BFloat16>,
                                    std::tuple<TensileLite::Float8>,
                                    std::tuple<TensileLite::BFloat8>,
                                    std::tuple<std::complex<float>>,
                                    std::tuple<std::complex<double>>,
                                    std::tuple<int8_t>,
                                    std::tuple<TensileLite::Int8x4>,
                                    std::tuple<int32_t>>;

TYPED_TEST_SUITE(TypedDataTypesTest, InputTypes);

TYPED_TEST(TypedDataTypesTest, TypeInfo_Sizing)
{
    using TheType    = typename TestFixture::DataType;
    using MyTypeInfo = TensileLite::TypeInfo<TheType>;

    static_assert(MyTypeInfo::ElementSize == sizeof(TheType), "Sizeof");
    static_assert(MyTypeInfo::ElementSize == MyTypeInfo::SegmentSize * MyTypeInfo::Packing,
                  "Packing");
}

TYPED_TEST(TypedDataTypesTest, TypeInfo_Consistency)
{
    using TheType = typename TestFixture::DataType;

    using MyTypeInfo = TensileLite::TypeInfo<TheType>;

    TensileLite::DataTypeInfo const& fromEnum = TensileLite::DataTypeInfo::Get(MyTypeInfo::Enum);

    EXPECT_EQ(fromEnum.dataType, MyTypeInfo::Enum);
    EXPECT_EQ(fromEnum.elementSize, sizeof(TheType));
    EXPECT_EQ(fromEnum.packing, MyTypeInfo::Packing);
    EXPECT_EQ(fromEnum.segmentSize, MyTypeInfo::SegmentSize);

    EXPECT_EQ(fromEnum.isComplex, MyTypeInfo::IsComplex);
    EXPECT_EQ(fromEnum.isIntegral, MyTypeInfo::IsIntegral);
}

static_assert(TensileLite::TypeInfo<float>::Enum == rocisa::DataType::Float, "Float");
static_assert(TensileLite::TypeInfo<double>::Enum == rocisa::DataType::Double, "Double");
static_assert(TensileLite::TypeInfo<std::complex<float>>::Enum == rocisa::DataType::ComplexFloat,
              "ComplexFloat");
static_assert(TensileLite::TypeInfo<std::complex<double>>::Enum == rocisa::DataType::ComplexDouble,
              "ComplexDouble");
static_assert(TensileLite::TypeInfo<TensileLite::Half>::Enum == rocisa::DataType::Half, "Half");
static_assert(TensileLite::TypeInfo<int8_t>::Enum == rocisa::DataType::Int8, "Int8");
static_assert(TensileLite::TypeInfo<TensileLite::Int8x4>::Enum == rocisa::DataType::Int8x4,
              "Int8x4");
static_assert(TensileLite::TypeInfo<int32_t>::Enum == rocisa::DataType::Int32, "Int32");
static_assert(TensileLite::TypeInfo<TensileLite::BFloat16>::Enum == rocisa::DataType::BFloat16,
              "BFloat16");
static_assert(TensileLite::TypeInfo<TensileLite::Float8>::Enum == rocisa::DataType::Float8,
              "Float8");
static_assert(TensileLite::TypeInfo<TensileLite::BFloat8>::Enum == rocisa::DataType::BFloat8,
              "BFloat8");

static_assert(TensileLite::TypeInfo<float>::Packing == 1, "Float");
static_assert(TensileLite::TypeInfo<double>::Packing == 1, "Double");
static_assert(TensileLite::TypeInfo<std::complex<float>>::Packing == 1, "ComplexFloat");
static_assert(TensileLite::TypeInfo<std::complex<double>>::Packing == 1, "ComplexDouble");
static_assert(TensileLite::TypeInfo<TensileLite::Half>::Packing == 1, "Half");
static_assert(TensileLite::TypeInfo<int8_t>::Packing == 1, "Int8");
static_assert(TensileLite::TypeInfo<TensileLite::Int8x4>::Packing == 4, "Int8x4");
static_assert(TensileLite::TypeInfo<int32_t>::Packing == 1, "Int32");
static_assert(TensileLite::TypeInfo<TensileLite::BFloat16>::Packing == 1, "BFloat16");
static_assert(TensileLite::TypeInfo<TensileLite::Float8>::Packing == 1, "Float8");
static_assert(TensileLite::TypeInfo<TensileLite::BFloat8>::Packing == 1, "BFloat8");

struct Enumerations : public ::testing::TestWithParam<rocisa::DataType>
{
};

TEST_P(Enumerations, Conversions)
{
    auto val = GetParam();

    auto const& typeInfo = TensileLite::DataTypeInfo::Get(val);

    EXPECT_EQ(typeInfo.name, TensileLite::ToString(val));
    EXPECT_EQ(typeInfo.abbrev, rocisa::TypeAbbrev(val));
    EXPECT_EQ(&typeInfo, &TensileLite::DataTypeInfo::Get(typeInfo.name));

    {
        std::istringstream input(typeInfo.name);
        rocisa::DataType   test;
        input >> test;
        EXPECT_EQ(test, val);
    }

    {
        std::ostringstream output;
        output << val;
        EXPECT_EQ(output.str(), typeInfo.name);
    }
}

INSTANTIATE_TEST_SUITE_P(DataTypesTest,
                         Enumerations,
                         ::testing::Values(rocisa::DataType::Float,
                                           rocisa::DataType::Double,
                                           rocisa::DataType::ComplexFloat,
                                           rocisa::DataType::ComplexDouble,
                                           rocisa::DataType::Half,
                                           rocisa::DataType::BFloat16,
                                           rocisa::DataType::Float8,
                                           rocisa::DataType::BFloat8,
                                           rocisa::DataType::Int8,
                                           rocisa::DataType::Int8x4,
                                           rocisa::DataType::Int32));
