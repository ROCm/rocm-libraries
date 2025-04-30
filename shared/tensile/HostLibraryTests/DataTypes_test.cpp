/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
// gtest), this fails to link when Tensile::Half is used by itself.  If we wrap
// this in a std::tuple, then it works correctly.
using InputTypes = ::testing::Types<std::tuple<float>,
                                    std::tuple<double>,
                                    std::tuple<Tensile::Half>,
                                    std::tuple<Tensile::BFloat16>,
                                    std::tuple<Tensile::Float8>,
                                    std::tuple<Tensile::BFloat8>,
                                    std::tuple<std::complex<float>>,
                                    std::tuple<std::complex<double>>,
                                    std::tuple<int8_t>,
                                    std::tuple<Tensile::Int8x4>,
                                    std::tuple<int32_t>>;

TYPED_TEST_SUITE(TypedDataTypesTest, InputTypes);

TYPED_TEST(TypedDataTypesTest, TypeInfo_Sizing)
{
    using TheType    = typename TestFixture::DataType;
    using MyTypeInfo = Tensile::TypeInfo<TheType>;

    static_assert(MyTypeInfo::ElementSize == sizeof(TheType), "Sizeof");
    static_assert(MyTypeInfo::ElementSize == MyTypeInfo::SegmentSize * MyTypeInfo::Packing,
                  "Packing");
}

TYPED_TEST(TypedDataTypesTest, TypeInfo_Consistency)
{
    using TheType = typename TestFixture::DataType;

    using MyTypeInfo = Tensile::TypeInfo<TheType>;

    Tensile::DataTypeInfo const& fromEnum = Tensile::DataTypeInfo::Get(MyTypeInfo::Enum);

    EXPECT_EQ(fromEnum.dataType, MyTypeInfo::Enum);
    EXPECT_EQ(fromEnum.elementSize, sizeof(TheType));
    EXPECT_EQ(fromEnum.packing, MyTypeInfo::Packing);
    EXPECT_EQ(fromEnum.segmentSize, MyTypeInfo::SegmentSize);

    EXPECT_EQ(fromEnum.isComplex, MyTypeInfo::IsComplex);
    EXPECT_EQ(fromEnum.isIntegral, MyTypeInfo::IsIntegral);
}

static_assert(Tensile::TypeInfo<float>::Enum == Tensile::DataType::Float, "Float");
static_assert(Tensile::TypeInfo<double>::Enum == Tensile::DataType::Double, "Double");
static_assert(Tensile::TypeInfo<std::complex<float>>::Enum == Tensile::DataType::ComplexFloat,
              "ComplexFloat");
static_assert(Tensile::TypeInfo<std::complex<double>>::Enum == Tensile::DataType::ComplexDouble,
              "ComplexDouble");
static_assert(Tensile::TypeInfo<Tensile::Half>::Enum == Tensile::DataType::Half, "Half");
static_assert(Tensile::TypeInfo<int8_t>::Enum == Tensile::DataType::Int8, "Int8");
static_assert(Tensile::TypeInfo<Tensile::Int8x4>::Enum == Tensile::DataType::Int8x4, "Int8x4");
static_assert(Tensile::TypeInfo<int32_t>::Enum == Tensile::DataType::Int32, "Int32");
static_assert(Tensile::TypeInfo<Tensile::BFloat16>::Enum == Tensile::DataType::BFloat16,
              "BFloat16");
static_assert(Tensile::TypeInfo<Tensile::Float8>::Enum == Tensile::DataType::Float8, "Float8");
static_assert(Tensile::TypeInfo<Tensile::BFloat8>::Enum == Tensile::DataType::BFloat8, "BFloat8");

static_assert(Tensile::TypeInfo<float>::Packing == 1, "Float");
static_assert(Tensile::TypeInfo<double>::Packing == 1, "Double");
static_assert(Tensile::TypeInfo<std::complex<float>>::Packing == 1, "ComplexFloat");
static_assert(Tensile::TypeInfo<std::complex<double>>::Packing == 1, "ComplexDouble");
static_assert(Tensile::TypeInfo<Tensile::Half>::Packing == 1, "Half");
static_assert(Tensile::TypeInfo<int8_t>::Packing == 1, "Int8");
static_assert(Tensile::TypeInfo<Tensile::Int8x4>::Packing == 4, "Int8x4");
static_assert(Tensile::TypeInfo<int32_t>::Packing == 1, "Int32");
static_assert(Tensile::TypeInfo<Tensile::BFloat16>::Packing == 1, "BFloat16");
static_assert(Tensile::TypeInfo<Tensile::Float8>::Packing == 1, "Float8");
static_assert(Tensile::TypeInfo<Tensile::BFloat8>::Packing == 1, "BFloat8");

struct Enumerations : public ::testing::TestWithParam<Tensile::DataType>
{
};

TEST_P(Enumerations, Conversions)
{
    auto val = GetParam();

    auto const& typeInfo = Tensile::DataTypeInfo::Get(val);

    EXPECT_EQ(typeInfo.name, Tensile::ToString(val));
    EXPECT_EQ(typeInfo.abbrev, Tensile::TypeAbbrev(val));
    EXPECT_EQ(&typeInfo, &Tensile::DataTypeInfo::Get(typeInfo.name));

    {
        std::istringstream input(typeInfo.name);
        Tensile::DataType  test;
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
                         ::testing::Values(Tensile::DataType::Float,
                                           Tensile::DataType::Double,
                                           Tensile::DataType::ComplexFloat,
                                           Tensile::DataType::ComplexDouble,
                                           Tensile::DataType::Half,
                                           Tensile::DataType::BFloat16,
                                           Tensile::DataType::Float8,
                                           Tensile::DataType::BFloat8,
                                           Tensile::DataType::Int8,
                                           Tensile::DataType::Int8x4,
                                           Tensile::DataType::Int32));
