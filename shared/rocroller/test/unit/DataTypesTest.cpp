/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include "SimpleFixture.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>

template <typename Tuple>
struct TypedDataTypesTest : public SimpleFixture
{
    using DataType = typename std::tuple_element<0, Tuple>::type;
};

// Due to a bug (could be in the compiler, in a hip runtime header, or in
// gtest), this fails to link when rocRoller::Half is used by itself.  If we wrap
// this in a std::tuple, then it works correctly.
using InputTypes = ::testing::Types<std::tuple<float>,
                                    std::tuple<double>,
                                    std::tuple<rocRoller::Half>,
                                    std::tuple<rocRoller::BFloat16>,
                                    std::tuple<std::complex<float>>,
                                    std::tuple<std::complex<double>>,
                                    std::tuple<int8_t>,
                                    std::tuple<rocRoller::Int8x4>,
                                    std::tuple<uint8_t>,
                                    std::tuple<rocRoller::UInt8x4>,
                                    std::tuple<int32_t>,
                                    std::tuple<int64_t>,
                                    std::tuple<rocRoller::Raw32>,
                                    std::tuple<uint32_t>,
                                    std::tuple<uint64_t>,
                                    std::tuple<rocRoller::BF8>,
                                    std::tuple<rocRoller::FP8>,
                                    std::tuple<rocRoller::BF8x4>,
                                    std::tuple<rocRoller::FP8x4>,
                                    std::tuple<rocRoller::FP6>,
                                    std::tuple<rocRoller::BF6>,
                                    std::tuple<rocRoller::FP4>,
                                    std::tuple<rocRoller::Bool32>,
                                    std::tuple<rocRoller::Bool64>,
                                    std::tuple<rocRoller::FP6x16>,
                                    std::tuple<rocRoller::BF6x16>>;

TYPED_TEST_SUITE(TypedDataTypesTest, InputTypes);

TYPED_TEST(TypedDataTypesTest, TypeInfo_Sizing)
{
    using TheType    = typename TestFixture::DataType;
    using MyTypeInfo = rocRoller::TypeInfo<TheType>;

    static_assert(MyTypeInfo::ElementBytes == sizeof(TheType), "Sizeof");
}

TYPED_TEST(TypedDataTypesTest, TypeInfo_Consistency)
{
    using TheType = typename TestFixture::DataType;

    using MyTypeInfo = rocRoller::TypeInfo<TheType>;

    rocRoller::DataTypeInfo const& fromEnum = rocRoller::DataTypeInfo::Get(MyTypeInfo::Var);

    EXPECT_EQ(fromEnum.variableType, MyTypeInfo::Var);
    EXPECT_EQ(rocRoller::CeilDivide(fromEnum.elementBits, 8u), sizeof(TheType));
    EXPECT_EQ(fromEnum.packing, MyTypeInfo::Packing);
    EXPECT_EQ(fromEnum.registerCount, MyTypeInfo::RegisterCount);

    EXPECT_EQ(fromEnum.isComplex, MyTypeInfo::IsComplex);
    EXPECT_EQ(fromEnum.isIntegral, MyTypeInfo::IsIntegral);
}

static_assert(rocRoller::TypeInfo<float>::Var == rocRoller::DataType::Float, "Float");
static_assert(rocRoller::TypeInfo<double>::Var == rocRoller::DataType::Double, "Double");
static_assert(rocRoller::TypeInfo<std::complex<float>>::Var == rocRoller::DataType::ComplexFloat,
              "ComplexFloat");
static_assert(rocRoller::TypeInfo<std::complex<double>>::Var == rocRoller::DataType::ComplexDouble,
              "ComplexDouble");
static_assert(rocRoller::TypeInfo<rocRoller::Half>::Var == rocRoller::DataType::Half, "Half");
static_assert(rocRoller::TypeInfo<int8_t>::Var == rocRoller::DataType::Int8, "Int8");
static_assert(rocRoller::TypeInfo<rocRoller::Int8x4>::Var == rocRoller::DataType::Int8x4, "Int8x4");
static_assert(rocRoller::TypeInfo<uint8_t>::Var == rocRoller::DataType::UInt8, "UInt8");
static_assert(rocRoller::TypeInfo<rocRoller::UInt8x4>::Var == rocRoller::DataType::UInt8x4,
              "UInt8x4");
static_assert(rocRoller::TypeInfo<int32_t>::Var == rocRoller::DataType::Int32, "Int32");
static_assert(rocRoller::TypeInfo<int64_t>::Var == rocRoller::DataType::Int64, "Int64");
static_assert(rocRoller::TypeInfo<rocRoller::Raw32>::Var == rocRoller::DataType::Raw32, "Raw32");
static_assert(rocRoller::TypeInfo<uint32_t>::Var == rocRoller::DataType::UInt32, "UInt32");
static_assert(rocRoller::TypeInfo<uint64_t>::Var == rocRoller::DataType::UInt64, "UInt64");
static_assert(rocRoller::TypeInfo<rocRoller::BFloat16>::Var == rocRoller::DataType::BFloat16,
              "BFloat16");
static_assert(rocRoller::TypeInfo<rocRoller::PointerLocal>::Var
                  == rocRoller::VariableType(rocRoller::DataType::None,
                                             rocRoller::PointerType::PointerLocal),
              "PointerLocal");
static_assert(rocRoller::TypeInfo<rocRoller::PointerGlobal>::Var
                  == rocRoller::VariableType(rocRoller::DataType::None,
                                             rocRoller::PointerType::PointerGlobal),
              "PointerGlobal");
static_assert(rocRoller::TypeInfo<rocRoller::Buffer>::Var
                  == rocRoller::VariableType(rocRoller::DataType::None,
                                             rocRoller::PointerType::Buffer),
              "Buffer");
static_assert(rocRoller::TypeInfo<rocRoller::BF8>::Var == rocRoller::DataType::BF8, "BF8");
static_assert(rocRoller::TypeInfo<rocRoller::FP8>::Var == rocRoller::DataType::FP8, "FP8");
static_assert(rocRoller::TypeInfo<rocRoller::BF6>::Var == rocRoller::DataType::BF6, "BF6");
static_assert(rocRoller::TypeInfo<rocRoller::FP6>::Var == rocRoller::DataType::FP6, "FP6");
static_assert(rocRoller::TypeInfo<rocRoller::FP4>::Var == rocRoller::DataType::FP4, "FP4");
static_assert(rocRoller::TypeInfo<rocRoller::BF8x4>::Var == rocRoller::DataType::BF8x4, "BF8x4");
static_assert(rocRoller::TypeInfo<rocRoller::FP8x4>::Var == rocRoller::DataType::FP8x4, "FP8x4");
static_assert(rocRoller::TypeInfo<rocRoller::BF6x16>::Var == rocRoller::DataType::BF6x16, "BF6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP6x16>::Var == rocRoller::DataType::FP6x16, "FP6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP4x8>::Var == rocRoller::DataType::FP4x8, "FP4x8");

static_assert(rocRoller::TypeInfo<float>::Packing == 1, "Float");
static_assert(rocRoller::TypeInfo<double>::Packing == 1, "Double");
static_assert(rocRoller::TypeInfo<std::complex<float>>::Packing == 1, "ComplexFloat");
static_assert(rocRoller::TypeInfo<std::complex<double>>::Packing == 1, "ComplexDouble");
static_assert(rocRoller::TypeInfo<rocRoller::Half>::Packing == 1, "Half");
static_assert(rocRoller::TypeInfo<int8_t>::Packing == 1, "Int8");
static_assert(rocRoller::TypeInfo<rocRoller::Int8x4>::Packing == 4, "Int8x4");
static_assert(rocRoller::TypeInfo<uint8_t>::Packing == 1, "UInt8");
static_assert(rocRoller::TypeInfo<rocRoller::UInt8x4>::Packing == 4, "UInt8x4");
static_assert(rocRoller::TypeInfo<int32_t>::Packing == 1, "Int32");
static_assert(rocRoller::TypeInfo<int64_t>::Packing == 1, "Int64");
static_assert(rocRoller::TypeInfo<rocRoller::Raw32>::Packing == 1, "Raw32");
static_assert(rocRoller::TypeInfo<uint32_t>::Packing == 1, "UInt32");
static_assert(rocRoller::TypeInfo<uint64_t>::Packing == 1, "UInt64");
static_assert(rocRoller::TypeInfo<rocRoller::BFloat16>::Packing == 1, "BFloat16");
static_assert(rocRoller::TypeInfo<rocRoller::PointerLocal>::Packing == 1, "PointerLocal");
static_assert(rocRoller::TypeInfo<rocRoller::PointerGlobal>::Packing == 1, "PointerGlobal");
static_assert(rocRoller::TypeInfo<rocRoller::Buffer>::Packing == 1, "Buffer");
static_assert(rocRoller::TypeInfo<rocRoller::BF8>::Packing == 1, "BF8");
static_assert(rocRoller::TypeInfo<rocRoller::FP8>::Packing == 1, "FP8");
static_assert(rocRoller::TypeInfo<rocRoller::BF6>::Packing == 1, "BF6");
static_assert(rocRoller::TypeInfo<rocRoller::FP6>::Packing == 1, "FP6");
static_assert(rocRoller::TypeInfo<rocRoller::FP4>::Packing == 1, "FP4");
static_assert(rocRoller::TypeInfo<rocRoller::BF8x4>::Packing == 4, "BF8x4");
static_assert(rocRoller::TypeInfo<rocRoller::FP8x4>::Packing == 4, "FP8x4");
static_assert(rocRoller::TypeInfo<rocRoller::BF6x16>::Packing == 16, "BF6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP6x16>::Packing == 16, "FP6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP4x8>::Packing == 8, "FP4x8");

static_assert(rocRoller::TypeInfo<float>::RegisterCount == 1, "Float");
static_assert(rocRoller::TypeInfo<double>::RegisterCount == 2, "Double");
static_assert(rocRoller::TypeInfo<std::complex<float>>::RegisterCount == 2, "ComplexFloat");
static_assert(rocRoller::TypeInfo<std::complex<double>>::RegisterCount == 4, "ComplexDouble");
static_assert(rocRoller::TypeInfo<rocRoller::Half>::RegisterCount == 1, "Half");
static_assert(rocRoller::TypeInfo<int8_t>::RegisterCount == 1, "Int8");
static_assert(rocRoller::TypeInfo<rocRoller::Int8x4>::RegisterCount == 1, "Int8x4");
static_assert(rocRoller::TypeInfo<uint8_t>::RegisterCount == 1, "UInt8");
static_assert(rocRoller::TypeInfo<rocRoller::UInt8x4>::RegisterCount == 1, "UInt8x4");
static_assert(rocRoller::TypeInfo<int32_t>::RegisterCount == 1, "Int32");
static_assert(rocRoller::TypeInfo<int64_t>::RegisterCount == 2, "Int64");
static_assert(rocRoller::TypeInfo<rocRoller::Raw32>::RegisterCount == 1, "Raw32");
static_assert(rocRoller::TypeInfo<uint32_t>::RegisterCount == 1, "UInt32");
static_assert(rocRoller::TypeInfo<uint64_t>::RegisterCount == 2, "UInt64");
static_assert(rocRoller::TypeInfo<rocRoller::BFloat16>::RegisterCount == 1, "BFloat16");
static_assert(rocRoller::TypeInfo<rocRoller::PointerLocal>::RegisterCount == 1, "PointerLocal");
static_assert(rocRoller::TypeInfo<rocRoller::PointerGlobal>::RegisterCount == 2, "PointerGlobal");
static_assert(rocRoller::TypeInfo<rocRoller::Buffer>::RegisterCount == 4, "Buffer");
static_assert(rocRoller::TypeInfo<rocRoller::BF8>::RegisterCount == 1, "BF8");
static_assert(rocRoller::TypeInfo<rocRoller::FP8>::RegisterCount == 1, "FP8");
static_assert(rocRoller::TypeInfo<rocRoller::BF6>::RegisterCount == 1, "BF6");
static_assert(rocRoller::TypeInfo<rocRoller::FP6>::RegisterCount == 1, "FP6");
static_assert(rocRoller::TypeInfo<rocRoller::FP4>::RegisterCount == 1, "FP4");
static_assert(rocRoller::TypeInfo<rocRoller::BF8x4>::RegisterCount == 1, "BF8x4");
static_assert(rocRoller::TypeInfo<rocRoller::FP8x4>::RegisterCount == 1, "FP8x4");
static_assert(rocRoller::TypeInfo<rocRoller::BF6x16>::RegisterCount == 3, "BF6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP6x16>::RegisterCount == 3, "FP6x16");
static_assert(rocRoller::TypeInfo<rocRoller::FP4x8>::RegisterCount == 1, "FP4x8");

struct Enumerations : public ::testing::TestWithParam<rocRoller::DataType>
{
};

TEST_P(Enumerations, Conversions)
{
    auto val = GetParam();

    auto const& typeInfo = rocRoller::DataTypeInfo::Get(val);

    EXPECT_EQ(typeInfo.name, rocRoller::toString(val));
    EXPECT_EQ(typeInfo.abbrev, rocRoller::TypeAbbrev(val));
    EXPECT_EQ(&typeInfo, &rocRoller::DataTypeInfo::Get(typeInfo.name));

    {
        std::istringstream  input(typeInfo.name);
        rocRoller::DataType test;
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
                         ::testing::Values(rocRoller::DataType::Float,
                                           rocRoller::DataType::Double,
                                           rocRoller::DataType::ComplexFloat,
                                           rocRoller::DataType::ComplexDouble,
                                           rocRoller::DataType::Half,
                                           rocRoller::DataType::Int8,
                                           rocRoller::DataType::Int8x4,
                                           rocRoller::DataType::UInt8,
                                           rocRoller::DataType::UInt8x4,
                                           rocRoller::DataType::Int32,
                                           rocRoller::DataType::Int64,
                                           rocRoller::DataType::Raw32,
                                           rocRoller::DataType::UInt32,
                                           rocRoller::DataType::UInt64,
                                           rocRoller::DataType::BFloat16,
                                           rocRoller::DataType::BF8,
                                           rocRoller::DataType::FP8,
                                           rocRoller::DataType::BF8x4,
                                           rocRoller::DataType::FP8x4,
                                           rocRoller::DataType::BF6,
                                           rocRoller::DataType::FP6,
                                           rocRoller::DataType::BF6x16,
                                           rocRoller::DataType::FP6x16,
                                           rocRoller::DataType::FP4));

class DataTypesTest : public SimpleFixture
{
};

TEST_F(DataTypesTest, Promotions)
{
    using namespace rocRoller;

    auto ExpectEqual = [&](VariableType result, VariableType lhs, VariableType rhs) {
        EXPECT_EQ(result, VariableType::Promote(lhs, rhs));
        EXPECT_EQ(result, VariableType::Promote(rhs, lhs));
    };

    ExpectEqual(DataType::Float, DataType::Float, DataType::Float);

    VariableType floatPtr(DataType::Float, PointerType::PointerGlobal);

    ExpectEqual(floatPtr, floatPtr, DataType::Int32);
    ExpectEqual(floatPtr, DataType::Int32, floatPtr);
    ExpectEqual(floatPtr, DataType::Int64, floatPtr);

    EXPECT_ANY_THROW(VariableType::Promote(floatPtr, floatPtr));

    ExpectEqual(DataType::Int64, DataType::Int32, DataType::Int64);
    ExpectEqual(DataType::Int64, DataType::Raw32, DataType::Int64);
    ExpectEqual(DataType::Int32, DataType::Raw32, DataType::Int32);

    ExpectEqual(DataType::Bool32, DataType::Bool, DataType::Bool32);
    ExpectEqual(DataType::Bool64, DataType::Bool, DataType::Bool64);
}

TEST_F(DataTypesTest, GetIntegerType)
{
    using namespace rocRoller;

    EXPECT_EQ(DataType::Int8, getIntegerType(true, 1));
    EXPECT_EQ(DataType::Int16, getIntegerType(true, 2));
    EXPECT_EQ(DataType::Int32, getIntegerType(true, 4));
    EXPECT_EQ(DataType::Int64, getIntegerType(true, 8));

    EXPECT_EQ(DataType::UInt8, getIntegerType(false, 1));
    EXPECT_EQ(DataType::UInt16, getIntegerType(false, 2));
    EXPECT_EQ(DataType::UInt32, getIntegerType(false, 4));
    EXPECT_EQ(DataType::UInt64, getIntegerType(false, 8));

    EXPECT_THROW(getIntegerType(false, 3), FatalError);
    EXPECT_THROW(getIntegerType(false, 5), FatalError);
    EXPECT_THROW(getIntegerType(false, 16), FatalError);

    EXPECT_EQ(DataType::UInt64,
              VariableType(DataType::Half, PointerType::PointerGlobal).getArithmeticType());
    EXPECT_EQ(DataType::UInt32,
              VariableType(DataType::Half, PointerType::PointerLocal).getArithmeticType());
    EXPECT_EQ(DataType::Half, VariableType(DataType::Half).getArithmeticType());

    EXPECT_EQ(DataType::UInt64,
              VariableType(DataType::Int32, PointerType::PointerGlobal).getArithmeticType());
    EXPECT_EQ(DataType::UInt32,
              VariableType(DataType::Int32, PointerType::PointerLocal).getArithmeticType());
    EXPECT_EQ(DataType::Int32, VariableType(DataType::Int32).getArithmeticType());
}

TEST_F(DataTypesTest, PackedUnpacked)
{
    using namespace rocRoller;
    EXPECT_EQ(DataType::Halfx2, DataTypeInfo::Get(DataType::Half).packedVariableType());
    EXPECT_EQ(std::nullopt, DataTypeInfo::Get(DataType::Halfx2).packedVariableType());

    EXPECT_EQ(DataType::BF8x4, DataTypeInfo::Get(DataType::BF8).packedVariableType());
    EXPECT_EQ(std::nullopt, DataTypeInfo::Get(DataType::Halfx2).packedVariableType());

    EXPECT_EQ(DataType::FP8x4, DataTypeInfo::Get(DataType::FP8).packedVariableType());
    EXPECT_EQ(std::nullopt, DataTypeInfo::Get(DataType::FP8x4).packedVariableType());

    EXPECT_EQ(std::nullopt, DataTypeInfo::Get(DataType::Halfx2).packedVariableType());
}
