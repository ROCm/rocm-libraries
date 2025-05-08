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

#include <gtest/gtest.h>

#include <tuple>

#include <DataInitializationTyped.hpp>

using namespace Tensile;
using namespace Tensile::Client;

namespace po = boost::program_options;

template <typename TypedInputs>
class DataInitializationTest : public ::testing::Test
{
public:
    using val = po::variable_value;

    using AType     = typename TypedInputs::AType;
    using BType     = typename TypedInputs::BType;
    using CType     = typename TypedInputs::CType;
    using DType     = typename TypedInputs::DType;
    using AlphaType = typename TypedInputs::AlphaType;
    using BetaType  = typename TypedInputs::BetaType;

    using TestDataInitialization = TypedDataInitialization<TypedInputs>;

    using ManagedInputs = typename TestDataInitialization::ManagedInputs;

    static_assert(std::is_same<AType, typename TestDataInitialization::AType>::value,
                  "inconsistent types");
    static_assert(std::is_same<BType, typename TestDataInitialization::BType>::value,
                  "inconsistent types");
    static_assert(std::is_same<CType, typename TestDataInitialization::CType>::value,
                  "inconsistent types");
    static_assert(std::is_same<DType, typename TestDataInitialization::DType>::value,
                  "inconsistent types");
    static_assert(std::is_same<AlphaType, typename TestDataInitialization::AlphaType>::value,
                  "inconsistent types");
    static_assert(std::is_same<BetaType, typename TestDataInitialization::BetaType>::value,
                  "inconsistent types");

    po::variables_map DataTypeArgs()
    {
        po::variables_map rv;

        rv.insert({"a-type", val(TypeInfo<AType>::Enum, false)});
        rv.insert({"b-type", val(TypeInfo<BType>::Enum, false)});
        rv.insert({"c-type", val(TypeInfo<CType>::Enum, false)});
        rv.insert({"d-type", val(TypeInfo<DType>::Enum, false)});
        rv.insert({"alpha-type", val(TypeInfo<AlphaType>::Enum, false)});
        rv.insert({"beta-type", val(TypeInfo<BetaType>::Enum, false)});

        return rv;
    }

    void RunDataContaminationTest(bool cEqualD, bool pristineGPU, BoundsCheckMode boundsCheck)
    {
        using val = po::variable_value;

        po::variables_map args = this->DataTypeArgs();

        args.insert({"init-a", val(InitMode::Zero, false)});
        args.insert({"init-b", val(InitMode::Zero, false)});
        args.insert({"init-c", val(InitMode::Zero, false)});
        args.insert({"init-d", val(InitMode::Zero, false)});
        args.insert({"init-alpha", val(InitMode::Zero, false)});
        args.insert({"init-beta", val(InitMode::Zero, false)});
        args.insert({"c-equal-d", val(cEqualD, false)});
        args.insert({"pristine-on-gpu", val(pristineGPU, false)});
        args.insert({"bounds-check", val(boundsCheck, false)});
        args.insert({"num-elements-to-validate", val(1, false)});
        args.insert({"offset-a", val((size_t)0, false)});
        args.insert({"offset-b", val((size_t)0, false)});
        args.insert({"offset-c", val((size_t)0, false)});
        args.insert({"offset-d", val((size_t)0, false)});
        args.insert({"strided-batched", val(false, false)});

        TensorDescriptor a(TypeInfo<typename TypedInputs::AType>::Enum, {10, 10, 1});
        TensorDescriptor b(TypeInfo<typename TypedInputs::BType>::Enum, {10, 10, 1});
        TensorDescriptor c(TypeInfo<typename TypedInputs::CType>::Enum, {10, 10, 1});
        TensorDescriptor d(TypeInfo<typename TypedInputs::DType>::Enum, {10, 10, 1});

        TensorOps nop;

        auto problem = ContractionProblem::GEMM(false, false, a, nop, b, nop, c, nop, d, nop, 1.5);

        ClientProblemFactory factory(problem);

        auto init = DataInitialization::Get(args, factory);

        auto genericInputs = init->prepareCPUInputs(problem);
        auto cpuInputs     = std::dynamic_pointer_cast<TypedInputs>(genericInputs);

        if(cpuInputs == nullptr)
            ASSERT_NE(cpuInputs, nullptr);

        if(cEqualD)
            EXPECT_EQ(cpuInputs->c, cpuInputs->d);
        else
            EXPECT_NE(cpuInputs->c, cpuInputs->d);

        auto zero = DataInitialization::getValue<DType, InitMode::Zero>();

        for(size_t i = 0; i < d.totalAllocatedElements(); i++)
            EXPECT_EQ(cpuInputs->d[i], zero) << i;

        for(size_t i = 0; i < d.totalAllocatedElements(); i++)
            cpuInputs->d[i] = DataInitialization::getValue<DType, InitMode::Random>();

        cpuInputs = std::dynamic_pointer_cast<TypedInputs>(init->prepareCPUInputs(problem));
        for(size_t i = 0; i < d.totalAllocatedElements(); i++)
            EXPECT_EQ(cpuInputs->d[i], zero) << i;

        auto gpuInputs = std::dynamic_pointer_cast<TypedInputs>(init->prepareGPUInputs(problem));

        std::vector<DType> gpuD(d.totalAllocatedElements());

        auto memcpyStatus
            = hipMemcpy(gpuD.data(), gpuInputs->d, d.totalAllocatedBytes(), hipMemcpyDeviceToHost);
        EXPECT_EQ(memcpyStatus, hipSuccess);

        for(size_t i = 0; i < d.totalAllocatedElements(); i++)
            EXPECT_EQ(gpuD[i], zero) << i;
    }
};

using InputTypes = ::testing::Types<TypedContractionInputs<float>,
                                    TypedContractionInputs<double>,
                                    TypedContractionInputs<Half>,
                                    ContractionInputs_B_B_S,
                                    TypedContractionInputs<std::complex<float>>,
                                    TypedContractionInputs<std::complex<double>>,
                                    TypedContractionInputs<Int8x4, Int8x4, int32_t>,
                                    TypedContractionInputs<int32_t>>;

TYPED_TEST_SUITE(DataInitializationTest, InputTypes);

/*
 * Unfortunately, googletest doesn't support tests that are both templated AND
 * parameterized, so this is the chosen compromise.
 */

TYPED_TEST(DataInitializationTest, Contamination_false_false_false)
{
    this->RunDataContaminationTest(false, false, BoundsCheckMode::Disable);
}

TYPED_TEST(DataInitializationTest, Contamination_false_true_false)
{
    this->RunDataContaminationTest(false, true, BoundsCheckMode::Disable);
}

TYPED_TEST(DataInitializationTest, Contamination_true_false_false)
{
    this->RunDataContaminationTest(true, false, BoundsCheckMode::Disable);
}

TYPED_TEST(DataInitializationTest, Contamination_true_true_false)
{
    this->RunDataContaminationTest(true, true, BoundsCheckMode::Disable);
}

TYPED_TEST(DataInitializationTest, Contamination_false_false_true)
{
    this->RunDataContaminationTest(false, false, BoundsCheckMode::NaN);
}

TYPED_TEST(DataInitializationTest, Contamination_false_true_true)
{
    this->RunDataContaminationTest(false, true, BoundsCheckMode::NaN);
}

TYPED_TEST(DataInitializationTest, Contamination_true_false_true)
{
    this->RunDataContaminationTest(true, false, BoundsCheckMode::NaN);
}

TYPED_TEST(DataInitializationTest, Contamination_true_true_true)
{
    this->RunDataContaminationTest(true, true, BoundsCheckMode::NaN);
}

template <typename T>
struct DataInitializationTestFloating : public ::testing::Test
{
    using TestType       = typename std::tuple_element<0, T>::type;
    using ComparisonType = TestType;
};

template <>
struct DataInitializationTestFloating<std::tuple<Half>> : public ::testing::Test
{
    using TestType       = Half;
    using ComparisonType = float;
};

// Typeinfo is not present for Half so wrap it in a tuple to avoid a missing
// symbol.
using FloatingPointTypes = ::testing::
    Types<std::tuple<float>, std::tuple<double>, std::tuple<Half>, std::tuple<BFloat16>>;
TYPED_TEST_SUITE(DataInitializationTestFloating, FloatingPointTypes);

TYPED_TEST(DataInitializationTestFloating, Simple)
{
    using Type       = typename TestFixture::TestType;
    using Comparison = typename TestFixture::ComparisonType;

    Type value(1.0);
    EXPECT_EQ(DataInitialization::isBadInput(value), false);
    EXPECT_EQ(DataInitialization::isBadOutput(value), false);

    value = DataInitialization::getValue<Type>(InitMode::BadInput);
    EXPECT_EQ(std::isnan(static_cast<Comparison>(value)), true) << value;
    EXPECT_EQ(DataInitialization::isBadInput(value), true) << value;
    EXPECT_EQ(DataInitialization::isBadOutput(value), false) << value;

    value = DataInitialization::getValue<Type>(InitMode::BadOutput);
    EXPECT_EQ(std::isinf(static_cast<Comparison>(value)), true) << value;
    EXPECT_EQ(DataInitialization::isBadInput(value), false) << value;
    EXPECT_EQ(DataInitialization::isBadOutput(value), true) << value;
}

template <typename T>
struct DataInitializationTestComplex : public ::testing::Test
{
};

using ComplexTypes = ::testing::Types<std::complex<float>, std::complex<double>>;

TYPED_TEST_SUITE(DataInitializationTestComplex, ComplexTypes);

TYPED_TEST(DataInitializationTestComplex, Simple)
{
    TypeParam value(1, 1);
    EXPECT_EQ(DataInitialization::isBadInput(value), false);
    EXPECT_EQ(DataInitialization::isBadOutput(value), false);

    value = DataInitialization::getValue<TypeParam>(InitMode::BadInput);
    EXPECT_EQ(std::isnan(value.real()), true);
    EXPECT_EQ(std::isnan(value.imag()), true);
    EXPECT_EQ(DataInitialization::isBadInput(value), true);
    EXPECT_EQ(DataInitialization::isBadOutput(value), false);

    value = DataInitialization::getValue<TypeParam>(InitMode::BadOutput);
    EXPECT_EQ(std::isinf(value.real()), true);
    EXPECT_EQ(std::isinf(value.imag()), true);
    EXPECT_EQ(DataInitialization::isBadInput(value), false);
    EXPECT_EQ(DataInitialization::isBadOutput(value), true);
}

TEST(DataInitializationTest, BadValues_int32)
{
    int32_t value = 1;
    EXPECT_EQ(DataInitialization::isBadInput(value), false);
    EXPECT_EQ(DataInitialization::isBadOutput(value), false);

    value = DataInitialization::getValue<int32_t>(InitMode::BadInput);
    EXPECT_EQ(std::numeric_limits<int32_t>::max(), value);
    EXPECT_EQ(DataInitialization::isBadInput(value), true);
    EXPECT_EQ(DataInitialization::isBadOutput(value), false);

    value = DataInitialization::getValue<int32_t>(InitMode::BadOutput);
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), value);
    EXPECT_EQ(DataInitialization::isBadInput(value), false);
    EXPECT_EQ(DataInitialization::isBadOutput(value), true);
}

TEST(DataInitializationTest, BadValues_Int8x4)
{
    auto maxval = std::numeric_limits<int8_t>::max();
    auto minval = std::numeric_limits<int8_t>::min();
    EXPECT_EQ(Int8x4(maxval, maxval, maxval, maxval),
              DataInitialization::getValue<Int8x4>(InitMode::BadInput));
    EXPECT_EQ(Int8x4(minval, minval, minval, minval),
              DataInitialization::getValue<Int8x4>(InitMode::BadOutput));
}
