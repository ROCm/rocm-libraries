// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuReferenceValidationFactory.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

// ============================================================================
// Floating-point validation tests
// ============================================================================

template <typename T>
class TestGpuFpValidation : public ::testing::Test
{
};

using FpTypes = ::testing::Types<float, half, bfloat16, double>;
TYPED_TEST_SUITE(TestGpuFpValidation, FpTypes, );

TYPED_TEST(TestGpuFpValidation, ExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({2, 3, 4});
    Tensor<TypeParam> impl({2, 3, 4});

    ref.fillWithRandomValues(static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(1.0f), 42);

    // Copy ref data into impl so they are identical
    const auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        implHost[i] = refHost[i];
    }

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, WithinTolerancePasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4, 4});
    Tensor<TypeParam> impl({4, 4});

    ref.fillWithRandomValues(static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(1.0f), 42);

    const auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        // Add small perturbation within tolerance
        implHost[i] = static_cast<TypeParam>(static_cast<float>(refHost[i]) + 1e-6f);
    }

    const float tolerance = 1e-4f;
    const GpuFpReferenceValidation<TypeParam> validator(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, BeyondToleranceFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4, 4});
    Tensor<TypeParam> impl({4, 4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(2.0f);
    }

    const float tolerance = 1e-6f;
    const GpuFpReferenceValidation<TypeParam> validator(tolerance, 0.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, NaNFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(1.0f);
    }

    // Set one element to NaN in the implementation
    implHost[0] = static_cast<TypeParam>(std::numeric_limits<float>::quiet_NaN());

    const GpuFpReferenceValidation<TypeParam> validator(1.0f, 1.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, NaNInReferenceFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(1.0f);
    }

    // Set one element to NaN in the reference
    refHost[0] = static_cast<TypeParam>(std::numeric_limits<float>::quiet_NaN());

    const GpuFpReferenceValidation<TypeParam> validator(1.0f, 1.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, InfFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(1.0f);
    }

    // Set one element to Inf in the reference
    refHost[0] = static_cast<TypeParam>(std::numeric_limits<float>::infinity());

    const GpuFpReferenceValidation<TypeParam> validator(1.0f, 1.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, InfInImplementationFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(1.0f);
    }

    // Set one element to Inf in the implementation
    implHost[0] = static_cast<TypeParam>(std::numeric_limits<float>::infinity());

    const GpuFpReferenceValidation<TypeParam> validator(1.0f, 1.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, SingleElementExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({1});
    Tensor<TypeParam> impl({1});

    ref.memory().hostData()[0] = static_cast<TypeParam>(1.0f);
    impl.memory().hostData()[0] = static_cast<TypeParam>(1.0f);

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, DimensionMismatchFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({2, 3});
    Tensor<TypeParam> impl({3, 2});

    const GpuFpReferenceValidation<TypeParam> validator;
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, EmptyTensorsPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref(std::vector<int64_t>{});
    Tensor<TypeParam> impl(std::vector<int64_t>{});

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpValidation, RelativeToleranceWorks)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();

    // Large values with small relative difference
    refHost[0] = static_cast<TypeParam>(100.0f);
    implHost[0] = static_cast<TypeParam>(100.05f);

    refHost[1] = static_cast<TypeParam>(100.0f);
    implHost[1] = static_cast<TypeParam>(100.05f);

    refHost[2] = static_cast<TypeParam>(100.0f);
    implHost[2] = static_cast<TypeParam>(100.05f);

    refHost[3] = static_cast<TypeParam>(100.0f);
    implHost[3] = static_cast<TypeParam>(100.05f);

    // rtol=0.001 gives threshold = 0.0 + 0.001 * 100.0 = 0.1 > 0.05 => pass
    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.001f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

// ============================================================================
// Integer validation tests
// ============================================================================

template <typename T>
class TestGpuIntValidation : public ::testing::Test
{
};

using IntTypes = ::testing::Types<int8_t, uint8_t, int32_t>;
TYPED_TEST_SUITE(TestGpuIntValidation, IntTypes, );

TYPED_TEST(TestGpuIntValidation, ExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({2, 3, 4});
    Tensor<TypeParam> impl({2, 3, 4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(i % 10);
        implHost[i] = static_cast<TypeParam>(i % 10);
    }

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuIntValidation, MismatchFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4});
    Tensor<TypeParam> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1);
        implHost[i] = static_cast<TypeParam>(1);
    }

    // Introduce a single mismatch
    implHost[2] = static_cast<TypeParam>(2);

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuIntValidation, SingleElementExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({1});
    Tensor<TypeParam> impl({1});

    ref.memory().hostData()[0] = static_cast<TypeParam>(5);
    impl.memory().hostData()[0] = static_cast<TypeParam>(5);

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuIntValidation, DimensionMismatchFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({2, 3});
    Tensor<TypeParam> impl({3, 2});

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuIntValidation, EmptyTensorsPasses)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref(std::vector<int64_t>{});
    Tensor<TypeParam> impl(std::vector<int64_t>{});

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_TRUE(validator.allClose(ref, impl));
}

// ============================================================================
// Factory function tests
// ============================================================================

TEST(TestGpuValidatorFactory, CreatesFloatValidator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::FLOAT);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesHalfValidator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::HALF);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesBfloat16Validator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::BFLOAT16);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesDoubleValidator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::DOUBLE);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesInt8Validator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::INT8);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesUint8Validator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::UINT8);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, CreatesInt32Validator)
{
    const auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::INT32);
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, FloatValidatorPassesMatchingData)
{
    SKIP_IF_NO_DEVICES();

    auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::FLOAT, 0.0f, 0.0f);
    ASSERT_NE(validator, nullptr);

    Tensor<float> ref({4});
    Tensor<float> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<float>(i);
        implHost[i] = static_cast<float>(i);
    }

    ASSERT_TRUE(validator->allClose(ref, impl));

    // Introduce a mismatch to confirm the validator actually checks values
    implHost[0] = 999.0f;
    impl.markHostModified();
    ASSERT_FALSE(validator->allClose(ref, impl));
}

TEST(TestGpuValidatorFactory, Int32ValidatorPassesMatchingData)
{
    SKIP_IF_NO_DEVICES();

    auto validator = createGpuAllCloseValidator(hipdnn_frontend::DataType::INT32);
    ASSERT_NE(validator, nullptr);

    Tensor<int32_t> ref({4});
    Tensor<int32_t> impl({4});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<int32_t>(i);
        implHost[i] = static_cast<int32_t>(i);
    }

    ASSERT_TRUE(validator->allClose(ref, impl));

    // Introduce a mismatch to confirm the validator actually checks values
    implHost[0] = 999;
    impl.markHostModified();
    ASSERT_FALSE(validator->allClose(ref, impl));
}

TEST(TestGpuValidatorFactory, ThrowsOnUnsupportedType)
{
    ASSERT_THROW(createGpuAllCloseValidator(hipdnn_frontend::DataType::FP8_E4M3),
                 std::runtime_error);
}

TEST(TestGpuValidatorFactory, TemplatedCreatesFloat)
{
    const auto validator = createGpuAllCloseValidator<float>();
    ASSERT_NE(validator, nullptr);
}

TEST(TestGpuValidatorFactory, TemplatedCreatesInt32)
{
    const auto validator = createGpuAllCloseValidator<int32_t>();
    ASSERT_NE(validator, nullptr);
}

// ============================================================================
// Constructor validation tests
// ============================================================================

TEST(TestGpuFpValidationConstruction, NegativeAbsoluteToleranceThrows)
{
    ASSERT_THROW(GpuFpReferenceValidation<float>(-1.0f, 0.0f), std::invalid_argument);
}

TEST(TestGpuFpValidationConstruction, NegativeRelativeToleranceThrows)
{
    ASSERT_THROW(GpuFpReferenceValidation<float>(0.0f, -1.0f), std::invalid_argument);
}

TEST(TestGpuFpValidationConstruction, NaNToleranceThrows)
{
    ASSERT_THROW(GpuFpReferenceValidation<float>(std::numeric_limits<float>::quiet_NaN(), 0.0f),
                 std::invalid_argument);
}

TEST(TestGpuFpValidationConstruction, InfToleranceThrows)
{
    ASSERT_THROW(GpuFpReferenceValidation<float>(std::numeric_limits<float>::infinity(), 0.0f),
                 std::invalid_argument);
}

// ============================================================================
// GPU vs CPU consistency tests
// ============================================================================

template <typename T>
class TestGpuVsCpuValidation : public ::testing::Test
{
};

using GpuCpuFpTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(TestGpuVsCpuValidation, GpuCpuFpTypes, );

TYPED_TEST(TestGpuVsCpuValidation, AgreeOnPass)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({8, 8});
    Tensor<TypeParam> impl({8, 8});

    ref.fillWithRandomValues(static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(1.0f), 42);

    const auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        implHost[i] = static_cast<TypeParam>(static_cast<float>(refHost[i]) + 1e-7f);
    }

    const float atol = 1e-4f;
    const float rtol = 0.0f;

    const GpuFpReferenceValidation<TypeParam> gpuValidator(atol, rtol);
    const CpuFpReferenceValidation<TypeParam> cpuValidator(atol, rtol);

    const auto gpuResult = gpuValidator.allClose(ref, impl);
    const auto cpuResult = cpuValidator.allClose(ref, impl);

    ASSERT_EQ(gpuResult, cpuResult)
        << "GPU and CPU validators disagree: GPU=" << gpuResult << ", CPU=" << cpuResult;
}

TYPED_TEST(TestGpuVsCpuValidation, AgreeOnFail)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({8, 8});
    Tensor<TypeParam> impl({8, 8});

    auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        refHost[i] = static_cast<TypeParam>(1.0f);
        implHost[i] = static_cast<TypeParam>(2.0f);
    }

    const float atol = 1e-6f;
    const float rtol = 1e-6f;

    const GpuFpReferenceValidation<TypeParam> gpuValidator(atol, rtol);
    const CpuFpReferenceValidation<TypeParam> cpuValidator(atol, rtol);

    const auto gpuResult = gpuValidator.allClose(ref, impl);
    const auto cpuResult = cpuValidator.allClose(ref, impl);

    ASSERT_EQ(gpuResult, cpuResult)
        << "GPU and CPU validators disagree: GPU=" << gpuResult << ", CPU=" << cpuResult;
}

// ============================================================================
// Large tensor test
// ============================================================================

TEST(TestGpuFpValidationLargeTensor, LargeTensorExactMatch)
{
    SKIP_IF_NO_DEVICES();

    // Test with a reasonably large tensor (64K elements)
    Tensor<float> ref({64, 32, 32});
    Tensor<float> impl({64, 32, 32});

    ref.fillWithRandomValues(-1.0f, 1.0f, 42);

    const auto* refHost = ref.memory().hostData();
    auto* implHost = impl.memory().hostData();
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        implHost[i] = refHost[i];
    }

    const GpuFpReferenceValidation<float> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

} // namespace
