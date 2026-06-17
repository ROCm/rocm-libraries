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

// ============================================================================
// Strided (unpacked) tensor validation tests
// ============================================================================

// Increment a multi-dimensional index counter (innermost-first).
// Returns false when all positions have been exhausted (wraps back to zero).
inline bool incrementIndices(std::vector<int64_t>& indices, const std::vector<int64_t>& dims)
{
    for(auto d = indices.size(); d-- > 0;)
    {
        if(++indices[d] < dims[d])
        {
            return true;
        }
        indices[d] = 0;
    }
    return false;
}

// Helper: copy data element-wise between two tensors with potentially different strides.
// Both tensors must have the same dims.
template <typename T>
void copyByLogicalIndex(Tensor<T>& dst, const Tensor<T>& src)
{
    const auto& dims = src.dims();
    std::vector<int64_t> indices(dims.size(), 0);
    for(size_t i = 0; i < src.elementCount(); ++i)
    {
        dst(indices) = src(indices);
        incrementIndices(indices, dims);
    }
}

template <typename T>
class TestGpuFpStridedValidation : public ::testing::Test
{
};

TYPED_TEST_SUITE(TestGpuFpStridedValidation, FpTypes, );

TYPED_TEST(TestGpuFpStridedValidation, StridedExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    // Non-contiguous strides: N-stride is 120 instead of 60, creating gaps
    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    ASSERT_FALSE(ref.isPacked());

    // Fill ref with values via logical indices
    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        ref(indices) = static_cast<TypeParam>(static_cast<float>(i) * 0.1f);
        incrementIndices(indices, dims);
    }

    copyByLogicalIndex(impl, ref);

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpStridedValidation, StridedMismatchFails)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    // Fill both with same data
    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        auto val = static_cast<TypeParam>(1.0f);
        ref(indices) = val;
        impl(indices) = val;

        incrementIndices(indices, dims);
    }

    // Inject mismatch at a specific logical position
    impl(1, 2, 3, 4) = static_cast<TypeParam>(999.0f);

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpStridedValidation, RefPackedImplStridedPasses)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> implStrides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims); // packed NCHW
    Tensor<TypeParam> impl(dims, implStrides); // strided (gaps in N dim)

    ASSERT_TRUE(ref.isPacked());
    ASSERT_FALSE(impl.isPacked());

    // Fill ref with values
    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        ref(indices) = static_cast<TypeParam>(static_cast<float>(i) * 0.1f);

        incrementIndices(indices, dims);
    }

    // Copy by logical index so impl has the same logical values in NHWC layout
    copyByLogicalIndex(impl, ref);

    const GpuFpReferenceValidation<TypeParam> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuFpStridedValidation, StridedNaNFails)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    // Fill both with 1.0
    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        ref(indices) = static_cast<TypeParam>(1.0f);
        impl(indices) = static_cast<TypeParam>(1.0f);

        incrementIndices(indices, dims);
    }

    // Inject NaN at a position that would be missed if kernel used linear indexing
    impl(1, 2, 1, 3) = static_cast<TypeParam>(std::numeric_limits<float>::quiet_NaN());

    const GpuFpReferenceValidation<TypeParam> validator(1.0f, 1.0f);
    ASSERT_FALSE(validator.allClose(ref, impl));
}

// Strided integer tests

template <typename T>
class TestGpuIntStridedValidation : public ::testing::Test
{
};

TYPED_TEST_SUITE(TestGpuIntStridedValidation, IntTypes, );

TYPED_TEST(TestGpuIntStridedValidation, StridedExactMatchPasses)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        auto val = static_cast<TypeParam>(i % 10);
        ref(indices) = val;
        impl(indices) = val;

        incrementIndices(indices, dims);
    }

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_TRUE(validator.allClose(ref, impl));
}

TYPED_TEST(TestGpuIntStridedValidation, StridedMismatchFails)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        auto val = static_cast<TypeParam>(1);
        ref(indices) = val;
        impl(indices) = val;

        incrementIndices(indices, dims);
    }

    impl(1, 2, 3, 4) = static_cast<TypeParam>(9);

    const GpuIntReferenceValidation<TypeParam> validator;
    ASSERT_FALSE(validator.allClose(ref, impl));
}

// GPU vs CPU consistency for strided tensors

TYPED_TEST(TestGpuVsCpuValidation, StridedAgreeOnPass)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 4, 4, 4};
    const std::vector<int64_t> strides = {128, 16, 4, 1}; // N-stride doubled for gaps
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    // Fill with identical data via logical indices
    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        auto val = static_cast<TypeParam>(static_cast<float>(i) * 0.01f);
        ref(indices) = val;
        impl(indices) = static_cast<TypeParam>(static_cast<float>(val) + 1e-7f);

        incrementIndices(indices, dims);
    }

    const float atol = 1e-4f;
    const float rtol = 0.0f;

    const GpuFpReferenceValidation<TypeParam> gpuValidator(atol, rtol);
    const CpuFpReferenceValidation<TypeParam> cpuValidator(atol, rtol);

    const auto gpuResult = gpuValidator.allClose(ref, impl);
    const auto cpuResult = cpuValidator.allClose(ref, impl);

    ASSERT_EQ(gpuResult, cpuResult)
        << "GPU and CPU validators disagree (strided): GPU=" // NOLINT(readability-implicit-bool-conversion)
        << gpuResult << ", CPU=" << cpuResult;
}

TYPED_TEST(TestGpuVsCpuValidation, StridedAgreeOnFail)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 4, 4, 4};
    const std::vector<int64_t> strides = {128, 16, 4, 1};
    Tensor<TypeParam> ref(dims, strides);
    Tensor<TypeParam> impl(dims, strides);

    std::vector<int64_t> indices(4, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        ref(indices) = static_cast<TypeParam>(1.0f);
        impl(indices) = static_cast<TypeParam>(2.0f);

        incrementIndices(indices, dims);
    }

    const float atol = 1e-6f;
    const float rtol = 1e-6f;

    const GpuFpReferenceValidation<TypeParam> gpuValidator(atol, rtol);
    const CpuFpReferenceValidation<TypeParam> cpuValidator(atol, rtol);

    const auto gpuResult = gpuValidator.allClose(ref, impl);
    const auto cpuResult = cpuValidator.allClose(ref, impl);

    ASSERT_EQ(gpuResult, cpuResult)
        << "GPU and CPU disagree (strided fail): GPU=" // NOLINT(readability-implicit-bool-conversion)
        << gpuResult << ", CPU=" << cpuResult;
}

// High-dimensional strided test

TEST(TestGpuFpValidationStrided, HighDimensionStridedPasses)
{
    SKIP_IF_NO_DEVICES();

    // 6D tensor with non-packed strides (stride[5]=2 instead of 1)
    const std::vector<int64_t> dims = {2, 2, 2, 2, 2, 3};
    const std::vector<int64_t> strides = {192, 96, 48, 24, 12, 2};
    Tensor<float> ref(dims, strides);
    Tensor<float> impl(dims, strides);

    ASSERT_FALSE(ref.isPacked());

    std::vector<int64_t> indices(6, 0);
    for(size_t i = 0; i < ref.elementCount(); ++i)
    {
        auto val = static_cast<float>(i) * 0.1f;
        ref(indices) = val;
        impl(indices) = val;

        incrementIndices(indices, dims);
    }

    const GpuFpReferenceValidation<float> validator(0.0f, 0.0f);
    ASSERT_TRUE(validator.allClose(ref, impl));
}

} // namespace
