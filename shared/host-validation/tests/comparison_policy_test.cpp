// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {
using roc::host_validation::ScalarType;
using roc::host_validation::Shape;
using roc::host_validation::Tensor;

size_t expectationCount = 0;
size_t expectedTrueCount = 0;
size_t expectedFalseCount = 0;
size_t failureCount = 0;
std::string_view currentExpectationSet = "comparison policy";

void recordExpectation(bool actual, bool expected, const char* expression, int line) {
    ++expectationCount;
    expected ? ++expectedTrueCount : ++expectedFalseCount;
    if (actual == expected) return;

    ++failureCount;
    std::cerr << currentExpectationSet << ':' << line << ": expected "
              << (expected ? "true" : "false") << ": " << expression << '\n';
}

#define EXPECT_TRUE(expression) \
    recordExpectation(static_cast<bool>(expression), true, #expression, __LINE__)
#define EXPECT_FALSE(expression) \
    recordExpectation(static_cast<bool>(expression), false, #expression, __LINE__)

template <typename T>
std::enable_if_t<std::is_integral_v<T>, bool> close(T observed, T expected) {
    return roc::host_validation::valuesClose(observed, expected);
}

bool close(float observed, float expected) {
    return roc::host_validation::valuesClose(
        observed, expected, roc::host_validation::defaultComparisonOptions(ScalarType::Float32));
}

bool close(double observed, double expected) {
    return roc::host_validation::valuesClose(
        observed, expected, roc::host_validation::defaultComparisonOptions(ScalarType::Float64));
}

bool close(std::complex<float> observed, std::complex<float> expected) {
    return roc::host_validation::valuesClose(
        observed, expected,
        roc::host_validation::defaultComparisonOptions(ScalarType::ComplexFloat32));
}

bool close(std::complex<double> observed, std::complex<double> expected) {
    return roc::host_validation::valuesClose(
        observed, expected,
        roc::host_validation::defaultComparisonOptions(ScalarType::ComplexFloat64));
}

Tensor quantizedScalar(ScalarType type, float value) {
    const std::array<float, 1> values{value};
    return Tensor::fromValues(type, Shape{1}, std::span<const float>(values));
}

Tensor scalarFromRawByte(ScalarType type, uint8_t raw) {
    if (roc::host_validation::scalarTypeInfo(type).storageBits != 8)
        throw std::invalid_argument("Raw-byte scalar requires an 8-bit storage type.");

    Tensor value(type, Shape{1});
    value.mutableStorage()[0] = static_cast<std::byte>(raw);
    return value;
}

bool close(const Tensor& observed, const Tensor& expected) {
    if (observed.type() != expected.type() || observed.size() != 1 || expected.size() != 1)
        throw std::invalid_argument(
            "Quantized scalar comparison requires matching one-element tensors.");

    return roc::host_validation::valuesClose(
        observed.view().loadAs<float>({0}), expected.view().loadAs<float>({0}),
        roc::host_validation::defaultComparisonOptions(observed.type()));
}

void testFloat() {
    EXPECT_TRUE(close(1.0f, 1.0f));
    EXPECT_TRUE(close(-1.0f, -1.0f));
    EXPECT_TRUE(close(0.0f, 0.0f));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, 1.0f));
    EXPECT_FALSE(close(1.0f, nan));

    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(close(inf, inf));
    EXPECT_TRUE(close(-inf, -inf));
    EXPECT_FALSE(close(inf, -inf));
    EXPECT_FALSE(close(inf, 1.0f));

    EXPECT_TRUE(close(1.0f, 1.0002f));
    EXPECT_FALSE(close(1.0f, 1.001f));

    {
        const float value = 10.0f;
        const float tolerance = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(ScalarType::Float32));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 0.99f;
        EXPECT_TRUE(close(value, value + delta));
    }

    {
        const float value = 10.0f;
        const float tolerance = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(ScalarType::Float32));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 1.01f;
        EXPECT_FALSE(close(value, value + delta));
    }

    EXPECT_TRUE(close(1.0f, 1.0002f));
    EXPECT_TRUE(close(1.0002f, 1.0f));
    EXPECT_FALSE(close(1.0f, 1.001f));
    EXPECT_FALSE(close(1.001f, 1.0f));

    EXPECT_TRUE(close(-0.0f, 0.0f));
    EXPECT_TRUE(close(0.0f, -0.0f));

    EXPECT_TRUE(close(0.0f, 0.00005f));
    EXPECT_FALSE(close(0.0f, 0.0005f));

    EXPECT_FALSE(close(-0.0002f, 0.0002f));

    EXPECT_TRUE(close(1e6f, 1e6f + 100.0f));
    EXPECT_FALSE(close(1e6f, 1e6f + 500.0f));

    EXPECT_TRUE(close(-1.0f, -1.0002f));
    EXPECT_FALSE(close(-1.0f, -1.001f));
}

void testDouble() {
    EXPECT_TRUE(close(1.0, 1.0));
    EXPECT_TRUE(close(-1.0, -1.0));
    EXPECT_TRUE(close(0.0, 0.0));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, 1.0));
    EXPECT_FALSE(close(1.0, nan));

    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(close(inf, inf));
    EXPECT_TRUE(close(-inf, -inf));
    EXPECT_FALSE(close(inf, -inf));
    EXPECT_FALSE(close(inf, 1.0));

    EXPECT_TRUE(close(1.0, 1.0 + 2e-12));
    EXPECT_FALSE(close(1.0, 1.0 + 1e-9));

    {
        const double value = 10.0;
        const double tolerance =
            roc::host_validation::defaultSymmetricRelativeTolerance(ScalarType::Float64);
        const double delta = tolerance * (2.0 * value + 1.0) / (1.0 - tolerance) * 0.99;
        EXPECT_TRUE(close(value, value + delta));
    }

    {
        const double value = 10.0;
        const double tolerance =
            roc::host_validation::defaultSymmetricRelativeTolerance(ScalarType::Float64);
        const double delta = tolerance * (2.0 * value + 1.0) / (1.0 - tolerance) * 1.01;
        EXPECT_FALSE(close(value, value + delta));
    }

    EXPECT_TRUE(close(1.0, 1.0 + 2e-12));
    EXPECT_TRUE(close(1.0 + 2e-12, 1.0));
    EXPECT_FALSE(close(1.0, 1.0 + 1e-9));
    EXPECT_FALSE(close(1.0 + 1e-9, 1.0));

    EXPECT_TRUE(close(-0.0, 0.0));
    EXPECT_TRUE(close(0.0, -0.0));

    EXPECT_TRUE(close(0.0, 5e-13));
    EXPECT_FALSE(close(0.0, 5e-9));

    EXPECT_FALSE(close(-1e-6, 1e-6));

    EXPECT_TRUE(close(1e12, 1e12 + 1.0));
    EXPECT_FALSE(close(1e12, 1e12 + 5.0));

    EXPECT_TRUE(close(-1.0, -1.0 - 2e-12));
    EXPECT_FALSE(close(-1.0, -1.0 - 1e-9));
}

void testFloat16() {
    constexpr ScalarType type = ScalarType::Float16;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(close(quantizedScalar(type, nan), quantizedScalar(type, nan)));
    EXPECT_FALSE(close(quantizedScalar(type, nan), quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, nan)));

    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(close(quantizedScalar(type, inf), quantizedScalar(type, inf)));
    EXPECT_TRUE(close(quantizedScalar(type, -inf), quantizedScalar(type, -inf)));
    EXPECT_FALSE(close(quantizedScalar(type, inf), quantizedScalar(type, -inf)));
    EXPECT_FALSE(close(quantizedScalar(type, inf), quantizedScalar(type, 1.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.02f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.1f)));

    {
        const float value = 4.0f;
        const float tolerance =
            static_cast<float>(roc::host_validation::defaultSymmetricRelativeTolerance(type));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 0.9f;
        EXPECT_TRUE(close(quantizedScalar(type, value), quantizedScalar(type, value + delta)));
    }

    {
        const float value = 4.0f;
        const float tolerance =
            static_cast<float>(roc::host_validation::defaultSymmetricRelativeTolerance(type));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 1.1f;
        EXPECT_FALSE(close(quantizedScalar(type, value), quantizedScalar(type, value + delta)));
    }

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.02f)));
    EXPECT_TRUE(close(quantizedScalar(type, 1.02f), quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.1f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.1f), quantizedScalar(type, 1.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.005f)));
    EXPECT_FALSE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.05f)));

    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.02f)));
    EXPECT_FALSE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.1f)));
}

void testBFloat16() {
    constexpr ScalarType type = ScalarType::BFloat16;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(close(quantizedScalar(type, nan), quantizedScalar(type, nan)));
    EXPECT_FALSE(close(quantizedScalar(type, nan), quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, nan)));

    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(close(quantizedScalar(type, inf), quantizedScalar(type, inf)));
    EXPECT_TRUE(close(quantizedScalar(type, -inf), quantizedScalar(type, -inf)));
    EXPECT_FALSE(close(quantizedScalar(type, inf), quantizedScalar(type, -inf)));
    EXPECT_FALSE(close(quantizedScalar(type, inf), quantizedScalar(type, 1.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.25f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));

    {
        const float value = 4.0f;
        const float tolerance =
            static_cast<float>(roc::host_validation::defaultSymmetricRelativeTolerance(type));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 0.85f;
        EXPECT_TRUE(close(quantizedScalar(type, value), quantizedScalar(type, value + delta)));
    }

    {
        const float value = 4.0f;
        const float tolerance =
            static_cast<float>(roc::host_validation::defaultSymmetricRelativeTolerance(type));
        const float delta = tolerance * (2.0f * value + 1.0f) / (1.0f - tolerance) * 1.15f;
        EXPECT_FALSE(close(quantizedScalar(type, value), quantizedScalar(type, value + delta)));
    }

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.25f)));
    EXPECT_TRUE(close(quantizedScalar(type, 2.25f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 1.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.05f)));
    EXPECT_FALSE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.5f)));

    EXPECT_TRUE(close(quantizedScalar(type, -2.0f), quantizedScalar(type, -2.25f)));
    EXPECT_FALSE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -2.0f)));
}

void testFloat8E4M3() {
    constexpr ScalarType type = ScalarType::Float8E4M3;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const Tensor nan = scalarFromRawByte(type, 0x7f);
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), nan));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.5f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 4.0f), quantizedScalar(type, 5.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 4.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.5f)));
    EXPECT_TRUE(close(quantizedScalar(type, 2.5f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 1.0f)));

    const Tensor maximum = scalarFromRawByte(type, 0x7e);
    EXPECT_TRUE(close(maximum, maximum));
}

void testFloat8E5M2() {
    constexpr ScalarType type = ScalarType::Float8E5M2;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const Tensor nan = scalarFromRawByte(type, 0x7f);
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), nan));

    const Tensor inf = scalarFromRawByte(type, 0x7c);
    const Tensor negativeInf = scalarFromRawByte(type, 0xfc);
    EXPECT_TRUE(close(inf, inf));
    EXPECT_TRUE(close(negativeInf, negativeInf));
    EXPECT_FALSE(close(inf, negativeInf));
    EXPECT_FALSE(close(inf, quantizedScalar(type, 1.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.5f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 3.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 3.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 4.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.5f)));
    EXPECT_TRUE(close(quantizedScalar(type, 1.5f), quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 3.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 3.0f), quantizedScalar(type, 1.0f)));
}

void testFloat8E4M3Fnuz() {
    constexpr ScalarType type = ScalarType::Float8E4M3Fnuz;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const Tensor nan = scalarFromRawByte(type, 0x80);
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), nan));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.5f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 4.0f), quantizedScalar(type, 5.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 4.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 2.5f)));
    EXPECT_TRUE(close(quantizedScalar(type, 2.5f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 2.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 1.0f)));

    const Tensor maximum = scalarFromRawByte(type, 0x7f);
    EXPECT_TRUE(close(maximum, maximum));
}

void testFloat8E5M2Fnuz() {
    constexpr ScalarType type = ScalarType::Float8E5M2Fnuz;

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, -1.0f), quantizedScalar(type, -1.0f)));
    EXPECT_TRUE(close(quantizedScalar(type, 0.0f), quantizedScalar(type, 0.0f)));

    const Tensor nan = scalarFromRawByte(type, 0x80);
    EXPECT_FALSE(close(nan, nan));
    EXPECT_FALSE(close(nan, quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), nan));

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.5f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 3.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 2.0f), quantizedScalar(type, 3.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 4.0f)));

    EXPECT_TRUE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 1.5f)));
    EXPECT_TRUE(close(quantizedScalar(type, 1.5f), quantizedScalar(type, 1.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 1.0f), quantizedScalar(type, 3.0f)));
    EXPECT_FALSE(close(quantizedScalar(type, 3.0f), quantizedScalar(type, 1.0f)));

    const Tensor maximum = scalarFromRawByte(type, 0x7f);
    EXPECT_TRUE(close(maximum, maximum));
}

void testInt8() {
    EXPECT_TRUE(close(int8_t(42), int8_t(42)));
    EXPECT_TRUE(close(int8_t(0), int8_t(0)));
    EXPECT_TRUE(close(int8_t(-1), int8_t(-1)));

    EXPECT_FALSE(close(int8_t(1), int8_t(2)));
    EXPECT_FALSE(close(int8_t(-1), int8_t(1)));

    EXPECT_FALSE(close(int8_t(10), int8_t(11)));
    EXPECT_FALSE(close(int8_t(0), int8_t(1)));

    EXPECT_TRUE(close(INT8_MAX, INT8_MAX));
    EXPECT_TRUE(close(INT8_MIN, INT8_MIN));
    EXPECT_FALSE(close(INT8_MIN, INT8_MAX));
}

void testInt() {
    EXPECT_TRUE(close(42, 42));
    EXPECT_TRUE(close(0, 0));
    EXPECT_TRUE(close(-100, -100));

    EXPECT_FALSE(close(1, 2));
    EXPECT_FALSE(close(-1, 1));

    EXPECT_FALSE(close(1000, 1001));
    EXPECT_FALSE(close(0, 1));

    EXPECT_TRUE(close(INT_MAX, INT_MAX));
    EXPECT_TRUE(close(INT_MIN, INT_MIN));
    EXPECT_FALSE(close(INT_MIN, INT_MAX));
}

void testUnsignedInt() {
    EXPECT_TRUE(close(42u, 42u));
    EXPECT_TRUE(close(0u, 0u));

    EXPECT_FALSE(close(1u, 2u));
    EXPECT_FALSE(close(100u, 200u));

    EXPECT_FALSE(close(1000u, 1001u));
    EXPECT_FALSE(close(0u, 1u));

    EXPECT_TRUE(close(UINT_MAX, UINT_MAX));
    EXPECT_TRUE(close(0u, 0u));
    EXPECT_FALSE(close(0u, UINT_MAX));
}

void testComplexFloat() {
    using Complex = std::complex<float>;

    EXPECT_TRUE(close(Complex(1.0f, 2.0f), Complex(1.0f, 2.0f)));
    EXPECT_TRUE(close(Complex(0.0f, 0.0f), Complex(0.0f, 0.0f)));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(close(Complex(nan, 0.0f), Complex(1.0f, 0.0f)));
    EXPECT_FALSE(close(Complex(0.0f, nan), Complex(0.0f, 1.0f)));
    EXPECT_FALSE(close(Complex(nan, nan), Complex(nan, nan)));

    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(close(Complex(inf, 0.0f), Complex(inf, 0.0f)));
    EXPECT_TRUE(close(Complex(0.0f, inf), Complex(0.0f, inf)));

    EXPECT_FALSE(close(Complex(1.0f, 0.0f), Complex(1.001f, 0.0f)));
    EXPECT_FALSE(close(Complex(0.0f, 1.0f), Complex(0.0f, 1.001f)));

    EXPECT_TRUE(close(Complex(1.0f, 1.0f), Complex(1.0002f, 1.0002f)));

    EXPECT_TRUE(close(Complex(1.0f, 1.0f), Complex(1.0002f, 1.0002f)));
    EXPECT_TRUE(close(Complex(1.0002f, 1.0002f), Complex(1.0f, 1.0f)));
    EXPECT_FALSE(close(Complex(1.0f, 1.0f), Complex(1.001f, 1.001f)));
    EXPECT_FALSE(close(Complex(1.001f, 1.001f), Complex(1.0f, 1.0f)));

    EXPECT_FALSE(close(Complex(1.0f, 1.0f), Complex(1.0002f, 1.001f)));
    EXPECT_FALSE(close(Complex(1.0f, 1.0f), Complex(1.001f, 1.0002f)));
}

void testComplexDouble() {
    using Complex = std::complex<double>;

    EXPECT_TRUE(close(Complex(1.0, 2.0), Complex(1.0, 2.0)));
    EXPECT_TRUE(close(Complex(0.0, 0.0), Complex(0.0, 0.0)));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(close(Complex(nan, 0.0), Complex(1.0, 0.0)));
    EXPECT_FALSE(close(Complex(0.0, nan), Complex(0.0, 1.0)));
    EXPECT_FALSE(close(Complex(nan, nan), Complex(nan, nan)));

    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(close(Complex(inf, 0.0), Complex(inf, 0.0)));
    EXPECT_TRUE(close(Complex(0.0, inf), Complex(0.0, inf)));

    EXPECT_FALSE(close(Complex(1.0, 0.0), Complex(1.0 + 1e-9, 0.0)));
    EXPECT_FALSE(close(Complex(0.0, 1.0), Complex(0.0, 1.0 + 1e-9)));

    EXPECT_TRUE(close(Complex(1.0, 1.0), Complex(1.0 + 2e-12, 1.0 + 2e-12)));

    EXPECT_TRUE(close(Complex(1.0, 1.0), Complex(1.0 + 2e-12, 1.0 + 2e-12)));
    EXPECT_TRUE(close(Complex(1.0 + 2e-12, 1.0 + 2e-12), Complex(1.0, 1.0)));
    EXPECT_FALSE(close(Complex(1.0, 1.0), Complex(1.0 + 1e-9, 1.0 + 1e-9)));
    EXPECT_FALSE(close(Complex(1.0 + 1e-9, 1.0 + 1e-9), Complex(1.0, 1.0)));

    EXPECT_FALSE(close(Complex(1.0, 1.0), Complex(1.0 + 2e-12, 1.0 + 1e-9)));
    EXPECT_FALSE(close(Complex(1.0, 1.0), Complex(1.0 + 1e-9, 1.0 + 2e-12)));
}

void testExplicitToleranceContract() {
    const auto defaults = roc::host_validation::defaultComparisonOptions(ScalarType::Float32);
    EXPECT_FALSE(roc::host_validation::valuesClose(1.0f, 1.02f, defaults));

    const auto overridden =
        roc::host_validation::defaultComparisonOptions(ScalarType::Float32, 0.01);
    EXPECT_TRUE(roc::host_validation::valuesClose(1.0f, 1.02f, overridden));

    auto exactBoundary = roc::host_validation::defaultComparisonOptions(ScalarType::Float32, 0.5);
    EXPECT_FALSE(roc::host_validation::valuesClose(0.0f, 1.0f, exactBoundary));
    exactBoundary.strictTolerance = false;
    EXPECT_TRUE(roc::host_validation::valuesClose(0.0f, 1.0f, exactBoundary));
}

void runExpectationSet(std::string_view name, void (*test)()) {
    currentExpectationSet = name;
    try {
        test();
    } catch (const std::exception& exception) {
        ++failureCount;
        std::cerr << name << ": unexpected exception: " << exception.what() << '\n';
    }
}
}  // namespace

int main() {
    runExpectationSet("Float32", testFloat);
    runExpectationSet("Float64", testDouble);
    runExpectationSet("Float16", testFloat16);
    runExpectationSet("BFloat16", testBFloat16);
    runExpectationSet("Float8E4M3", testFloat8E4M3);
    runExpectationSet("Float8E5M2", testFloat8E5M2);
    runExpectationSet("Float8E4M3Fnuz", testFloat8E4M3Fnuz);
    runExpectationSet("Float8E5M2Fnuz", testFloat8E5M2Fnuz);
    runExpectationSet("Int8", testInt8);
    runExpectationSet("Int32", testInt);
    runExpectationSet("UInt32", testUnsignedInt);
    runExpectationSet("ComplexFloat32", testComplexFloat);
    runExpectationSet("ComplexFloat64", testComplexDouble);

    if (expectationCount != 222 || expectedTrueCount != 111 || expectedFalseCount != 111) {
        ++failureCount;
        std::cerr << "Migrated expectation inventory mismatch: " << expectationCount << " total, "
                  << expectedTrueCount << " true, " << expectedFalseCount << " false.\n";
    }

    const size_t migratedExpectationCount = expectationCount;
    runExpectationSet("ExplicitToleranceContract", testExplicitToleranceContract);
    if (expectationCount != 226) {
        ++failureCount;
        std::cerr << "Comparison-policy expectation inventory mismatch: " << expectationCount
                  << " total after explicit-tolerance checks.\n";
    }

    if (failureCount != 0) return 1;

    std::cout << "Passed all " << migratedExpectationCount
              << " migrated comparison-policy expectations and "
              << expectationCount - migratedExpectationCount
              << " explicit-tolerance expectations.\n";
    return 0;
}
