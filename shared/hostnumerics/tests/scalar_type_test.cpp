// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <initializer_list>
#include <memory>
#include <roc/hostnumerics/scalar_type.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

TEST(Core, ScalarTypes) {
    using namespace roc::hostnumerics;
    // nativeScalarType<std::string> is a compilation error. Use NativeScalar<T>
    // to query whether T has a mapping without causing an error.
    static_assert(nativeScalarType<bool> == ScalarType::Boolean);
    static_assert(nativeScalarType<uint8_t> == ScalarType::UInt8);
    static_assert(nativeScalarType<int8_t> == ScalarType::Int8);
    static_assert(nativeScalarType<uint16_t> == ScalarType::UInt16);
    static_assert(nativeScalarType<int16_t> == ScalarType::Int16);
    static_assert(nativeScalarType<uint32_t> == ScalarType::UInt32);
    static_assert(nativeScalarType<int32_t> == ScalarType::Int32);
    static_assert(nativeScalarType<uint64_t> == ScalarType::UInt64);
    static_assert(nativeScalarType<int64_t> == ScalarType::Int64);
    static_assert(nativeScalarType<float> == ScalarType::Float32);
    static_assert(nativeScalarType<double> == ScalarType::Float64);
    static_assert(nativeScalarType<std::complex<float>> == ScalarType::ComplexFloat32);
    static_assert(nativeScalarType<std::complex<double>> == ScalarType::ComplexFloat64);
    static_assert(nativeScalarType<const float&> == ScalarType::Float32);
    static_assert(nativeScalarType<const volatile int32_t&> == ScalarType::Int32);
    static_assert(nativeScalarType<volatile std::complex<float>&&> == ScalarType::ComplexFloat32);
    static_assert(NativeScalar<bool> && NativeScalar<int32_t> &&
                  NativeScalar<std::complex<double>>);
    static_assert(NativeScalar<const volatile float&>);
    static_assert(!NativeScalar<char>);
    static_assert(!NativeScalar<void>);
    static_assert(!NativeScalar<float*>);
    static_assert(!NativeScalar<float[2]>);
    static_assert(!NativeScalar<long double>);
    static_assert(!NativeScalar<std::string>);
    static_assert(!NativeScalar<std::string_view>);

    static_assert(scalarTypeInfo(ScalarType::Float32).exponentBits == 8);

    // IEEE binary16 has a five-bit exponent, ten fraction bits, and bias 15.
    const auto& half = scalarTypeInfo(ScalarType::Float16);
    EXPECT_EQ(half.name, "f16");
    EXPECT_EQ(half.category, ScalarCategory::FloatingPoint);
    EXPECT_EQ(half.storageBits, 16);
    EXPECT_EQ(half.exponentBits, 5);
    EXPECT_EQ(half.mantissaBits, 10);
    EXPECT_EQ(half.exponentBias, 15);
    EXPECT_TRUE(half.supportsNaN);
    EXPECT_TRUE(half.supportsInfinity);
    EXPECT_FALSE(half.isPacked());
    const auto& complex = scalarTypeInfo(ScalarType::ComplexFloat32);
    EXPECT_EQ(complex.category, ScalarCategory::Complex);
    EXPECT_EQ(complex.storageBits, 64);
    EXPECT_EQ(complex.exponentBits, 8);
    EXPECT_EQ(complex.mantissaBits, 23);
    const auto& packed = scalarTypeInfo(ScalarType::Float6E3M2);
    EXPECT_EQ(packed.storageBits, 6);
    EXPECT_TRUE(packed.isPacked());
    EXPECT_FALSE(packed.supportsNaN);
    EXPECT_FALSE(packed.supportsInfinity);
    EXPECT_EQ(scalarTypeInfo(ScalarType::Int4).category, ScalarCategory::SignedInteger);
    EXPECT_EQ(scalarTypeInfo(ScalarType::E8M0).category, ScalarCategory::FloatingPoint);

    // Check each identifier independently of its position in the metadata table.
    constexpr std::pair<ScalarType, std::string_view> expectedNames[] = {
        {ScalarType::Boolean, "bool"},
        {ScalarType::Int4, "i4"},
        {ScalarType::Int8, "i8"},
        {ScalarType::Int16, "i16"},
        {ScalarType::Int32, "i32"},
        {ScalarType::Int64, "i64"},
        {ScalarType::UInt8, "u8"},
        {ScalarType::UInt16, "u16"},
        {ScalarType::UInt32, "u32"},
        {ScalarType::UInt64, "u64"},
        {ScalarType::Float4E2M1, "f4e2m1"},
        {ScalarType::Float6E2M3, "f6e2m3"},
        {ScalarType::Float6E3M2, "f6e3m2"},
        {ScalarType::Float8E4M3, "f8e4m3"},
        {ScalarType::Float8E5M2, "f8e5m2"},
        {ScalarType::Float8E4M3Fnuz, "f8e4m3fnuz"},
        {ScalarType::Float8E5M2Fnuz, "f8e5m2fnuz"},
        {ScalarType::E4M3, "e4m3"},
        {ScalarType::E5M3, "e5m3"},
        {ScalarType::E8M0, "e8m0"},
        {ScalarType::E8M0Zero, "e8m0_zero"},
        {ScalarType::Float16, "f16"},
        {ScalarType::BFloat16, "bf16"},
        {ScalarType::Float32, "f32"},
        {ScalarType::Float64, "f64"},
        {ScalarType::ComplexFloat32, "c64"},
        {ScalarType::ComplexFloat64, "c128"},
    };
    static_assert(std::size(expectedNames) == scalarTypeCount);
    for (const auto& [type, name] : expectedNames) {
        SCOPED_TRACE(name);
        EXPECT_EQ(scalarTypeName(type), name);
        EXPECT_LT(scalarTypeInfo(type).category, ScalarCategory::Count);
    }

    for (ScalarType type :
         {ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0, ScalarType::E8M0Zero}) {
        SCOPED_TRACE(scalarTypeName(type));
        const auto& info = scalarTypeInfo(type);
        EXPECT_EQ(info.category, ScalarCategory::FloatingPoint);
        EXPECT_FALSE(info.isSigned);
    }
    EXPECT_TRUE(scalarTypeInfo(ScalarType::Float32).isSigned);
    EXPECT_TRUE(scalarTypeInfo(ScalarType::ComplexFloat32).isSigned);
    EXPECT_TRUE(scalarTypeInfo(ScalarType::Int4).isSigned);
    EXPECT_FALSE(scalarTypeInfo(ScalarType::UInt8).isSigned);
    EXPECT_FALSE(scalarTypeInfo(ScalarType::Boolean).isSigned);

    for (ScalarType invalid : {ScalarType::Count, static_cast<ScalarType>(0xffff)}) {
        EXPECT_FALSE(isConcreteScalarType(invalid))
            << "ScalarType value: " << static_cast<int>(invalid);
    }
}

namespace {
using namespace roc::hostnumerics;

template <typename Tag, typename T>
void sqrtElements(std::vector<T>& elements) {
    // Only Float32 and Float64 store native floating-point values. Other
    // floating-point formats need decoding before arithmetic.
    if constexpr (std::is_floating_point_v<typename Tag::Storage> &&
                  std::is_same_v<typename Tag::Storage, T>) {
        for (auto& element : elements) element = std::sqrt(element);
    } else {
        throw std::runtime_error("Expected matching native floating-point elements.");
    }
}

TEST(Core, ScalarDispatchSqrt) {
    auto squareRoots = []<typename Tag>(auto& elements) { sqrtElements<Tag>(elements); };

    std::vector<float> floats{0.0f, 0.25f, 4.0f, 9.0f};
    visitScalarType(ScalarType::Float32, squareRoots, floats);
    EXPECT_EQ(floats, (std::vector<float>{0.0f, 0.5f, 2.0f, 3.0f}));

    std::vector<double> doubles{0.0, 0.25, 4.0, 9.0};
    visitScalarType(ScalarType::Float64, squareRoots, doubles);
    EXPECT_EQ(doubles, (std::vector<double>{0.0, 0.5, 2.0, 3.0}));

    std::vector<int> integers{4, 9};
    EXPECT_THROW(visitScalarType(ScalarType::Int32, squareRoots, integers), std::runtime_error);
    EXPECT_EQ(integers, (std::vector<int>{4, 9}));
    // Half-precision storage contains encoded bits, not native arithmetic values.
    std::vector<uint16_t> halfBits{0x4400, 0x4880};
    EXPECT_THROW(visitScalarType(ScalarType::Float16, squareRoots, halfBits), std::runtime_error);
    EXPECT_EQ(halfBits, (std::vector<uint16_t>{0x4400, 0x4880}));
    EXPECT_THROW(visitScalarType(ScalarType::Float64, squareRoots, floats), std::runtime_error);
}

TEST(Core, ScalarDispatchTags) {
    // Verify storage independently of enum order, especially encodings sharing
    // a storage type and packed formats that cannot occupy one C++ object.
    const std::pair<ScalarType, std::type_index> storageTypes[] = {
        {ScalarType::Boolean, typeid(uint8_t)},
        {ScalarType::Int4, typeid(void)},
        {ScalarType::Int8, typeid(int8_t)},
        {ScalarType::Int16, typeid(int16_t)},
        {ScalarType::Int32, typeid(int32_t)},
        {ScalarType::Int64, typeid(int64_t)},
        {ScalarType::UInt8, typeid(uint8_t)},
        {ScalarType::UInt16, typeid(uint16_t)},
        {ScalarType::UInt32, typeid(uint32_t)},
        {ScalarType::UInt64, typeid(uint64_t)},
        {ScalarType::Float4E2M1, typeid(void)},
        {ScalarType::Float6E2M3, typeid(void)},
        {ScalarType::Float6E3M2, typeid(void)},
        {ScalarType::Float8E4M3, typeid(uint8_t)},
        {ScalarType::Float8E5M2, typeid(uint8_t)},
        {ScalarType::Float8E4M3Fnuz, typeid(uint8_t)},
        {ScalarType::Float8E5M2Fnuz, typeid(uint8_t)},
        {ScalarType::E4M3, typeid(uint8_t)},
        {ScalarType::E5M3, typeid(uint8_t)},
        {ScalarType::E8M0, typeid(uint8_t)},
        {ScalarType::E8M0Zero, typeid(uint8_t)},
        {ScalarType::Float16, typeid(uint16_t)},
        {ScalarType::BFloat16, typeid(uint16_t)},
        {ScalarType::Float32, typeid(float)},
        {ScalarType::Float64, typeid(double)},
        {ScalarType::ComplexFloat32, typeid(std::complex<float>)},
        {ScalarType::ComplexFloat64, typeid(std::complex<double>)},
    };
    static_assert(std::size(storageTypes) == scalarTypeCount);
    for (const auto& [type, expected] : storageTypes) {
        SCOPED_TRACE(scalarTypeName(type));
        int calls = 0;
        EXPECT_EQ(visitScalarType(type,
                                  [&]<typename Tag>() {
                                      ++calls;
                                      EXPECT_EQ(Tag::type, type);
                                      return std::type_index(typeid(typename Tag::Storage));
                                  }),
                  expected);
        EXPECT_EQ(calls, 1);
    }
}

enum class VisitorQualification { LValue, ConstLValue, RValue };

// Reports which overload ran. The move-only member makes the visitor
// non-copyable, so dispatching an lvalue visitor shows that it is not copied.
struct QualifiedVisitor {
    std::unique_ptr<int> moveOnly;

    template <typename Tag>
    VisitorQualification operator()() & {
        return VisitorQualification::LValue;
    }
    template <typename Tag>
    VisitorQualification operator()() const& {
        return VisitorQualification::ConstLValue;
    }
    template <typename Tag>
    VisitorQualification operator()() && {
        return VisitorQualification::RValue;
    }
};

TEST(Core, ScalarDispatchForwarding) {
    QualifiedVisitor visitor;
    EXPECT_EQ(visitScalarType(ScalarType::Float32, visitor), VisitorQualification::LValue);
    EXPECT_EQ(visitScalarType(ScalarType::Float32, std::as_const(visitor)),
              VisitorQualification::ConstLValue);
    EXPECT_EQ(visitScalarType(ScalarType::Float32, std::move(visitor)),
              VisitorQualification::RValue);

    // A move-only rvalue argument reaches a visitor that takes it by value.
    auto argument = std::make_unique<int>(42);
    EXPECT_EQ(visitScalarType(
                  ScalarType::Int4, []<typename Tag>(std::unique_ptr<int> p) { return *p; },
                  std::move(argument)),
              42);
    EXPECT_EQ(argument, nullptr);
}

TEST(Core, ScalarDispatchReturnTypes) {
    int value = 5;
    auto identity = []<typename Tag>(auto&& argument) -> decltype(auto) {
        return std::forward<decltype(argument)>(argument);
    };
    static_assert(
        std::is_same_v<decltype(visitScalarType(ScalarType::UInt8, identity, value)), int&>);
    EXPECT_EQ(&visitScalarType(ScalarType::UInt8, identity, value), &value);
    static_assert(
        std::is_same_v<decltype(visitScalarType(ScalarType::Float16, identity, std::move(value))),
                       int&&>);
    int&& returned = visitScalarType(ScalarType::Float16, identity, std::move(value));
    EXPECT_EQ(&returned, &value);

    auto increment = []<typename Tag>(int& argument) { ++argument; };
    static_assert(std::is_void_v<decltype(visitScalarType(ScalarType::Boolean, increment, value))>);
    visitScalarType(ScalarType::Boolean, increment, value);
    EXPECT_EQ(value, 6);
}

TEST(Core, ScalarDispatchExceptions) {
    int calls = 0;
    auto visitor = [&calls]<typename Tag>(std::unique_ptr<int> argument) {
        ++calls;
        return *argument;
    };
    auto argument = std::make_unique<int>(42);
    // Invalid types are rejected before the visitor is called: calls stays zero,
    // and the argument passed with std::move is never moved from.
    // Deliberately construct unnamed enum values to test boundary rejection.
    // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
    const ScalarType invalidTypes[] = {ScalarType::Count,
                                       static_cast<ScalarType>(scalarTypeCount + 1),
                                       static_cast<ScalarType>(0xffff)};
    // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
    for (const auto type : invalidTypes) {
        try {
            visitScalarType(type, visitor, std::move(argument));
            FAIL() << "Invalid scalar type was accepted";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(
                std::string_view(error.what()).find(std::to_string(static_cast<unsigned>(type))),
                std::string_view::npos);
        }
        EXPECT_EQ(calls, 0);
        ASSERT_NE(argument, nullptr);
    }
    EXPECT_EQ(visitScalarType(ScalarType::BFloat16, visitor, std::move(argument)), 42);
    EXPECT_EQ(calls, 1);

    // A visitor's own failure retains its original type and payload.
    struct VisitorError {
        int marker;
    };
    try {
        visitScalarType(ScalarType::ComplexFloat64, []<typename Tag>() { throw VisitorError{37}; });
        FAIL() << "Visitor error was swallowed";
    } catch (const VisitorError& error) {
        EXPECT_EQ(error.marker, 37);
    }
}
}  // namespace
