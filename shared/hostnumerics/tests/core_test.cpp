// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <roc/hostnumerics/detail/prime.hpp>
#include <roc/hostnumerics/index_selection.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <roc/hostnumerics/shape.hpp>
#include <roc/hostnumerics/version.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace roc::hostnumerics {
void PrintTo(const Shape& shape, std::ostream* output) {
    *output << "Shape{";
    for (size_t axis = 0; axis < shape.rank(); ++axis) {
        if (axis != 0) *output << ", ";
        *output << shape[axis];
    }
    *output << '}';
}
}  // namespace roc::hostnumerics

namespace {
using namespace roc::hostnumerics;

TEST(Core, Version) {
    EXPECT_EQ(version(), "0.1.0");
    EXPECT_EQ(scalarTypeInfo(ScalarType::Float16).storageBits, 16);
}

TEST(Core, ScalarTypes) {
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

// Tutorial: choosing a traversal order with Permutation.
//
// An axis order lists every axis once, from slowest (outermost loop) to fastest
// (innermost loop). Shape and Layout take orders only as Permutations, so a
// vector of flat indices or extents cannot be passed where an order is meant.
TEST(PermutationTutorial, ColumnMajorMatrixIndexing) {
    // A 2x3 matrix stored column-major: rows change fastest, so the column axis
    // (1) is outermost and the row axis (0) innermost.
    const Shape matrix{2, 3};
    const Permutation columnMajor = Permutation::reverse(matrix.rank());
    EXPECT_EQ(columnMajor, (Permutation{1, 0}));

    // Element (row 1, column 2) is the last of six in either order, while
    // (row 1, column 0) is second in column-major order and fourth in row-major.
    const std::array<size_t, 2> lastElement{1, 2};
    const std::array<size_t, 2> secondRowFirstColumn{1, 0};
    EXPECT_EQ(matrix.linearIndex(lastElement, columnMajor), 5);
    EXPECT_EQ(matrix.linearIndex(secondRowFirstColumn, columnMajor), 1);
    EXPECT_EQ(matrix.linearIndex(secondRowFirstColumn, Permutation::identity(matrix.rank())), 3);

    // The matching contiguous storage has unit stride along the fastest axis.
    EXPECT_EQ(Layout::contiguous(matrix, columnMajor), (Layout(matrix, {1, 2}, 0)));
}

TEST(PermutationTutorial, ReorderingExtentsAndUndoingIt) {
    // apply() reorders values by position: result[i] == values[order[i]]. With
    // an axis order, it lists the extents from the outermost loop inward.
    const Shape shape{2, 3, 5};
    const Permutation order{1, 2, 0};
    const std::vector<size_t> loopExtents = order.apply(shape.dimensions());
    EXPECT_EQ(loopExtents, (std::vector<size_t>{3, 5, 2}));

    // inverse() undoes the reordering, recovering each axis's original extent.
    EXPECT_EQ(order.inverse(), (Permutation{2, 0, 1}));
    EXPECT_EQ(order.inverse().apply(loopExtents), (std::vector<size_t>{2, 3, 5}));
}

template <typename T>
constexpr bool acceptsBracedPair = requires(void (*sink)(T)) { sink({1, 0}); };

template <typename Indices>
constexpr bool acceptsIndicesAsOrder =
    requires(const Shape& shape, const Indices& indices) { shape.linearIndex(indices, indices); };

TEST(PermutationCpp, ConstructionValidatesEntries) {
    static_assert(std::ranges::random_access_range<Permutation>);
    static_assert(!std::is_default_constructible_v<Permutation>);
    static_assert(!std::is_convertible_v<std::vector<size_t>, Permutation>);
    static_assert(!std::is_convertible_v<std::span<const size_t>, Permutation>);
    static_assert(!acceptsBracedPair<const Permutation&>);
    static_assert(!acceptsIndicesAsOrder<std::vector<size_t>>);

    const std::array<size_t, 3> entries{2, 0, 1};
    for (const Permutation& permutation :
         {Permutation{2, 0, 1}, Permutation(std::vector<size_t>{2, 0, 1}),
          Permutation(std::span<const size_t>(entries))}) {
        EXPECT_EQ(permutation.size(), 3);
        EXPECT_EQ(permutation[0], 2);
        EXPECT_EQ(std::vector<size_t>(permutation.begin(), permutation.end()),
                  (std::vector<size_t>{2, 0, 1}));
        EXPECT_TRUE(std::ranges::equal(permutation.entries(), entries));
    }
    EXPECT_EQ(Permutation(std::vector<size_t>{}).size(), 0);
    for (const std::vector<size_t>& invalid :
         {std::vector<size_t>{0, 0}, std::vector<size_t>{1, 2}, std::vector<size_t>{1},
          std::vector<size_t>{0, 1, std::numeric_limits<size_t>::max()}})
        EXPECT_THROW(Permutation{invalid}, std::runtime_error);
    try {
        const Permutation repeated{0, 2, 2};
        FAIL() << "Expected a repeated entry to be rejected.";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(),
                     "Permutation entry 2 at position 2 is repeated or outside [0, 3); each "
                     "value must appear exactly once.");
    }
}

TEST(PermutationCpp, NamedOrdersInverseAndApply) {
    for (size_t size : {0, 1, 4}) {
        const Permutation identity = Permutation::identity(size);
        const Permutation reverse = Permutation::reverse(size);
        for (size_t position = 0; position < size; ++position) {
            EXPECT_EQ(identity[position], position);
            EXPECT_EQ(reverse[position], size - 1 - position);
        }
        EXPECT_EQ(identity.inverse(), identity);
        EXPECT_EQ(reverse.inverse(), reverse);
    }
    const Permutation permutation{3, 0, 2, 1};
    const Permutation inverse = permutation.inverse();
    for (size_t position = 0; position < permutation.size(); ++position)
        EXPECT_EQ(inverse[permutation[position]], position);
    const std::vector<std::string> values{"a", "b", "c", "d"};
    EXPECT_EQ(permutation.apply(values), (std::vector<std::string>{"d", "a", "c", "b"}));
    EXPECT_EQ(inverse.apply(permutation.apply(values)), values);
    EXPECT_THROW(permutation.apply(std::vector<int>{1, 2, 3}), std::runtime_error);
    EXPECT_NE(permutation, inverse);
}

TEST(PermutationCpp, ApplyArraysSpansAndBracedLists) {
    const Permutation order{2, 0, 1};
    std::array<int, 3> values{10, 20, 30};
    const std::vector<int> expected{30, 10, 20};
    EXPECT_EQ(order.apply(values), expected);
    EXPECT_EQ(order.apply((std::array<int, 3>{10, 20, 30})), expected);
    EXPECT_EQ(order.apply(std::span<int>(values)), expected);
    EXPECT_EQ(order.apply(std::span<const int>(values)), expected);
    EXPECT_EQ(order.apply(std::span(values)), expected);
    EXPECT_EQ(order.apply(std::span<const int, 3>(values)), expected);
    EXPECT_EQ(order.apply({10, 20, 30}), expected);
    const int array[]{10, 20, 30};
    EXPECT_EQ(order.apply(array), expected);

    const Permutation empty = Permutation::identity(0);
    EXPECT_TRUE(empty.apply(std::array<int, 0>{}).empty());
    EXPECT_TRUE(empty.apply(std::span<const int>{}).empty());
    EXPECT_TRUE(empty.apply(std::initializer_list<int>{}).empty());
    EXPECT_THROW(order.apply(std::array<int, 0>{}), std::runtime_error);
    EXPECT_THROW(order.apply(std::span(values).first(2)), std::runtime_error);
    EXPECT_THROW(order.apply({10, 20}), std::runtime_error);
}

TEST(PermutationCpp, ApplyBooleanVector) {
    const Permutation order{2, 0, 1};
    const std::vector<bool> values{true, false, false};
    EXPECT_EQ(order.apply(values), (std::vector<bool>{false, true, false}));
    EXPECT_EQ(order.inverse().apply(order.apply(values)), values);
    EXPECT_TRUE(Permutation::identity(0).apply(std::vector<bool>{}).empty());
    EXPECT_THROW(order.apply(std::vector<bool>{true}), std::runtime_error);
}

TEST(ShapeCpp, ConstructionAndOwnership) {
    EXPECT_EQ(::testing::PrintToString(Shape{}), "Shape{}");
    EXPECT_EQ(::testing::PrintToString(Shape{2, 3, 4}), "Shape{2, 3, 4}");
    static_assert(!std::is_constructible_v<Shape, size_t>);
    static_assert(!std::is_convertible_v<std::vector<size_t>, Shape>);
    static_assert(!std::is_convertible_v<std::span<const size_t>, Shape>);
    EXPECT_EQ(Shape{}, Shape({}));
    EXPECT_EQ(Shape{7}.rank(), 1);
    std::vector<size_t> source{2, 3, 4};
    const Shape copied(source), spanCopy{std::span<const size_t>(source)};
    source[0] = 9;
    EXPECT_EQ(copied, (Shape{2, 3, 4}));
    EXPECT_EQ(spanCopy, copied);
    EXPECT_EQ(Shape(std::move(source)), (Shape{9, 3, 4}));
    EXPECT_EQ(Shape(std::array<size_t, 3>{2, 3, 4}), copied);
    const size_t maximum = std::numeric_limits<size_t>::max();
    const std::vector<size_t> oversized{maximum, 2};
    EXPECT_THROW((Shape(oversized)), std::runtime_error);
    EXPECT_THROW((Shape{maximum, 2}), std::runtime_error);
    EXPECT_THROW(Shape(std::span<const size_t>(oversized)), std::runtime_error);
    EXPECT_THROW((Shape(std::vector<size_t>{maximum, 2})), std::runtime_error);
    for (const auto& extents :
         {std::vector<size_t>{0, maximum, 2}, std::vector<size_t>{maximum, 0, 2},
          std::vector<size_t>{maximum, 2, 0}}) {
        EXPECT_EQ(Shape(extents).elementCount(), 0);
        EXPECT_EQ(Shape(std::span<const size_t>(extents)).elementCount(), 0);
    }
    EXPECT_EQ(Shape{}.elementCount(), 1);
}

TEST(ShapeCpp, DimensionAccess) {
    const Shape shape{2, 3, 5};
    const std::array<size_t, 3> expected{2, 3, 5};
    static_assert(noexcept(shape[0]));
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        EXPECT_EQ(shape[axis], expected[axis]);
        EXPECT_EQ(shape.extent(axis), expected[axis]);
    }
    EXPECT_THROW(shape.extent(shape.rank()), std::runtime_error);
}

TEST(ShapeCpp, ArrayArguments) {
    const Shape shape{2, 3, 5};
    // C++20 span parameters accept std::array, but not braced initializer lists.
    const std::array<size_t, 3> axes{2, 0, 2};
    EXPECT_EQ(shape.fromDimensions(axes), (Shape{5, 2, 5}));

    const std::array<size_t, 3> indices{1, 2, 3};
    const Permutation order{1, 2, 0};
    const std::vector<size_t> expected(indices.begin(), indices.end());
    EXPECT_EQ(shape.linearIndexLastDimensionFastest(indices), 28);
    EXPECT_EQ(shape.coordinatesLastDimensionFastest(28), expected);
    EXPECT_EQ(shape.linearIndex(indices, order), 27);
    EXPECT_EQ(shape.coordinates(27, order), expected);
}

TEST(ShapeCpp, CachedCountAcrossCopiesAndMoves) {
    for (const Shape original :
         {Shape{}, Shape{2, 3, 5}, Shape{std::numeric_limits<size_t>::max(), 2, 0}}) {
        Shape copied(original);
        EXPECT_EQ(copied, original);
        EXPECT_EQ(copied.elementCount(), original.elementCount());
        Shape moved(std::move(copied));
        EXPECT_EQ(moved, original);
        EXPECT_EQ(moved.elementCount(), original.elementCount());
        // The move contract defines a scalar source that can be queried.
        // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_EQ(copied.rank(), 0);
        EXPECT_EQ(copied.elementCount(), 1);
        EXPECT_TRUE(copied.coordinatesLastDimensionFastest(0).empty());
        EXPECT_THROW(copied.coordinatesLastDimensionFastest(1), std::runtime_error);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

        Shape assigned{7};
        assigned = original;
        EXPECT_EQ(assigned, original);
        EXPECT_EQ(assigned.elementCount(), original.elementCount());
        Shape moveAssigned{9};
        moveAssigned = std::move(assigned);
        EXPECT_EQ(moveAssigned, original);
        EXPECT_EQ(moveAssigned.elementCount(), original.elementCount());
        // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_EQ(assigned.rank(), 0);
        EXPECT_EQ(assigned.elementCount(), 1);
        EXPECT_TRUE(assigned.coordinatesLastDimensionFastest(0).empty());
        EXPECT_THROW(assigned.coordinatesLastDimensionFastest(1), std::runtime_error);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

        Shape& alias = moveAssigned;
        moveAssigned = std::move(alias);
        EXPECT_EQ(moveAssigned, original);
        EXPECT_EQ(moveAssigned.elementCount(), original.elementCount());
    }
}

TEST(PrimeCpp, InclusiveSearchAndPrimality) {
    const std::array<size_t, 25> primes{2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
                                        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    for (size_t value = 0; value <= primes.back(); ++value) {
        SCOPED_TRACE(value);
        EXPECT_EQ(detail::isPrime(value), std::binary_search(primes.begin(), primes.end(), value));
        EXPECT_EQ(detail::getNextPrime(value),
                  *std::lower_bound(primes.begin(), primes.end(), value));
    }
    EXPECT_EQ(detail::getNextPrime(121), 127);
    EXPECT_FALSE(detail::getNextPrime(std::numeric_limits<size_t>::max()));
}

TEST(IndexSelectionCpp, StrictAndNamedOrders) {
    static_assert(!std::is_default_constructible_v<IndexSelection>);
    const Shape shape{2, 3};
    const auto ascending = IndexSelection(std::vector<size_t>({1}), std::vector<size_t>{0, 1});
    const auto descending = IndexSelection(std::vector<size_t>({1}), std::vector<size_t>{1, 0});
    EXPECT_EQ(
        shape.coordinates(shape.selectedIndices(ascending).front(), shape.indexOrder(ascending)),
        (std::vector<size_t>{0, 1}));
    EXPECT_EQ(
        shape.coordinates(shape.selectedIndices(descending).front(), shape.indexOrder(descending)),
        (std::vector<size_t>{1, 0}));
    const auto scalar = IndexSelection(IndexRange::all(), std::vector<size_t>{});
    EXPECT_TRUE(Shape{}.indexOrder(scalar).empty());
    EXPECT_THROW(shape.indexOrder(scalar), std::runtime_error);
    EXPECT_THROW(Shape{6}.indexOrder(ascending), std::runtime_error);
    for (const auto& order :
         {std::vector<size_t>{0, 0}, std::vector<size_t>{1}, std::vector<size_t>{0, 2}}) {
        EXPECT_THROW(IndexSelection(IndexRange::all(), order), std::runtime_error);
        EXPECT_THROW(IndexSelection(IndexRange(0, 2, 3), order), std::runtime_error);
        EXPECT_THROW(IndexSelection(std::vector<size_t>({1}), order), std::runtime_error);
        EXPECT_THROW(IndexSelection(primeIndexRange(6, 6, 0), order), std::runtime_error);
    }
    const std::array<IndexSelection, 4> named{
        IndexSelection(IndexRange::all(), LastDimensionFastest{}),
        IndexSelection(IndexRange(0, 2, 3), LastDimensionFastest{}),
        IndexSelection(std::vector<size_t>({1}), LastDimensionFastest{}),
        IndexSelection(primeIndexRange(6, 6, 2), LastDimensionFastest{})};
    for (const auto& selection : named) {
        EXPECT_TRUE(Shape{}.indexOrder(selection).empty());
        EXPECT_EQ(Shape({2, 3}).indexOrder(selection), (std::vector<size_t>{0, 1}));
        EXPECT_EQ(Shape({2, 3, 4}).indexOrder(selection), (std::vector<size_t>{0, 1, 2}));
        EXPECT_EQ(shape.coordinates(1, Shape({2, 3}).indexOrder(selection)),
                  (std::vector<size_t>{0, 1}));
    }
}

TEST(IndexSelectionCpp, OwnsSpanOrdersAndIndices) {
    auto selections = [] {
        std::array<size_t, 3> order{2, 1, 0};
        std::vector<size_t> indices{7, 1, 7};
        std::array<IndexSelection, 4> result{IndexSelection(IndexRange::all(), order),
                                             IndexSelection(IndexRange(0, 2, 3), order),
                                             IndexSelection(std::vector<size_t>(indices), order),
                                             IndexSelection(primeIndexRange(24, 24, 5), order)};
        order[0] = 9;
        indices[0] = 99;
        return result;
    }();
    for (const auto& selection : selections) {
        const auto order = Shape({2, 3, 4}).indexOrder(selection);
        EXPECT_EQ((std::vector<size_t>(order.begin(), order.end())),
                  (std::vector<size_t>{2, 1, 0}));
    }
    auto copied = selections[2];
    selections[2] = IndexSelection(IndexRange::all(), LastDimensionFastest{});
    EXPECT_EQ(Shape({2, 2, 2}).selectedIndices(copied), (std::vector<size_t>{1, 7}));
    const auto moved = std::move(copied);
    EXPECT_EQ(Shape({2, 2, 2}).selectedIndices(moved), (std::vector<size_t>{1, 7}));
    EXPECT_EQ(Shape({2, 2, 2}).indexOrder(moved), (std::vector<size_t>{2, 1, 0}));
}

TEST(IndexSelectionCpp, SizeBoundaries) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    const Shape shape{maximum};
    EXPECT_EQ(shape.selectedCount(IndexSelection(IndexRange::all(), LastDimensionFastest{})),
              maximum);
    EXPECT_EQ(shape.selectedIndices(
                  IndexSelection(IndexRange(maximum - 2, 3, maximum), LastDimensionFastest{})),
              (std::vector<size_t>{maximum - 2}));
    EXPECT_EQ(shape.selectedIndices(
                  IndexSelection(IndexRange(1, maximum - 2, maximum), LastDimensionFastest{})),
              (std::vector<size_t>{1, maximum - 1}));
    EXPECT_THROW(primeIndexRange(2, maximum, 1), std::runtime_error);
}

TEST(IndexSelectionCpp, VisitsCoordinatesAndChecksBoundsBeforeCallbacks) {
    const Shape shape{2, 4};
    const std::array<size_t, 2> order{0, 1};
    for (const auto& selection : {IndexSelection(IndexRange(1, 2, 3), order),
                                  IndexSelection(std::vector<size_t>{5, 1, 3, 5}, order)}) {
        std::vector<std::vector<size_t>> visited;
        shape.forEachCoordinate(selection, [&](size_t, std::span<const size_t> coordinates) {
            visited.emplace_back(coordinates.begin(), coordinates.end());
        });
        EXPECT_EQ(visited, (std::vector<std::vector<size_t>>{{0, 1}, {0, 3}, {1, 1}}));
    }
    EXPECT_TRUE(shape.selectsAll(IndexSelection(IndexRange(0, 1, 8), order)));
    size_t calls = 0;
    const IndexSelection invalid(std::vector<size_t>{0, 8}, order);
    EXPECT_THROW(
        shape.forEachCoordinate(invalid, [&](size_t, std::span<const size_t>) { ++calls; }),
        std::runtime_error);
    EXPECT_EQ(calls, 0);
    EXPECT_THROW(IndexRange(0, 0, 3), std::runtime_error);
}

}  // namespace
