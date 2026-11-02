// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <roc/hostnumerics/layout.hpp>
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
    const std::array<size_t, 3> order{1, 2, 0};
    const std::vector<size_t> expected(indices.begin(), indices.end());
    EXPECT_EQ(shape.linearIndex(indices), 28);
    EXPECT_EQ(shape.coordinates(28), expected);
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
        EXPECT_TRUE(copied.coordinates(0).empty());
        EXPECT_THROW(copied.coordinates(1), std::runtime_error);
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
        EXPECT_TRUE(assigned.coordinates(0).empty());
        EXPECT_THROW(assigned.coordinates(1), std::runtime_error);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

        Shape& alias = moveAssigned;
        moveAssigned = std::move(alias);
        EXPECT_EQ(moveAssigned, original);
        EXPECT_EQ(moveAssigned.elementCount(), original.elementCount());
    }
}

TEST(LayoutCpp, OwnedStridesAndArrayCoordinates) {
    std::vector<ptrdiff_t> strides{5, -1};
    const Layout layout(Shape{2, 3}, strides, 2);
    const Layout spanCopy(Shape{2, 3}, std::span<const ptrdiff_t>(strides), 2);
    strides[0] = 9;
    EXPECT_EQ(layout.stride(0), 5);
    EXPECT_EQ(spanCopy, layout);
    EXPECT_EQ((Layout(Shape{2, 3}, std::array<ptrdiff_t, 2>{5, -1}, 2)), layout);
    EXPECT_EQ((Layout(Shape{2, 3}, {5, -1}, 2)), layout);
    EXPECT_EQ(Layout(Shape{}, {}, 0), Layout{});
    const auto offset = layout.elementOffset(std::array<size_t, 2>{1, 2});
    ASSERT_TRUE(offset);
    EXPECT_EQ(*offset, 5);
}

TEST(LayoutCpp, FactoriesAndExplicitOrders) {
    const Shape shape{2, 3, 5};
    const Layout rowMajor(shape, {15, 5, 1}, 0);
    EXPECT_EQ(Layout::contiguous(shape), rowMajor);
    EXPECT_EQ(Layout::contiguousLastDimensionFastest(shape), rowMajor);
    EXPECT_EQ(Layout::contiguousFirstDimensionFastest(shape), (Layout(shape, {1, 2, 6}, 0)));
    const std::array<size_t, 3> order{1, 2, 0};
    const auto checked = Layout::getContiguousLayout(shape, order);
    ASSERT_TRUE(checked);
    EXPECT_EQ(*checked, (Layout(shape, {1, 10, 2}, 0)));
    EXPECT_EQ(Layout::contiguous(shape, order), *checked);
    EXPECT_EQ(Layout::contiguous(shape, {1, 2, 0}), *checked);
    const std::array<size_t, 3> coordinates{1, 2, 3};
    const auto offset = checked->elementOffset(coordinates);
    ASSERT_TRUE(offset);
    EXPECT_EQ(*offset, static_cast<ptrdiff_t>(shape.linearIndex(coordinates, order)));
    EXPECT_EQ(Layout::contiguous(Shape{}, {}), Layout{});
    EXPECT_EQ(Layout::contiguousLastDimensionFastest(Shape{2, 0, 3}),
              (Layout(Shape{2, 0, 3}, {0, 3, 1}, 0)));
    EXPECT_EQ(Layout::contiguousFirstDimensionFastest(Shape{2, 0, 3}),
              (Layout(Shape{2, 0, 3}, {1, 2, 0}, 0)));
}

TEST(LayoutCpp, InvalidStrideRanksAndDimensionsThrow) {
    const Shape shape{2, 3};
    EXPECT_THROW((Layout(shape, std::vector<ptrdiff_t>{1}, 0)), std::runtime_error);
    EXPECT_THROW(Layout(shape, {3, 1}, 0).stride(shape.rank()), std::runtime_error);
    EXPECT_THROW(Layout::contiguous(shape, {0}), std::runtime_error);
    EXPECT_THROW(Layout::contiguous(Shape{std::numeric_limits<size_t>::max()}), std::runtime_error);
}

TEST(LayoutCpp, CopiesAndMovesPreserveLayoutMetadata) {
    for (const Layout original :
         {Layout{}, Layout(Shape{}, {}, -7), Layout(Shape{2, 3}, {5, -1}, 2),
          Layout(Shape{2, 0}, {0, 1}, 4)}) {
        Layout copied(original);
        EXPECT_EQ(copied, original);
        Layout moved(std::move(copied));
        EXPECT_EQ(moved, original);
        // Layout moves reset the source to a scalar with zero offset.
        // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_TRUE(copied == Layout{});
        EXPECT_EQ(copied.shape().elementCount(), 1);
        EXPECT_TRUE(copied.strides().empty());
        const auto copiedOffset = copied.elementOffset({});
        ASSERT_TRUE(copiedOffset);
        EXPECT_EQ(*copiedOffset, 0);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

        Layout assigned;
        assigned = original;
        EXPECT_EQ(assigned, original);
        Layout moveAssigned(Shape{9}, {1}, 0);
        moveAssigned = std::move(assigned);
        EXPECT_EQ(moveAssigned, original);
        // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_TRUE(assigned == Layout{});
        EXPECT_EQ(assigned.shape().elementCount(), 1);
        EXPECT_TRUE(assigned.strides().empty());
        const auto assignedOffset = assigned.elementOffset({});
        ASSERT_TRUE(assignedOffset);
        EXPECT_EQ(*assignedOffset, 0);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        Layout& self = moveAssigned;
        moveAssigned = std::move(self);
        EXPECT_EQ(moveAssigned, original);
    }
}

TEST(LayoutCpp, ContiguousFailureIsReturnedBeforeBoundaryReporting) {
    using namespace roc::hostnumerics::detail;
    const auto result = Layout::getContiguousLayout(Shape{std::numeric_limits<size_t>::max()});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().error, LayoutError::ExtentOutOfRange);
    EXPECT_EQ(result.error().dimension, 0);
    EXPECT_EQ(result.error().value, std::numeric_limits<size_t>::max());
    EXPECT_EQ(result.error().limit, static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()));
    const auto valid = Layout::getContiguousLayout(Shape{2, 3});
    ASSERT_TRUE(valid);
    EXPECT_EQ(valid->stride(0), 3);
    EXPECT_EQ(valid->stride(1), 1);
    const auto invalidOrder = Layout::getContiguousLayout(Shape{2, 3}, std::array<size_t, 2>{1, 1});
    ASSERT_FALSE(invalidOrder);
    EXPECT_EQ(invalidOrder.error().error, LayoutError::InvalidOrder);
    EXPECT_EQ(invalidOrder.error().dimension, 1);
    EXPECT_EQ(invalidOrder.error().value, 1);
    EXPECT_EQ(invalidOrder.error().limit, 2);
    const auto outside = Layout::getContiguousLayout(Shape{2, 3}, {0, 2});
    ASSERT_FALSE(outside);
    EXPECT_EQ(outside.error().error, LayoutError::InvalidOrder);
    EXPECT_EQ(outside.error().value, 2);
    const auto rank = Layout::getContiguousLayout(Shape{2, 3}, {0});
    ASSERT_FALSE(rank);
    EXPECT_EQ(rank.error().error, LayoutError::OrderRankMismatch);
    EXPECT_EQ(rank.error().value, 1);
    EXPECT_EQ(rank.error().limit, 2);
    const size_t halfRange = (static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) + 1) / 2;
    const auto overflow = Layout::getContiguousLayout(Shape{2, halfRange});
    ASSERT_FALSE(overflow);
    EXPECT_EQ(overflow.error().error, LayoutError::ContiguousStrideOverflow);
    EXPECT_EQ(overflow.error().dimension, 0);
    static_assert(noexcept(std::declval<const Layout&>() == std::declval<const Layout&>()));
}

TEST(LayoutCpp, OffsetFailuresCarryContextWithoutThrowing) {
    using namespace roc::hostnumerics::detail;
    static_assert(noexcept(std::declval<const Layout&>().elementOffset({})));
    static_assert(std::is_nothrow_constructible_v<Expected<ptrdiff_t, LayoutFailure>, ptrdiff_t>);
    static_assert(std::is_nothrow_constructible_v<Expected<ptrdiff_t, LayoutFailure>,
                                                  Unexpected<LayoutFailure>>);
    const Layout layout(Shape{3, 3}, {std::numeric_limits<ptrdiff_t>::max(), 1}, 0);
    const auto rank = layout.elementOffset(std::array<size_t, 1>{0});
    ASSERT_FALSE(rank);
    EXPECT_EQ(rank.error().error, LayoutError::CoordinateRankMismatch);
    EXPECT_EQ(rank.error().value, 1);
    EXPECT_EQ(rank.error().limit, 2);
    const auto bounds = layout.elementOffset(std::array<size_t, 2>{2, 3});
    ASSERT_FALSE(bounds);
    EXPECT_EQ(bounds.error().error, LayoutError::CoordinateOutOfRange);
    EXPECT_EQ(bounds.error().dimension, 1);
    EXPECT_EQ(bounds.error().value, 3);
    EXPECT_EQ(bounds.error().limit, 3);
    const auto multiply = layout.elementOffset(std::array<size_t, 2>{2, 0});
    ASSERT_FALSE(multiply);
    EXPECT_EQ(multiply.error().error, LayoutError::OffsetMultiplyOverflow);
    EXPECT_EQ(multiply.error().dimension, 0);
    const auto add = layout.elementOffset(std::array<size_t, 2>{1, 1});
    ASSERT_FALSE(add);
    EXPECT_EQ(add.error().error, LayoutError::OffsetAddOverflow);
    EXPECT_EQ(add.error().dimension, 1);
    const auto secondAxis = Layout(Shape{2, 3}, {0, std::numeric_limits<ptrdiff_t>::max()}, 0)
                                .elementOffset(std::array<size_t, 2>{0, 2});
    ASSERT_FALSE(secondAxis);
    EXPECT_EQ(secondAxis.error().error, LayoutError::OffsetMultiplyOverflow);
    EXPECT_EQ(secondAxis.error().dimension, 1);
}

TEST(LayoutCpp, ReachableBoundsAndOverflowContext) {
    using namespace roc::hostnumerics::detail;
    static_assert(noexcept(std::declval<const Layout&>().getElementBounds()));
    for (const auto& [layout, expected] :
         {std::pair{Layout{}, std::pair<ptrdiff_t, ptrdiff_t>{0, 0}},
          std::pair{Layout(Shape{}, {}, -7), std::pair<ptrdiff_t, ptrdiff_t>{-7, -7}},
          std::pair{Layout(Shape{2, 3}, {-3, 1}, 3), std::pair<ptrdiff_t, ptrdiff_t>{0, 5}},
          std::pair{Layout(Shape{2, 3}, {0, 0}, 4), std::pair<ptrdiff_t, ptrdiff_t>{4, 4}},
          std::pair{Layout(Shape{2, 0}, {1, 1}, 9), std::pair<ptrdiff_t, ptrdiff_t>{0, -1}}}) {
        const auto bounds = layout.getElementBounds();
        ASSERT_TRUE(bounds);
        EXPECT_EQ(*bounds, expected);
    }
    const auto maximum = std::numeric_limits<ptrdiff_t>::max();
    const size_t halfRange = (static_cast<size_t>(maximum) + 1) / 2;
    const auto boundary =
        Layout(Shape{2, halfRange}, {static_cast<ptrdiff_t>(halfRange), 1}, 0).getElementBounds();
    ASSERT_TRUE(boundary);
    EXPECT_EQ(boundary->first, 0);
    EXPECT_EQ(boundary->second, maximum);
    const auto multiply = Layout(Shape{2, 3}, {0, maximum}, 0).getElementBounds();
    ASSERT_FALSE(multiply);
    EXPECT_EQ(multiply.error().error, LayoutError::OffsetMultiplyOverflow);
    EXPECT_EQ(multiply.error().dimension, 1);
    const auto add = Layout(Shape{2, 2}, {0, 1}, maximum).getElementBounds();
    ASSERT_FALSE(add);
    EXPECT_EQ(add.error().error, LayoutError::OffsetAddOverflow);
    EXPECT_EQ(add.error().dimension, 1);
    const auto minimum = std::numeric_limits<ptrdiff_t>::min();
    const auto negative = Layout(Shape{2}, {minimum}, 0).getElementBounds();
    ASSERT_TRUE(negative);
    EXPECT_EQ(negative->first, minimum);
    EXPECT_EQ(negative->second, 0);
    const auto negativeOverflow = Layout(Shape{2, 2}, {0, -1}, minimum).getElementBounds();
    ASSERT_FALSE(negativeOverflow);
    EXPECT_EQ(negativeOverflow.error().error, LayoutError::OffsetAddOverflow);
    EXPECT_EQ(negativeOverflow.error().dimension, 1);
}
}  // namespace
