// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <roc/hostnumerics/shape.hpp>
#include <string>
#include <string_view>
#include <type_traits>

namespace {
using namespace roc::hostnumerics;

void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Function>
void expectException(Function function, std::string_view expectedMessage = {}) {
    try {
        function();
    } catch (const Exception& exception) {
        check(std::string_view(exception.what()).find(expectedMessage) != std::string_view::npos,
              "Shape exception is missing the expected context.");
        return;
    }
    throw std::runtime_error("Expected Shape exception was not thrown.");
}

void checkShape() {
    static_assert(!std::is_constructible_v<Shape, size_t>);
    static_assert(!std::is_convertible_v<std::vector<size_t>, Shape>);
    static_assert(!std::is_convertible_v<std::span<const size_t>, Shape>);
    const Shape scalar;
    check(scalar.rank() == 0 && scalar.dimensions().empty() && scalar.elementCount() == 1,
          "Rank-zero shape must describe one scalar.");
    check(Shape{} == scalar && Shape({}) == scalar && Shape{7}.rank() == 1,
          "Brace construction must describe extents, not a dimension count.");
    std::vector<size_t> source{2, 3, 4};
    const Shape copied(source);
    const Shape spanCopy{std::span<const size_t>(source)};
    source[0] = 9;
    check(copied == Shape{2, 3, 4} && spanCopy == copied,
          "Shape must own vector and span inputs independently.");
    check(Shape(std::move(source)) == Shape{9, 3, 4}, "Rvalue vector construction failed.");
    const std::array<size_t, 3> extents{2, 3, 4};
    const Shape shape{2, 3, 4};
    check(shape == Shape(std::vector<size_t>{2, 3, 4}) &&
              shape == Shape(std::span<const size_t>(extents)),
          "Shape constructors disagree.");
    check(shape != Shape{2, 4, 3}, "Shape equality must preserve dimension order.");
    check(shape.rank() == 3 && shape[1] == 3 && shape.extent(2) == 4 &&
              std::ranges::equal(shape.dimensions(), extents),
          "Shape dimensions disagree with supplied extents.");
    check(shape.elementCount() == 24 && shape.elementCount(1, 3) == 12 &&
              shape.elementCount(3, 3) == 1 && shape.elementCountExcluding(1) == 8,
          "Shape products are incorrect.");
    check(Shape{7}.elementCountExcluding(0) == 1,
          "Excluding the only dimension must produce an empty product.");
    expectException<std::out_of_range>([&] { (void)shape.extent(3); },
                                       "dimension 3 is outside [0, 3)");
    expectException<std::out_of_range>([&] { (void)shape[3]; });
    expectException<std::out_of_range>(
        [&] { (void)shape.elementCount(2, 1); },
        "range [2, 1) is invalid for rank 3: the start exceeds the end");
    expectException<std::out_of_range>(
        [&] { (void)shape.elementCount(0, 4); },
        "range [0, 4) is invalid for rank 3: the end exceeds the shape rank");
    expectException<std::out_of_range>([&] { (void)shape.elementCountExcluding(3); },
                                       "dimension 3 is outside [0, 3)");

    const std::array<size_t, 3> position{1, 0, 2};
    for (auto order : {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
        std::array<size_t, 3> coordinates{};
        shape.coordinates(17, order, coordinates);
        check(shape.linearIndex(coordinates, order) == 17 &&
                  shape.coordinates(17, order) ==
                      std::vector<size_t>(coordinates.begin(), coordinates.end()),
              "Coordinate span output disagrees with the owning result.");
        expectException<std::invalid_argument>(
            [&] { shape.coordinates(0, order, {}); },
            "coordinate output rank 0 does not match shape rank 3");
    }
    // Invalid enum values and the span-output overload cannot be exercised via Python.
    for (auto order : {IndexOrder::Count, static_cast<IndexOrder>(-1)}) {
        expectException<std::invalid_argument>([&] { (void)shape.linearIndex(position, order); });
        expectException<std::invalid_argument>([&] { (void)shape.coordinates(0, order); });
        expectException<std::invalid_argument>([&] { (void)scalar.linearIndex({}, order); });
        expectException<std::invalid_argument>([&] { (void)scalar.coordinates(0, order); });
    }
}
}  // namespace

int main() {
    checkShape();
}
