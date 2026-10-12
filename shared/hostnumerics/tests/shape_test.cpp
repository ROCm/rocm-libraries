// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <limits>
#include <roc/hostnumerics/shape.hpp>
#include <string>
#include <string_view>

namespace {
using namespace roc::hostnumerics;

void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Function>
void expectException(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("Expected Shape exception was not thrown.");
}

void checkShape() {
    const Shape scalar;
    check(scalar.rank() == 0 && scalar.dimensions().empty() && scalar.elementCount() == 1,
          "Rank-zero shape must describe one scalar.");
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
    expectException<std::out_of_range>([&] { (void)shape.extent(3); });
    expectException<std::out_of_range>([&] { (void)shape[3]; });
    expectException<std::out_of_range>([&] { (void)shape.elementCount(2, 1); });
    expectException<std::out_of_range>([&] { (void)shape.elementCount(0, 4); });
    expectException<std::out_of_range>([&] { (void)shape.elementCountExcluding(3); });

    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    check(Shape{maximum, 1}.elementCount() == maximum, "Representable count was rejected.");
    expectException<std::overflow_error>([&] { (void)Shape{maximum, 2}.elementCount(); });
    expectException<std::overflow_error>(
        [&] { (void)Shape{maximum, 1, 2}.elementCountExcluding(1); });
    for (const Shape& empty : {Shape{0, maximum, 2}, Shape{maximum, 0, 2}, Shape{maximum, 2, 0}}) {
        check(empty.elementCount() == 0, "Zero extent must take precedence over overflow.");
        expectException<std::out_of_range>(
            [&] { (void)empty.coordinates(0, IndexOrder::LastDimensionFastest); });
    }
    check(Shape{1, maximum, 2, 0}.elementCountExcluding(0) == 0 &&
              Shape{1, maximum, 2, 0}.elementCount(1, 4) == 0 &&
              Shape{2, 0, 4}.elementCountExcluding(1) == 8,
          "Selected dimension products handled zero incorrectly.");
    expectException<std::overflow_error>(
        [&] { (void)Shape{maximum, 2, 0}.elementCountExcluding(2); });
    expectException<std::overflow_error>([&] { (void)Shape{maximum, 2, 0}.elementCount(0, 2); });

    const std::array<size_t, 3> position{1, 0, 2};
    check(shape.linearIndex(position, IndexOrder::FirstDimensionFastest) == 13 &&
              shape.linearIndex(position, IndexOrder::LastDimensionFastest) == 14,
          "Logical index order is incorrect.");
    check(shape.coordinates(17, IndexOrder::FirstDimensionFastest) ==
                  std::vector<size_t>({1, 2, 2}) &&
              shape.coordinates(17, IndexOrder::LastDimensionFastest) ==
                  std::vector<size_t>({1, 1, 1}),
          "Logical coordinate order is incorrect.");
    for (auto order : {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
        for (size_t i = 0; i < 24; ++i) {
            std::array<size_t, 3> coordinates{};
            shape.coordinates(i, order, coordinates);
            check(shape.linearIndex(coordinates, order) == i &&
                      shape.coordinates(i, order) ==
                          std::vector<size_t>(coordinates.begin(), coordinates.end()),
                  "Coordinate/index round trip failed.");
        }
        check(scalar.coordinates(0, order).empty() && scalar.linearIndex({}, order) == 0,
              "Rank-zero coordinate conversion failed.");
        expectException<std::out_of_range>([&] { (void)scalar.coordinates(1, order); });
        expectException<std::out_of_range>([&] { (void)shape.coordinates(24, order); });
        expectException<std::invalid_argument>([&] { (void)shape.linearIndex({}, order); });
        expectException<std::invalid_argument>([&] { shape.coordinates(0, order, {}); });
        expectException<std::out_of_range>([&] {
            const std::array<size_t, 3> invalid{1, 3, 0};
            (void)shape.linearIndex(invalid, order);
        });
        expectException<std::out_of_range>([&] {
            const std::array<size_t, 1> invalid{0};
            (void)Shape{0}.linearIndex(invalid, order);
        });
    }
    for (auto order : {IndexOrder::Count, static_cast<IndexOrder>(-1)}) {
        expectException<std::invalid_argument>([&] { (void)shape.linearIndex(position, order); });
        expectException<std::invalid_argument>([&] { (void)shape.coordinates(0, order); });
        expectException<std::invalid_argument>([&] { (void)scalar.linearIndex({}, order); });
        expectException<std::invalid_argument>([&] { (void)scalar.coordinates(0, order); });
    }
    const Shape oversized{maximum, 2};
    const std::array<size_t, 2> smallPosition{0, 1};
    check(oversized.linearIndex(smallPosition, IndexOrder::LastDimensionFastest) == 1,
          "Representable index in an oversized shape was rejected.");
    expectException<std::overflow_error>([&] {
        const std::array<size_t, 2> position{maximum - 1, 1};
        (void)oversized.linearIndex(position, IndexOrder::LastDimensionFastest);
    });
    expectException<std::overflow_error>([&] {
        const std::array<size_t, 2> position{maximum - 1, 1};
        (void)oversized.linearIndex(position, IndexOrder::FirstDimensionFastest);
    });
    expectException<std::overflow_error>(
        [&] { (void)oversized.coordinates(0, IndexOrder::LastDimensionFastest); });
}
}  // namespace

int main() {
    checkShape();
}
