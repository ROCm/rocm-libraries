// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <roc/hostnumerics/periodic_initialization.hpp>

namespace {
using namespace roc::hostnumerics;

TEST(PeriodicInitialization, MultidimensionalChunksAndSnapshot) {
    Tensor seed(ScalarType::Int32, Layout(Shape{2, 3, 2}, {19, 5, -1}, 1));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 2; ++k) seed.storeFrom({i, j, k}, int(100 * i + 10 * j + k));
    PeriodicInitialization init(seed, Shape{5, 7, 3});
    seed.storeFrom({1, 2, 1}, -123);
    Tensor exposedCopy = init.copySeedTile();
    exposedCopy.storeFrom({1, 2, 1}, -456);
    PeriodicInitialization copied = init;
    EXPECT_EQ(copied.seedStorageBytes(), 12 * sizeof(int32_t));

    Tensor whole(ScalarType::Int32, Shape{5, 7, 3});
    init.fillInto(whole);
    // Deliberately visit chunks backwards, with boundaries through the periods.
    for (size_t i : {size_t(3), size_t(0)}) {
        const size_t rows = i == 3 ? 2 : 3;
        Tensor chunk(ScalarType::Int32, Layout(Shape{rows, 7, 3}, {1, 4, 31}, 2));
        std::fill(chunk.rawEncodedBackingStorage().begin(), chunk.rawEncodedBackingStorage().end(),
                  std::byte{0xa5});
        copied.fillInto(chunk, std::array<size_t, 3>{i, 0, 0});
        EXPECT_EQ(chunk.rawEncodedBackingStorage()[0], std::byte{0xa5});
        for (size_t x = 0; x < rows; ++x)
            for (size_t y = 0; y < 7; ++y)
                for (size_t z = 0; z < 3; ++z) {
                    const int expected = int(100 * ((i + x) % 2) + 10 * (y % 3) + z % 2);
                    EXPECT_EQ(chunk.loadAs<int>({x, y, z}), expected);
                    EXPECT_EQ(whole.loadAs<int>({i + x, y, z}), expected);
                }
    }
}

TEST(PeriodicInitialization, PreservesEveryScalarEncodingAndPadding) {
    for (int t = 0; t < static_cast<int>(ScalarType::Count); ++t) {
        const auto type = static_cast<ScalarType>(t);
        const auto bits = scalarTypeInfo(type).storageBits;
        SCOPED_TRACE(t);
        Tensor seed(type, Shape{2});
        auto storage = seed.rawEncodedBackingStorage();
        for (size_t i = 0; i < storage.size(); ++i) storage[i] = std::byte((i * 73 + 0xab) & 0xff);
        const std::vector<std::byte> original(storage.begin(), storage.end());
        PeriodicInitialization init(seed, Shape{7});
        // Odd element offset and stride exercise packed values sharing bytes.
        Tensor output(type, Layout(Shape{5}, {2}, 1));
        std::fill(output.rawEncodedBackingStorage().begin(),
                  output.rawEncodedBackingStorage().end(), std::byte{0xa5});
        init.fillInto(output, std::array<size_t, 1>{1});
        const auto actual = output.rawEncodedBackingStorage();
        for (size_t b = 0; b < actual.size() * 8; ++b) {
            const size_t element = b / bits;
            const bool written = element >= 1 && element <= 9 && element % 2 == 1;
            const size_t sourceBit = (((element - 1) / 2 + 1) % 2) * bits + b % bits;
            const unsigned expected =
                written
                    ? (std::to_integer<unsigned>(original[sourceBit / 8]) >> (sourceBit % 8)) & 1
                    : (0xa5U >> (b % 8)) & 1;
            EXPECT_EQ((std::to_integer<unsigned>(actual[b / 8]) >> (b % 8)) & 1, expected);
        }
    }
}

TEST(PeriodicInitialization, ScalarEmptyAndOversizedPeriods) {
    const PeriodicInitialization scalar(Tensor(3.25), Shape{});
    Tensor scalarResult(ScalarType::Float64, Shape{});
    scalar.fillInto(scalarResult);
    EXPECT_EQ(scalarResult.item<double>(), 3.25);

    const PeriodicInitialization empty(Tensor(ScalarType::Int4, Shape{0, 2}), Shape{0, 9});
    Tensor emptyResult(ScalarType::Int4, Shape{0, 3});
    EXPECT_NO_THROW(empty.fillInto(emptyResult, std::array<size_t, 2>{0, 6}));

    Tensor seed(ScalarType::Int32, Shape{5});
    seed.storeFrom({1}, 17);
    PeriodicInitialization largePeriod(seed, Shape{2});
    Tensor result(ScalarType::Int32, Shape{1});
    largePeriod.fillInto(result, std::array<size_t, 1>{1});
    EXPECT_EQ(result.loadAs<int>({0}), 17);
}

TEST(PeriodicInitialization, RejectsInvalidRegionsBeforeWriting) {
    EXPECT_THROW((PeriodicInitialization(Tensor(1), Shape{1})), std::runtime_error);
    EXPECT_THROW((PeriodicInitialization(Tensor(ScalarType::Int32, Shape{0}), Shape{2})),
                 std::runtime_error);
    Tensor seed(ScalarType::Int32, Shape{2, 2});
    PeriodicInitialization init(seed, Shape{7, 9});
    Tensor output(ScalarType::Int32, Shape{3, 3});
    EXPECT_THROW(init.fillInto(output, std::array<size_t, 2>{5, 0}), std::runtime_error);
    EXPECT_THROW(init.fillInto(output, std::array<size_t, 2>{0, SIZE_MAX}), std::runtime_error);
    EXPECT_THROW(init.fillInto(output, std::array<size_t, 1>{0}), std::runtime_error);
    EXPECT_THROW(init.fillInto(Tensor(ScalarType::Float32, Shape{3, 3})), std::runtime_error);
    EXPECT_THROW(init.fillInto(Tensor(ScalarType::Int32, Layout(Shape{3, 3}, {1, 1}, 0))),
                 std::runtime_error);
}
}  // namespace
