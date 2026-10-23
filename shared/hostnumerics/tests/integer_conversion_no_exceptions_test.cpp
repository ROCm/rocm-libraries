// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compiled without exception handling. Allocation attempts abort, including
// on failure paths, so diagnostics cannot silently acquire heap storage.
#include <array>
#include <cstdlib>
#include <limits>
#include <new>
#include <roc/hostnumerics/detail/layout_offset.hpp>
#include <roc/hostnumerics/scalar_integer_conversion.hpp>

void* operator new(std::size_t) {
    std::abort();
}
void* operator new[](std::size_t) {
    std::abort();
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    using namespace roc::hostnumerics;
    using namespace roc::hostnumerics::detail;
    const ScalarConversionOptions reject;
    for (int input = -128; input <= 127; ++input) {
        const auto result = getConvertToInteger<int8_t>(static_cast<double>(input), reject);
        if (!result || *result != input) return 1;
    }
    const auto outOfRange = getConvertToInteger<int8_t>(300.0, reject);
    if (outOfRange.has_value() || outOfRange.error().code != ConversionError::RoundedOutOfRange)
        return 2;
    const auto integral = getConvertToInteger<uint8_t>(-1, reject);
    if (integral.has_value() || integral.error().code != ConversionError::IntegerOutOfRange)
        return 3;
    const auto nan = getConvertToInteger<int64_t>(std::numeric_limits<double>::quiet_NaN(), reject);
    if (nan.has_value() || nan.error().code != ConversionError::NaN) return 4;
    const auto infinity =
        getConvertToInteger<int8_t>(std::numeric_limits<double>::infinity(), reject);
    if (infinity.has_value() || infinity.error().code != ConversionError::Infinity) return 5;
    auto saturate = reject;
    saturate.integerOverflow = IntegerOverflow::Saturate;
    const auto saturated = getConvertToInteger<int8_t>(300.0, saturate);
    if (!saturated || *saturated != 127) return 6;
    auto wrap = reject;
    wrap.integerOverflow = IntegerOverflow::ModuloWrap;
    const auto wrapped = getConvertToInteger<int8_t>(300.0, wrap);
    if (!wrapped || *wrapped != 44) return 7;
    auto nearest = reject;
    nearest.integerRounding = IntegerRounding::NearestEven;
    const auto rounded = getConvertToInteger<int8_t>(2.5, nearest);
    if (!rounded || *rounded != 2) return 8;
    const auto nonreal = getConvertToInteger<int8_t>(std::complex<double>(2, 1), reject);
    if (nonreal.has_value() || nonreal.error().code != ConversionError::Nonreal) return 9;
    nearest.integerRounding = IntegerRounding::Count;
    const auto invalid = getConvertToInteger<int8_t>(2.0, nearest);
    if (invalid.has_value() || invalid.error().code != ConversionError::InvalidRounding) return 10;

    const std::array<size_t, 2> dimensions{2, 3}, coordinates{1, 2};
    const std::array<ptrdiff_t, 2> strides{-3, 1};
    const auto offset = getElementOffset(dimensions, strides, 3, coordinates);
    if (!offset || *offset != 2) return 11;
    const auto rank = getElementOffset(dimensions, strides, 3, {});
    if (rank || rank.error().error != LayoutError::CoordinateRankMismatch ||
        rank.error().value != 0 || rank.error().limit != 2)
        return 12;
    const std::array<size_t, 2> outside{0, 3};
    const auto range = getElementOffset(dimensions, strides, 3, outside);
    if (range || range.error().error != LayoutError::CoordinateOutOfRange ||
        range.error().dimension != 1 || range.error().value != 3 || range.error().limit != 3)
        return 13;
    const std::array<size_t, 1> extent{3}, index{2};
    const std::array<ptrdiff_t, 1> largeStride{std::numeric_limits<ptrdiff_t>::max()};
    const auto multiply = getElementOffset(extent, largeStride, 0, index);
    if (multiply || multiply.error().error != LayoutError::OffsetMultiplyOverflow) return 14;
    const std::array<ptrdiff_t, 1> unitStride{1};
    const auto addition =
        getElementOffset(extent, unitStride, std::numeric_limits<ptrdiff_t>::max(), index);
    if (addition || addition.error().error != LayoutError::OffsetAddOverflow) return 15;
    const std::array<size_t, 1> hugeExtent{SIZE_MAX}, hugeIndex{SIZE_MAX - 1};
    const std::array<ptrdiff_t, 1> zeroStride{0};
    const auto repeated = getElementOffset(hugeExtent, zeroStride, 0, hugeIndex);
    if (!repeated || *repeated != 0) return 16;
    return 0;
}
