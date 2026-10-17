// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compiled without exception handling. Allocation attempts abort, including
// on failure paths, so diagnostics cannot silently acquire heap storage.
#include <cstdlib>
#include <limits>
#include <new>
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
        const auto result = tryConvertToInteger<int8_t>(static_cast<double>(input), reject);
        if (!result || *result != input) return 1;
    }
    const auto outOfRange = tryConvertToInteger<int8_t>(300.0, reject);
    if (outOfRange.has_value() || outOfRange.error().code != ConversionError::RoundedOutOfRange)
        return 2;
    const auto integral = tryConvertToInteger<uint8_t>(-1, reject);
    if (integral.has_value() || integral.error().code != ConversionError::IntegerOutOfRange)
        return 3;
    const auto nan = tryConvertToInteger<int64_t>(std::numeric_limits<double>::quiet_NaN(), reject);
    if (nan.has_value() || nan.error().code != ConversionError::NaN) return 4;
    const auto infinity =
        tryConvertToInteger<int8_t>(std::numeric_limits<double>::infinity(), reject);
    if (infinity.has_value() || infinity.error().code != ConversionError::Infinity) return 5;
    auto saturate = reject;
    saturate.integerOverflow = IntegerOverflow::Saturate;
    const auto saturated = tryConvertToInteger<int8_t>(300.0, saturate);
    if (!saturated || *saturated != 127) return 6;
    auto wrap = reject;
    wrap.integerOverflow = IntegerOverflow::ModuloWrap;
    const auto wrapped = tryConvertToInteger<int8_t>(300.0, wrap);
    if (!wrapped || *wrapped != 44) return 7;
    auto nearest = reject;
    nearest.integerRounding = IntegerRounding::NearestEven;
    const auto rounded = tryConvertToInteger<int8_t>(2.5, nearest);
    if (!rounded || *rounded != 2) return 8;
    const auto nonreal = tryConvertToInteger<int8_t>(std::complex<double>(2, 1), reject);
    if (nonreal.has_value() || nonreal.error().code != ConversionError::Nonreal) return 9;
    nearest.integerRounding = IntegerRounding::Count;
    const auto invalid = tryConvertToInteger<int8_t>(2.0, nearest);
    if (invalid.has_value() || invalid.error().code != ConversionError::InvalidRounding) return 10;
    return 0;
}
