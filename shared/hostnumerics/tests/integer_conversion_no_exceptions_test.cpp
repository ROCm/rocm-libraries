// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compiled without exception handling. Allocation attempts abort throughout
// native conversion, including rounding, wrapping and saturation.
#include <cstdlib>
#include <limits>
#include <new>
#include <roc/hostnumerics/scalar_conversion.hpp>

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
    const ScalarConversionOptions reject;
    const ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
    const ScalarConversionOptions wrap{IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
    const ScalarConversionOptions nearest{IntegerRounding::NearestEven, IntegerOverflow::Reject};
    static_assert(noexcept(convertScalar<int8_t>(128)));
    static_assert(noexcept(convertScalar<int8_t>(2.5, reject)));
    for (int input = -128; input <= 127; ++input)
        if (convertScalar<int8_t>(static_cast<double>(input), reject) != input) return 1;
    if (convertScalar<int8_t>(128) != -128 || convertScalar<int8_t>(300.0, saturate) != 127 ||
        convertScalar<int8_t>(300.0, wrap) != 44 || convertScalar<int8_t>(2.5, nearest) != 2 ||
        convertScalar<int8_t>(3.5, nearest) != 4)
        return 2;
    if (convertScalar<uint64_t>(std::numeric_limits<uint64_t>::max(), reject) !=
            std::numeric_limits<uint64_t>::max() ||
        convertScalar<int64_t>(std::numeric_limits<uint64_t>::max(), wrap) != -1 ||
        convertScalar<int64_t>(std::numeric_limits<int64_t>::min(), reject) !=
            std::numeric_limits<int64_t>::min())
        return 3;
    if (convertScalar<int8_t>(std::numeric_limits<double>::infinity(), saturate) != 127 ||
        convertScalar<uint8_t>(-std::numeric_limits<double>::infinity(), saturate) != 0 ||
        convertScalar<uint8_t>(-1.75) != 255 || convertScalar<int8_t>(-128.75, reject) != -128 ||
        convertScalar<uint8_t>(-0.75, reject) != 0)
        return 4;
    if (convertScalar<std::complex<double>>(1.5) != std::complex<double>(1.5, 0.0) ||
        convertScalar<int32_t>(std::complex<double>(3.5, 0.0), nearest) != 4 ||
        convertScalar<double>(std::complex<double>(2.25, 0.0)) != 2.25)
        return 5;
    if (!convertScalar<bool>(std::numeric_limits<double>::quiet_NaN()) ||
        convertScalar<bool>(-0.0) || !convertScalar<bool>(-0.25))
        return 6;
    return 0;
}
