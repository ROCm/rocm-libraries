// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/mx_packing.hpp"

#include <roc/host_numerics/scalar_storage.hpp>

#include "detail/checked_arithmetic.hpp"

namespace roc::host_numerics::detail {
std::vector<std::byte> packRawValues(std::span<const uint8_t> rawValues, uint16_t bitsPerValue,
                                     int threadCount) {
    (void)threadCount;
    const size_t totalBits = checkedMultiply(rawValues.size(), static_cast<size_t>(bitsPerValue),
                                             "MX packed storage size overflow.");
    std::vector<std::byte> storage(ceilDivide(totalBits, size_t{8}), std::byte{0});
    if (bitsPerValue == 8) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t index = 0; index < rawValues.size(); ++index)
            storage[index] = static_cast<std::byte>(rawValues[index]);
    } else if (bitsPerValue == 4) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t byteIndex = 0; byteIndex < storage.size(); ++byteIndex) {
            const size_t firstIndex = byteIndex * 2;
            const uint8_t first = rawValues[firstIndex] & 0x0fU;
            const uint8_t second =
                firstIndex + 1 < rawValues.size() ? rawValues[firstIndex + 1] & 0x0fU : 0;
            storage[byteIndex] = static_cast<std::byte>(first | (second << 4));
        }
    } else if (bitsPerValue == 6) {
        const size_t groups = (rawValues.size() + 3U) / 4U;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threadCount)
#endif
        for (size_t group = 0; group < groups; ++group) {
            const size_t firstIndex = group * 4U;
            uint32_t word = 0;
            for (size_t offset = 0; offset < 4U && firstIndex + offset < rawValues.size(); ++offset)
                word |= static_cast<uint32_t>(rawValues[firstIndex + offset] & 0x3fU)
                        << (offset * 6U);
            const size_t byteOffset = group * 3U;
            for (size_t byte = 0; byte < 3U && byteOffset + byte < storage.size(); ++byte)
                storage[byteOffset + byte] = static_cast<std::byte>((word >> (byte * 8U)) & 0xffU);
        }
    } else {
        for (size_t index = 0; index < rawValues.size(); ++index)
            writePackedBits(storage, index * static_cast<size_t>(bitsPerValue), bitsPerValue,
                            rawValues[index]);
    }
    return storage;
}
}  // namespace roc::host_numerics::detail
