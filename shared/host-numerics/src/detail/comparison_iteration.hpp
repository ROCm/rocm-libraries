// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "comparison_common.hpp"

namespace roc::host_numerics::detail {
template <typename Function>
void forEachSelectedIndex(const Shape& shape, const OutputSelection& selection,
                          Function&& function) {
    const size_t total = shape.elementCount();
    std::vector<size_t> coordinates(shape.rank(), 0);
    if (selection.kind() == OutputSelectionKind::Explicit) {
        for (const size_t logicalIndex : selection.indices(total)) {
            coordinatesForLinearIndex(logicalIndex, shape, selection.indexOrder(), coordinates);
            function(logicalIndex, std::span<const size_t>(coordinates));
        }
        return;
    }
    if (selection.selectsAll()) {
        for (size_t logicalIndex = 0; logicalIndex < total; ++logicalIndex) {
            coordinatesForLinearIndex(logicalIndex, shape, selection.indexOrder(), coordinates);
            function(logicalIndex, std::span<const size_t>(coordinates));
        }
        return;
    }
    size_t selected = 0;
    for (size_t logicalIndex = selection.first();
         logicalIndex < total && selected < selection.maxElements(); ++selected) {
        coordinatesForLinearIndex(logicalIndex, shape, selection.indexOrder(), coordinates);
        function(logicalIndex, std::span<const size_t>(coordinates));
        if (selection.stride() > std::numeric_limits<size_t>::max() - logicalIndex) break;
        logicalIndex += selection.stride();
    }
}

ComparisonValue loadComparisonValue(const Tensor& view, ptrdiff_t logicalOffset);

template <typename Tag>
ComparisonValue loadComparisonValueKnown(std::span<const std::byte> storage,
                                         ptrdiff_t logicalOffset) {
    if (logicalOffset < 0) throw std::out_of_range("Comparison logical offset is negative.");

    constexpr ScalarType type = Tag::type;
    constexpr size_t storageBits = scalarTypeInfo(type).storageBits;
    const size_t byteOffset = static_cast<size_t>(logicalOffset) * storageBits / 8;
    const auto readNativeUnchecked = [&]<typename T>() {
        T value;
        std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
        return value;
    };

    if constexpr (type == ScalarType::Boolean)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint8_t>() != 0),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt8)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint8_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int8)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int8_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt16)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint16_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int16)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int16_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint32_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int32_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt64)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint64_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int64)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int64_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float16)
        return {
            static_cast<double>(decodeFloat16(readNativeUnchecked.template operator()<uint16_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::BFloat16)
        return {
            static_cast<double>(
                decodeBFloat16(readNativeUnchecked.template operator()<uint16_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<float>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float64)
        return { readNativeUnchecked.template operator()<double>(), 0.0, false };
    else if constexpr (type == ScalarType::ComplexFloat32) {
        const auto value = readNativeUnchecked.template operator()<std::complex<float>>();
        return {static_cast<double>(value.real()), static_cast<double>(value.imag()), true};
    } else if constexpr (type == ScalarType::ComplexFloat64) {
        const auto value = readNativeUnchecked.template operator()<std::complex<double>>();
        return {value.real(), value.imag(), true};
    } else if constexpr (type == ScalarType::Float8E4M3 || type == ScalarType::Float8E5M2 ||
                         type == ScalarType::Float8E4M3Fnuz || type == ScalarType::Float8E5M2Fnuz ||
                         type == ScalarType::E5M3 || type == ScalarType::E4M3)
        return {
            static_cast<double>(
                decodeBinaryFloat(type, readNativeUnchecked.template operator()<uint8_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::E8M0)
        return {
            static_cast<double>(decodeE8M0(readNativeUnchecked.template operator()<uint8_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::E8M0Zero)
        return {
            static_cast<double>(decodeE8M0Zero(readNativeUnchecked.template operator()<uint8_t>())),
            0.0,
            false
        };
    else
        return typedComparisonValue(decodeScalarKnown<Tag, double>(storage, logicalOffset));
}

template <typename Tag>
auto loadFastComparisonReal(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    constexpr ScalarType type = Tag::type;
    constexpr ScalarCategory category = scalarTypeInfo(type).category;

    const auto readNativeUnchecked = [&]<typename T>() {
        T value;
        std::memcpy(&value, storage.data() + static_cast<size_t>(logicalOffset) * sizeof(T),
                    sizeof(T));
        return value;
    };

    if constexpr (type == ScalarType::Float32) {
        return readNativeUnchecked.template operator()<float>();
    } else if constexpr (type == ScalarType::Float64) {
        return readNativeUnchecked.template operator()<double>();
    } else if constexpr (category == ScalarCategory::Boolean) {
        return readNativeUnchecked.template operator()<uint8_t>() != 0;
    } else if constexpr (category == ScalarCategory::SignedInteger) {
        if constexpr (std::is_void_v<typename Tag::Storage>)
            return decodeScalarKnown<Tag, int64_t>(storage, logicalOffset);
        else
            return readNativeUnchecked.template operator()<typename Tag::Storage>();
    } else if constexpr (category == ScalarCategory::UnsignedInteger) {
        return readNativeUnchecked.template operator()<typename Tag::Storage>();
    } else {
        return loadComparisonValueKnown<Tag>(storage, logicalOffset).real;
    }
}

template <typename Tag>
bool knownAllCloseDecision(const Tensor& observed, ptrdiff_t observedOffset, const Tensor& expected,
                           ptrdiff_t expectedOffset, const ComparisonOptions& options) {
    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
        const ComparisonValue observedValue =
            loadComparisonValueKnown<Tag>(observed.rawEncodedBackingStorage(), observedOffset);
        const ComparisonValue expectedValue =
            loadComparisonValueKnown<Tag>(expected.rawEncodedBackingStorage(), expectedOffset);
        if (options.complexComparisonMode == ComplexComparisonMode::Magnitude)
            return compareComplexMagnitude(observedValue, expectedValue, options).close;
        return valuesClose(observedValue.real, expectedValue.real, options) &&
               valuesClose(observedValue.imaginary, expectedValue.imaginary, options);
    } else {
        return valuesClose(
            loadFastComparisonReal<Tag>(observed.rawEncodedBackingStorage(), observedOffset),
            loadFastComparisonReal<Tag>(expected.rawEncodedBackingStorage(), expectedOffset),
            options);
    }
}

bool isUnwrittenSentinelValue(ScalarType type, const ComparisonValue& value);
void validateSentinelRange(ScalarType type, std::span<const std::byte> storage, size_t firstElement,
                           size_t elementCount);

}  // namespace roc::host_numerics::detail
