// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/hostnumerics/expected.hpp>
#include <span>
#include <string>

namespace roc::hostnumerics::detail {
enum class LayoutError {
    ExtentOutOfRange,
    ContiguousStrideOverflow,
    OffsetMultiplyOverflow,
    OffsetAddOverflow,
    OrderRankMismatch,
    InvalidOrder,
    CoordinateRankMismatch,
    CoordinateOutOfRange,
};

// For CoordinateRankMismatch, value and limit are the coordinate and shape ranks.
// For CoordinateOutOfRange, value is the coordinate on axis `dimension` and limit
// is that axis's extent. Other errors leave these fields zero.
struct LayoutFailure {
    LayoutError error;
    size_t dimension = 0;
    size_t value = 0;
    size_t limit = 0;

    std::string message() const;
};

class LayoutOffset {
   public:
    // Returns value * factor if it fits ptrdiff_t. Plain multiplication would
    // convert factor to size_t and wrap silently, and a signed product can
    // overflow. Instead, compare unsigned magnitudes against the limit before
    // multiplying. The magnitude of a negative factor is computed as
    // -(factor + 1) + 1 because -PTRDIFF_MIN does not fit ptrdiff_t.
    static std::optional<ptrdiff_t> getOffsetMultiply(size_t value, ptrdiff_t factor) noexcept {
        if (value == 0 || factor == 0) return 0;
        const bool negative = factor < 0;
        const uintmax_t magnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        // A negative product can reach PTRDIFF_MIN, one past -PTRDIFF_MAX.
        const uintmax_t limit = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) +
                                static_cast<uintmax_t>(negative);
        if (static_cast<uintmax_t>(value) > limit / magnitude) return std::nullopt;
        const uintmax_t product = static_cast<uintmax_t>(value) * magnitude;
        if (!negative) return static_cast<ptrdiff_t>(product);
        if (product == limit) return std::numeric_limits<ptrdiff_t>::min();
        return -static_cast<ptrdiff_t>(product);
    }

    static std::optional<ptrdiff_t> getOffsetAdd(ptrdiff_t left, ptrdiff_t right) noexcept {
        if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
            return std::nullopt;
        return left + right;
    }
    // Layout construction establishes matching extent and stride ranks. Coordinate
    // validation and arithmetic return failures without owning metadata or messages.
    static Expected<ptrdiff_t, LayoutFailure> getElementOffset(
        std::span<const size_t> dimensions, std::span<const ptrdiff_t> strides, ptrdiff_t offset,
        std::span<const size_t> indices) noexcept {
        assert(dimensions.size() == strides.size() && "Extent and stride ranks must match.");
        if (indices.size() != dimensions.size())
            return Unexpected(LayoutFailure{LayoutError::CoordinateRankMismatch, 0, indices.size(),
                                            dimensions.size()});

        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= dimensions[dimension])
                return Unexpected(LayoutFailure{LayoutError::CoordinateOutOfRange, dimension,
                                                indices[dimension], dimensions[dimension]});
        }
        ptrdiff_t result = offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            const auto delta = getOffsetMultiply(indices[dimension], strides[dimension]);
            if (!delta) return Unexpected(LayoutFailure{LayoutError::OffsetMultiplyOverflow});
            const auto sum = getOffsetAdd(result, *delta);
            if (!sum) return Unexpected(LayoutFailure{LayoutError::OffsetAddOverflow});
            result = *sum;
        }
        return result;
    }

   private:
    LayoutOffset() = delete;
};
}  // namespace roc::hostnumerics::detail
