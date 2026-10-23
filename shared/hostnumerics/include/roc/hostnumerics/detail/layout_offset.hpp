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

struct LayoutFailure {
    LayoutError error;
    size_t dimension = 0;
    size_t value = 0;
    size_t limit = 0;

    // Formatting a diagnostic can allocate, so this operation can throw.
    std::string message() const;
};

class LayoutOffset {
   public:
    static std::optional<ptrdiff_t> getOffsetMultiply(size_t value, ptrdiff_t factor) noexcept {
        if (value == 0 || factor == 0) return 0;
        const bool negative = factor < 0;
        // -(factor + 1) is representable even for PTRDIFF_MIN. Add the missing one
        // after converting to unsigned, where the full magnitude is representable.
        const uintmax_t magnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        // A negative result can have magnitude PTRDIFF_MAX + 1 (PTRDIFF_MIN).
        const uintmax_t limit = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) +
                                static_cast<uintmax_t>(negative);
        // Test against the bound before multiplying, so the product cannot wrap.
        if (static_cast<uintmax_t>(value) > limit / magnitude) return std::nullopt;
        const uintmax_t product = static_cast<uintmax_t>(value) * magnitude;
        if (!negative) return static_cast<ptrdiff_t>(product);
        // This magnitude has no positive ptrdiff_t representation to negate.
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
