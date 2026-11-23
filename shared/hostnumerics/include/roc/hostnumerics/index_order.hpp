// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/permutation.hpp>
#include <roc/hostnumerics/shape.hpp>
#include <stdexcept>
#include <utility>
#include <variant>

namespace roc::hostnumerics {
// How an operation numbers the coordinates of a tensor whose rank it learns
// only when the order is used: either the last dimension fastest, which fits
// every rank, or an explicit axis permutation from slowest to fastest, which
// must have exactly the tensor's rank. There is no default order.
class IndexOrder {
   public:
    IndexOrder(LastDimensionFastest) noexcept {}
    IndexOrder(Permutation axes) noexcept : m_order(std::move(axes)) {}

    // Returns the axes, slowest first, for a tensor of `rank` dimensions.
    // Throws std::runtime_error if an explicit permutation has another size.
    [[nodiscard]] Permutation forRank(size_t rank) const {
        if (std::holds_alternative<LastDimensionFastest>(m_order))
            return Permutation::identity(rank);
        const Permutation& axes = std::get<Permutation>(m_order);
        if (axes.size() != rank)
            throw std::runtime_error("Index order rank does not match the tensor rank.");
        return axes;
    }

   private:
    std::variant<LastDimensionFastest, Permutation> m_order;
};
}  // namespace roc::hostnumerics
