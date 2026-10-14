// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ===================================================================
// DynLayout — Runtime type-erased wrapper for mosaic2 layouts
//
// This is a thin dispatch layer, NOT a reimplementation. Every
// algebraic operation (coalesce, blocked_product, right_inverse, etc.)
// is performed by:
//   1. Constructing a mosaic2::Layout<tuple<size_t,...>, tuple<size_t,...>>
//      from the stored shape/stride vectors
//   2. Calling the EXACT mosaic2 template function
//   3. Extracting shape/stride from the result back into vectors
//
// No layout math lives here — it all lives in the mosaic2 headers.
// ===================================================================

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mosaic {

class DynLayout {
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;

   public:
    DynLayout() = default;

    DynLayout(std::vector<std::size_t> shape, std::vector<std::size_t> stride)
        : shape_(std::move(shape)), stride_(std::move(stride)) {
        if (shape_.size() != stride_.size()) {
            throw std::invalid_argument("shape and stride must have same length");
        }
    }

    // ---------------------------------------------------------------
    // Evaluation — dispatches to mosaic2 template code by rank
    // ---------------------------------------------------------------

    // Flat-index evaluation: L(idx) → offset
    std::size_t eval(std::size_t flat_idx) const;

    // Coordinate evaluation: L(row, col, ...) → offset
    std::size_t eval_coord(const std::vector<std::size_t>& coords) const;

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    const std::vector<std::size_t>& shape() const {
        return shape_;
    }
    const std::vector<std::size_t>& stride() const {
        return stride_;
    }

    std::size_t size() const {
        std::size_t s = 1;
        for (auto d : shape_) s *= d;
        return s;
    }

    std::size_t cosize() const {
        if (shape_.empty()) return 0;
        std::size_t cs = 1;
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            cs += (shape_[i] - 1) * stride_[i];
        }
        return cs;
    }

    int rank() const {
        return static_cast<int>(shape_.size());
    }

    // ---------------------------------------------------------------
    // String repr: "(4,8):(8,1)"
    // ---------------------------------------------------------------

    std::string to_string() const;

    // ---------------------------------------------------------------
    // Enumerate all offsets: [L(0), L(1), ..., L(size-1)]
    // ---------------------------------------------------------------

    std::vector<std::size_t> enumerate() const;
};

// ===================================================================
// Factory functions — construct via mosaic2 templates, extract result
// ===================================================================

DynLayout dyn_row_major(const std::vector<std::size_t>& dims);
DynLayout dyn_column_major(const std::vector<std::size_t>& dims);
DynLayout dyn_make_layout(const std::vector<std::size_t>& shape,
                          const std::vector<std::size_t>& stride);

// ===================================================================
// Algebra — dispatch to mosaic2 template functions, return DynLayout
// ===================================================================

DynLayout dyn_coalesce(const DynLayout& layout);
DynLayout dyn_blocked_product(const DynLayout& inner, const DynLayout& outer);
DynLayout dyn_right_inverse(const DynLayout& layout);
DynLayout dyn_complement(const DynLayout& layout, std::size_t cosize_val);
DynLayout dyn_logical_product(const DynLayout& tile, const DynLayout& grid);

}  // namespace mosaic
