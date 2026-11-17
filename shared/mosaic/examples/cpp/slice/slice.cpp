// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Demonstrates CuTe-style underscore slicing.
//
//   _   is a placeholder marking "keep this dimension"
//   L(_, k)  → returns a sub-layout for the column at coord k along mode 1
//   L(i, _)  → row sub-layout
//
// The slice itself is a Layout, so it has size(), cosize(), and can be
// evaluated like any other layout.

#include <iostream>
#include <mosaic/mosaic.hpp>

using namespace mosaic;

template <class L>
void enumerate(const char* label, L const& layout) {
    std::cout << "  " << label << " = " << layout << "  (size=" << size(layout) << "):  ";
    for (int i = 0; i < size(layout); ++i) std::cout << layout(i) << " ";
    std::cout << "\n";
}

int main() {
    std::cout << "Mosaic slice demo\n";
    std::cout << "=================\n\n";

    // -------------------------------------------------------------------
    // Example 1: Column slice of a flat 2D layout
    //
    // Slicing returns a sub-LAYOUT (the iteration pattern of the kept
    // dimensions). The absolute base offset for the bound dimensions is
    // tracked separately — pair the slice with crd2idx() to get full
    // offsets if you need them.
    // -------------------------------------------------------------------
    std::cout << "Example 1: 4x8 column-major layout\n";
    auto L = make_layout(make_shape(4, 8));  // default = compact column-major
    std::cout << "  L = " << L << "\n";

    std::cout << "\n  L(_, 3) — sub-layout for column 3 (rank-1, size 4):\n";
    auto col3 = L(_, 3);
    enumerate("col3", col3);
    auto col3_base = crd2idx(make_coord(0, 3), L.shape(), L.stride());
    std::cout << "  base offset of column 3 in L: " << col3_base << "\n";
    std::cout << "  full column-3 offsets in L:   ";
    for (int i = 0; i < size(col3); ++i) std::cout << (col3_base + col3(i)) << " ";
    std::cout << "\n";

    std::cout << "\n  L(2, _) — sub-layout for row 2 (rank-1, size 8):\n";
    auto row2 = L(2, _);
    enumerate("row2", row2);
    auto row2_base = crd2idx(make_coord(2, 0), L.shape(), L.stride());
    std::cout << "  base offset of row 2 in L: " << row2_base << "\n";
    std::cout << "  full row-2 offsets in L:   ";
    for (int i = 0; i < size(row2); ++i) std::cout << (row2_base + row2(i)) << " ";
    std::cout << "\n";

    // -------------------------------------------------------------------
    // Example 2: Slicing via the free-function form
    // -------------------------------------------------------------------
    std::cout << "\n\nExample 2: free-function slice() — same result\n";
    auto col3_alt = slice(make_coord(_, 3), L);
    std::cout << "  slice(make_coord(_, 3), L) = " << col3_alt << "\n";
    enumerate("col3_alt", col3_alt);

    // -------------------------------------------------------------------
    // Example 3: Hierarchical layout — slice an inner mode
    // -------------------------------------------------------------------
    std::cout << "\n\nExample 3: hierarchical layout, slicing an inner mode\n";
    auto inner = make_layout(make_shape(2, 2), compact_row_major(make_shape(2, 2)));
    auto outer = make_layout(make_shape(3, 3), compact_col_major(make_shape(3, 3)));
    auto H = make_layout(inner, outer);
    std::cout << "  H (rank-2, both modes are 2D) = " << H << "\n";

    // Coord shape mirrors layout's nested shape: ((c0_inner, c1_inner), (c0_outer, c1_outer))
    auto inner_keep = H(make_coord(_, make_coord(1, 1)));
    std::cout << "\n  H(_, (1,1)) — fix outer mode at (1,1), keep inner mode free:\n";
    std::cout << "  result = " << inner_keep << "\n";
    enumerate("inner_keep", inner_keep);

    // -------------------------------------------------------------------
    // Example 4: dice — opposite of slice
    // -------------------------------------------------------------------
    std::cout << "\n\nExample 4: dice() — keep the *bound* dimensions, drop free ones\n";
    auto fixed = dice(make_coord(_, 3), L.shape());
    std::cout << "  dice(make_coord(_, 3), L.shape()) = ";
    std::apply([](auto const&... a) { ((std::cout << "(" << a << ") "), ...); }, fixed);
    std::cout << "  (the parts of L's shape that were FIXED by the coord)\n";

    return 0;
}
