// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Demonstrates right_inverse: given a layout L, find R such that L(R(i)) = i
// for all i in [0, cosize(L)).
//
// Useful when you have a layout that maps logical → physical positions and
// you need to invert: given a physical offset, recover the logical coord.
//
// NOTE: mosaic's current right_inverse implementation is correct for compact
// rank-1 / rank-2 column-major-like layouts. Hierarchical inverses involve
// complement + composition machinery that mosaic doesn't fully implement
// yet — this example sticks to cases the impl handles.

#include <iostream>
#include <mosaic/algebra.hpp>  // right_inverse
#include <mosaic/layout.hpp>

using namespace mosaic;

int main() {
    std::cout << "Mosaic right_inverse Demo\n";
    std::cout << "==========================\n\n";

    // Example 1: Compact column-major 4x4. Self-inverse.
    std::cout << "Example 1: Column-major 4x4 layout\n";
    std::cout << "-----------------------------------\n";
    auto L = make_layout(make_shape(4, 4));  // default = compact column-major
    auto L_inv = right_inverse(L);

    std::cout << "L:     " << L << "\n";
    std::cout << "L_inv: " << L_inv << "\n";

    std::cout << "\nVerifying L_inv(L(i)) = i:\n";
    bool ok = true;
    for (int i = 0; i < size(L); ++i) {
        auto y = L(i);
        auto x = L_inv(y);
        ok &= (x == i);
        std::cout << "  i=" << i << ":  L(" << i << ")=" << y << ",  L_inv(" << y << ")=" << x
                  << (x == i ? "  ok" : "  MISMATCH") << "\n";
    }

    // Example 2: Free-function accessors and round-trip count.
    std::cout << "\n\nExample 2: The inverse is itself a Layout\n";
    std::cout << "------------------------------------------\n";
    std::cout << "size(L_inv)    = " << size(L_inv) << "\n";
    std::cout << "cosize(L_inv)  = " << cosize(L_inv) << "\n";
    std::cout << "L_inv(3)       = " << L_inv(3) << "\n";
    std::cout << "L_inv(7)       = " << L_inv(7) << "\n";

    std::cout << "\nIterate L_inv via explicit index loop:\n  ";
    for (int i = 0; i < size(L_inv); ++i) {
        std::cout << L_inv(i) << " ";
    }
    std::cout << "\n";

    std::cout << "\nAll round-trips passed: " << (ok ? "YES" : "NO") << "\n";
    return ok ? 0 : 1;
}
