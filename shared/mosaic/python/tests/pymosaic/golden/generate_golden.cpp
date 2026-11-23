// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Golden-vector generator for the mosaic Python layout bindings.     ║
// ║                                                                    ║
// ║ This program computes reference outputs DIRECTLY from the mosaic   ║
// ║ C++ headers — the ground truth — for a curated set of layouts      ║
// ║ spanning every composition type the dynamic JIT route supports     ║
// ║ (flat rank-1/2/3, row/col-major, padded/non-compact, 2-level and   ║
// ║ 3-level hierarchical, and mixed scalar+tuple modes).               ║
// ║                                                                    ║
// ║ For each layout it records, for every flat index i in [0, size):   ║
// ║   - offsets[i]    = L(i)                  (flat eval)               ║
// ║   - top_coords[i] = colex decomposition   (eval_coord input)       ║
// ║   - idx2crd[i]    = idx2crd(i, L)         (nested coordinate)       ║
// ║ plus size, cosize, rank, and the "(shape):(stride)" repr.          ║
// ║                                                                    ║
// ║ The Python test (test_golden.py) reconstructs each layout from the ║
// ║ emitted (shape, stride) and asserts the JIT-backed Python Layout   ║
// ║ reproduces every one of these vectors. So the C++ headers define   ║
// ║ correctness and the whole binding stack (render -> JIT -> nanobind ║
// ║ -> Python) is validated against them.                              ║
// ║                                                                    ║
// ║ Build (host-only, no GPU):                                         ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/generate_golden.cpp -o /tmp/gen_golden   ║
// ║   /tmp/gen_golden python/tests/pymosaic/golden/golden_vectors.json          ║
// ║                                                                    ║
// ║ test_golden.py does NOT rewrite the committed file: its drift test ║
// ║ (test_golden_matches_cpp_source) recompiles this generator and     ║
// ║ asserts the committed JSON still matches, failing if you forgot to ║
// ║ regenerate after editing the headers. Rerun the two commands above ║
// ║ to refresh it.                                                     ║
// ╚══════════════════════════════════════════════════════════════════╝

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mosaic/mosaic.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mosaic;
using std::size_t;

namespace {

// --- tiny JSON emitters -------------------------------------------------
// No JSON library dependency: the values are all ints / nested int-tuples /
// short ASCII strings, so hand-rolled emitters keep the build trivial.

std::string j_int(long long v) {
    return std::to_string(v);
}

// Render a (possibly nested) mosaic int-tuple as a JSON array, scalars as
// bare numbers. Mirrors the nesting exactly so Python can rebuild the
// shape/stride and compare idx2crd outputs structurally.
template <class T>
std::string j_tuple(T const& v) {
    if constexpr (mosaic::is_tuple_v<T>) {
        std::string s = "[";
        bool first = true;
        std::apply(
            [&](auto const&... e) {
                (((s += (first ? "" : ","), first = false, s += j_tuple(e))), ...);
            },
            v);
        s += "]";
        return s;
    } else {
        return std::to_string(static_cast<long long>(v));
    }
}

std::string j_vec(std::vector<size_t> const& xs) {
    std::string s = "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(static_cast<long long>(xs[i]));
    }
    s += "]";
    return s;
}

std::string j_vecvec(std::vector<std::vector<size_t>> const& xss) {
    std::string s = "[";
    for (size_t i = 0; i < xss.size(); ++i) {
        if (i) s += ",";
        s += j_vec(xss[i]);
    }
    s += "]";
    return s;
}

std::string j_str(std::string const& v) {
    // Layout repr is only digits, parens, commas, colons — no escaping needed,
    // but guard against quotes/backslashes just in case the format changes.
    std::string s = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\') s += '\\';
        s += c;
    }
    s += "\"";
    return s;
}

// --- per-layout golden record ------------------------------------------

template <class L>
std::string emit_layout(std::string const& name, std::string const& family, L const& lay) {
    constexpr int R = L::rank;
    const size_t n = static_cast<size_t>(mosaic::size(lay));

    // Per top-level-mode sizes (product of that mode's leaves). Used to
    // colex-decompose a flat index into one scalar coordinate per mode —
    // exactly the eval_coord input on the Python side.
    std::array<size_t, R> mode_size{};
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((mode_size[I] = static_cast<size_t>(mosaic::size<static_cast<int>(I)>(lay))), ...);
    }(std::make_index_sequence<R>{});

    std::vector<size_t> offsets;
    std::vector<std::vector<size_t>> top_coords;
    std::string idx2crd_json = "[";
    offsets.reserve(n);
    top_coords.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        // Flat eval — the L(i) path.
        offsets.push_back(static_cast<size_t>(lay(i)));

        // Colex (column-major) decomposition into one scalar per mode.
        std::vector<size_t> coord(R);
        size_t rem = i;
        for (int r = 0; r < R; ++r) {
            coord[r] = (mode_size[r] == 0) ? 0 : rem % mode_size[r];
            if (mode_size[r] != 0) rem /= mode_size[r];
        }
        top_coords.push_back(coord);

        // idx2crd — nested coordinate matching the shape's profile.
        if (i) idx2crd_json += ",";
        idx2crd_json += j_tuple(mosaic::idx2crd(i, lay));
    }
    idx2crd_json += "]";

    std::ostringstream repr;
    repr << lay;

    std::ostringstream out;
    out << "  {\n";
    out << "    \"name\": " << j_str(name) << ",\n";
    out << "    \"family\": " << j_str(family) << ",\n";
    out << "    \"shape\": " << j_tuple(lay.shape()) << ",\n";
    out << "    \"stride\": " << j_tuple(lay.stride()) << ",\n";
    out << "    \"rank\": " << j_int(R) << ",\n";
    out << "    \"size\": " << j_int(static_cast<long long>(n)) << ",\n";
    out << "    \"cosize\": " << j_int(static_cast<long long>(mosaic::cosize(lay))) << ",\n";
    out << "    \"repr\": " << j_str(repr.str()) << ",\n";
    out << "    \"offsets\": " << j_vec(offsets) << ",\n";
    out << "    \"top_coords\": " << j_vecvec(top_coords) << ",\n";
    out << "    \"idx2crd\": " << idx2crd_json << "\n";
    out << "  }";
    return out.str();
}

// Shorthand for runtime size_t literals so every leaf is a runtime value
// (Layout<Shape<size_t,...>,...>), matching exactly what the JIT compiles.
constexpr size_t U(size_t x) {
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> records;

    // === flat rank-1 ===================================================
    records.push_back(emit_layout("flat_rank1_compact", "flat",
                                  make_layout(make_shape(U(12)), make_stride(U(1)))));
    records.push_back(emit_layout("flat_rank1_strided", "flat",
                                  make_layout(make_shape(U(6)), make_stride(U(3)))));

    // === flat rank-2 ===================================================
    records.push_back(emit_layout("flat_rank2_rowmajor", "flat",
                                  make_layout(make_shape(U(4), U(8)), make_stride(U(8), U(1)))));
    records.push_back(emit_layout("flat_rank2_colmajor", "flat",
                                  make_layout(make_shape(U(4), U(8)), make_stride(U(1), U(4)))));
    // Padded leading stride — cosize > size (a tile inside a wider buffer).
    records.push_back(emit_layout("flat_rank2_padded", "flat",
                                  make_layout(make_shape(U(4), U(8)), make_stride(U(16), U(1)))));

    // === flat rank-3 ===================================================
    records.push_back(
        emit_layout("flat_rank3_rowmajor", "flat",
                    make_layout(make_shape(U(2), U(3), U(4)), make_stride(U(12), U(4), U(1)))));
    records.push_back(
        emit_layout("flat_rank3_colmajor", "flat",
                    make_layout(make_shape(U(2), U(3), U(4)), make_stride(U(1), U(2), U(6)))));

    // === 2-level hierarchical ==========================================
    // ((2,2),(3,3)) assembled from two sub-layouts (auto col-major strides).
    {
        auto inner = make_layout(make_shape(U(2), U(2)));
        auto outer = make_layout(make_shape(U(3), U(3)));
        records.push_back(
            emit_layout("hier_2level_2x2_3x3", "hierarchical", make_layout(inner, outer)));
    }
    // Blocked-tiling style ((2,2),(2,2)) with explicit interleaved strides —
    // the shape produced by basic_tiling's blocked_product.
    records.push_back(
        emit_layout("hier_blocked_tiling", "hierarchical",
                    make_layout(make_shape(make_shape(U(2), U(2)), make_shape(U(2), U(2))),
                                make_stride(make_stride(U(1), U(2)), make_stride(U(4), U(8))))));

    // === mixed scalar + tuple modes ====================================
    // ((2,2),3): first mode nested, second mode a bare scalar.
    records.push_back(emit_layout("mixed_2x2_and_3", "mixed",
                                  make_layout(make_shape(make_shape(U(2), U(2)), U(3)),
                                              make_stride(make_stride(U(1), U(2)), U(4)))));

    // === 3-level hierarchical ==========================================
    // (((2,2),2),(4,)) — depth 3, models a tiled-of-tiled grid.
    {
        auto m0 = make_layout(make_shape(make_shape(U(2), U(2)), U(2)),
                              make_stride(make_stride(U(1), U(2)), U(4)));
        auto m1 = make_layout(make_shape(U(4)));
        records.push_back(emit_layout("hier_3level", "hierarchical", make_layout(m0, m1)));
    }

    std::ostringstream doc;
    doc << "[\n";
    for (size_t i = 0; i < records.size(); ++i) {
        doc << records[i];
        if (i + 1 < records.size()) doc << ",";
        doc << "\n";
    }
    doc << "]\n";

    if (argc > 1) {
        std::ofstream f(argv[1]);
        if (!f) {
            std::cerr << "generate_golden: cannot open " << argv[1] << " for writing\n";
            return 1;
        }
        f << doc.str();
        std::cerr << "wrote " << records.size() << " golden records to " << argv[1] << "\n";
    } else {
        std::cout << doc.str();
    }
    return 0;
}
