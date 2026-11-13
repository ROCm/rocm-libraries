// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Golden-vector generator for pymosaic's `coalesce` binding.         ║
// ║                                                                    ║
// ║ mosaic::coalesce is a COMPILE-TIME operation: try_merge's runtime  ║
// ║ branch returns two different tuple types under a non-constexpr if, ║
// ║ so it only instantiates on STATIC (C<v>) layouts. We therefore     ║
// ║ build each reference layout from compile-time constants, coalesce  ║
// ║ it with the header (the ground truth), and emit both the input and ║
// ║ the coalesced (shape, stride).                                     ║
// ║                                                                    ║
// ║ The Python test (test_coalesce.py) rebuilds the input as an        ║
// ║ ordinary (runtime) Layout, calls .coalesce() — which runs the      ║
// ║ runtime fold in the JIT module — and asserts the result matches    ║
// ║ this header-computed reference (scalar vs rank-1 normalised).      ║
// ║                                                                    ║
// ║ IMPORTANT: cases must avoid size-1 modes. For static layouts the   ║
// ║ header absorbs size-1 modes (is_constant_v<1,...>); the runtime    ║
// ║ fold cannot see that and does not. With no size-1 modes the two    ║
// ║ paths apply the identical (s_prev*d_prev == d) rule and agree.     ║
// ║                                                                    ║
// ║ Build (host-only):                                                 ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/generate_coalesce_golden.cpp \  ║
// ║       -o /tmp/gen_coalesce                                         ║
// ║   /tmp/gen_coalesce python/tests/pymosaic/golden/coalesce_golden.json ║
// ╚══════════════════════════════════════════════════════════════════╝

#include <cstddef>
#include <fstream>
#include <iostream>
#include <mosaic/mosaic.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace mosaic;
using std::size_t;

namespace {

// Render a (possibly nested) mosaic int-tuple as a JSON array; scalar leaves
// (runtime size_t or compile-time C<v>) become bare numbers.
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

std::string j_str(std::string const& v) {
    return "\"" + v + "\"";
}

// Emit one {name, shape, stride, coalesce_shape, coalesce_stride} record.
// `lay` is built from C<v> constants so coalesce(lay) instantiates.
template <class L>
std::string emit(std::string const& name, L const& lay) {
    auto co = coalesce(lay);
    std::ostringstream out;
    out << "  {\n";
    out << "    \"name\": " << j_str(name) << ",\n";
    out << "    \"shape\": " << j_tuple(lay.shape()) << ",\n";
    out << "    \"stride\": " << j_tuple(lay.stride()) << ",\n";
    out << "    \"coalesce_shape\": " << j_tuple(co.shape()) << ",\n";
    out << "    \"coalesce_stride\": " << j_tuple(co.stride()) << "\n";
    out << "  }";
    return out.str();
}

template <auto V>
constexpr C<V> K{};  // shorthand for a compile-time constant leaf

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> recs;

    // --- flat, fully contiguous -> collapse to one mode -----------------
    // column-major (4,8):(1,4): 4*1==4 -> merge -> (32):(1)
    recs.push_back(emit("cm_4x8", make_layout(make_shape(K<4>, K<8>), make_stride(K<1>, K<4>))));
    // column-major (2,3,4):(1,2,6) -> (24):(1)
    recs.push_back(
        emit("cm_2x3x4", make_layout(make_shape(K<2>, K<3>, K<4>), make_stride(K<1>, K<2>, K<6>))));

    // --- flat, no contiguous merge --------------------------------------
    // row-major (4,8):(8,1): 4*8==32 != 1 -> stays (4,8):(8,1)
    recs.push_back(emit("rm_4x8", make_layout(make_shape(K<4>, K<8>), make_stride(K<8>, K<1>))));
    // padded leading stride (4,8):(16,1): 4*16==64 != 1 -> stays
    recs.push_back(
        emit("padded_4x8", make_layout(make_shape(K<4>, K<8>), make_stride(K<16>, K<1>))));

    // --- flat, partial merge --------------------------------------------
    // (2,4,8):(1,2,16): merge 0,1 (2*1==2) -> (8,16):... then 8*1==8 != 16
    //   wait: result-of-merge stride is 1, 8*1==8 != 16 -> (8):(1),(8):(16)
    //   -> (8,8):(1,16). Let the header decide; we just record it.
    recs.push_back(emit("partial_2x4x8",
                        make_layout(make_shape(K<2>, K<4>, K<8>), make_stride(K<1>, K<2>, K<16>))));

    // --- hierarchical (nested) -> flatten then coalesce -----------------
    // ((2,2),(2,2)):((1,2),(4,8)) flattens to (2,2,2,2):(1,2,4,8) -> (16):(1)
    recs.push_back(
        emit("nested_blocked",
             make_layout(make_shape(make_shape(K<2>, K<2>), make_shape(K<2>, K<2>)),
                         make_stride(make_stride(K<1>, K<2>), make_stride(K<4>, K<8>)))));
    // mixed ((2,2),3):((1,2),4) -> flat (2,2,3):(1,2,4) -> (12):(1)
    recs.push_back(emit("mixed_2x2_3", make_layout(make_shape(make_shape(K<2>, K<2>), K<3>),
                                                   make_stride(make_stride(K<1>, K<2>), K<4>))));

    // --- single-mode (nothing to merge) ---------------------------------
    recs.push_back(emit("strided_single", make_layout(make_shape(K<4>), make_stride(K<2>))));

    std::ostringstream doc;
    doc << "[\n";
    for (size_t i = 0; i < recs.size(); ++i) {
        doc << recs[i] << (i + 1 < recs.size() ? "," : "") << "\n";
    }
    doc << "]\n";

    if (argc > 1) {
        std::ofstream f(argv[1]);
        if (!f) {
            std::cerr << "cannot open " << argv[1] << "\n";
            return 1;
        }
        f << doc.str();
        std::cerr << "wrote " << recs.size() << " coalesce records to " << argv[1] << "\n";
    } else {
        std::cout << doc.str();
    }
    return 0;
}
