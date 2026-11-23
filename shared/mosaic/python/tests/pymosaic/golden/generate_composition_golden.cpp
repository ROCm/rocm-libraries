// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Golden-vector generator for pymosaic's `composition` binding.      ║
// ║                                                                    ║
// ║ mosaic::composition is a COMPILE-TIME, type-changing operation:    ║
// ║ the result's nesting structure depends on the input stride VALUES, ║
// ║ so it only instantiates on STATIC (C<v>) layouts. We build each    ║
// ║ reference layout from compile-time constants, compose with the     ║
// ║ header (the ground truth), and emit the input + result            ║
// ║ (shape, stride).                                                   ║
// ║                                                                    ║
// ║ The Python test (test_composition.py) rebuilds the inputs as       ║
// ║ ordinary Layouts, calls .composition(...) — which runs the pure-   ║
// ║ Python port in pymosaic._algebra — and asserts the result matches  ║
// ║ this header-computed reference structurally.                       ║
// ║                                                                    ║
// ║ Build (host-only):                                                 ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/generate_composition_golden.cpp \║
// ║       -o /tmp/gen_composition                                      ║
// ║   /tmp/gen_composition python/tests/pymosaic/golden/composition_golden.json ║
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

// Emit one {name, lhs_shape, lhs_stride, rhs_shape, rhs_stride,
// composition_shape, composition_stride} record. The lhs/rhs are recorded so
// the Python side can reconstruct the exact inputs.
template <class A, class B>
std::string emit(std::string const& name, A const& a, B const& b) {
    auto c = composition(a, b);
    std::ostringstream out;
    out << "  {\n";
    out << "    \"name\": " << j_str(name) << ",\n";
    out << "    \"lhs_shape\": " << j_tuple(a.shape()) << ",\n";
    out << "    \"lhs_stride\": " << j_tuple(a.stride()) << ",\n";
    out << "    \"rhs_shape\": " << j_tuple(b.shape()) << ",\n";
    out << "    \"rhs_stride\": " << j_tuple(b.stride()) << ",\n";
    out << "    \"composition_shape\": " << j_tuple(c.shape()) << ",\n";
    out << "    \"composition_stride\": " << j_tuple(c.stride()) << "\n";
    out << "  }";
    return out.str();
}

template <auto V>
constexpr C<V> K{};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> recs;

    // Branch 4: flat-scalar LHS, closed form.
    recs.push_back(emit("flatA_unit", make_layout(K<8>, K<1>), make_layout(K<4>, K<2>)));
    recs.push_back(emit("flatA_nonunit", make_layout(K<8>, K<3>), make_layout(K<4>, K<2>)));

    // Branch 3: RHS stride-0 shortcut.
    recs.push_back(emit("rhs_stride0", make_layout(K<8>, K<1>), make_layout(K<4>, K<0>)));

    // Branch 5: multi-mode LHS, cross-boundary RHS.
    recs.push_back(emit("span_boundary",
                        make_layout(make_shape(K<4>, K<3>), make_stride(K<1>, K<4>)),
                        make_layout(K<6>, K<2>)));
    recs.push_back(emit("unwrap", make_layout(make_shape(K<4>, K<3>), make_stride(K<1>, K<4>)),
                        make_layout(K<4>, K<1>)));
    recs.push_back(emit("skip_first", make_layout(make_shape(K<4>, K<3>), make_stride(K<1>, K<4>)),
                        make_layout(K<3>, K<4>)));
    recs.push_back(emit("wrapup_empty",
                        make_layout(make_shape(K<4>, K<3>), make_stride(K<1>, K<4>)),
                        make_layout(K<12>, K<1>)));

    // Branch 1: tuple RHS, right-distributivity.
    recs.push_back(emit("tupleB", make_layout(K<8>, K<1>),
                        make_layout(make_shape(K<2>, K<4>), make_stride(K<1>, K<2>))));

    // 2D LHS x 2D RHS.
    recs.push_back(emit("multi_multi", make_layout(make_shape(K<8>, K<4>), make_stride(K<4>, K<1>)),
                        make_layout(make_shape(K<4>, K<2>), make_stride(K<1>, K<4>))));

    // Permutation tiler (gather evens).
    recs.push_back(emit("perm_evens", make_layout(K<8>, K<1>), make_layout(K<4>, K<2>)));

    // Non-injective LHS (stride-0 mode).
    recs.push_back(emit("noninjective_lhs",
                        make_layout(make_shape(K<4>, K<4>), make_stride(K<1>, K<0>)),
                        make_layout(K<4>, K<1>)));

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
        std::cerr << "wrote " << recs.size() << " composition records to " << argv[1] << "\n";
    } else {
        std::cout << doc.str();
    }
    return 0;
}
