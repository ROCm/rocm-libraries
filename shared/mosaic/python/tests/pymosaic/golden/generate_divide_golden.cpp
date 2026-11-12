// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Golden-vector generator for pymosaic's logical_divide family       ║
// ║ (logical_divide, zipped_divide, tiled_divide, flat_divide).        ║
// ║                                                                    ║
// ║ Like composition, these are COMPILE-TIME, type-changing ops that   ║
// ║ only instantiate on STATIC (C<v>) layouts. We build references     ║
// ║ from compile-time constants and emit input + result (shape,        ║
// ║ stride). The "tiler" field is recorded in a Python-reconstructable ║
// ║ form: a layout tiler as {"layout": [shape, stride]}, an int tiler  ║
// ║ as {"int": N}, and a tuple tiler as {"tuple": [n0, n1, ...]}.      ║
// ║                                                                    ║
// ║ Build (host-only):                                                 ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/generate_divide_golden.cpp \    ║
// ║       -o /tmp/gen_divide                                           ║
// ║   /tmp/gen_divide python/tests/pymosaic/golden/divide_golden.json  ║
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

// Emit a record for a divide-family op. `tiler_json` is a pre-rendered JSON
// object describing how to reconstruct the tiler on the Python side.
template <class R>
std::string emit(std::string const& name, std::string const& op, std::string const& layout_shape,
                 std::string const& layout_stride, std::string const& tiler_json, R const& r) {
    std::ostringstream out;
    out << "  {\n";
    out << "    \"name\": " << j_str(name) << ",\n";
    out << "    \"op\": " << j_str(op) << ",\n";
    out << "    \"layout_shape\": " << layout_shape << ",\n";
    out << "    \"layout_stride\": " << layout_stride << ",\n";
    out << "    \"tiler\": " << tiler_json << ",\n";
    out << "    \"result_shape\": " << j_tuple(r.shape()) << ",\n";
    out << "    \"result_stride\": " << j_tuple(r.stride()) << "\n";
    out << "  }";
    return out.str();
}

template <auto V>
constexpr C<V> K{};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> recs;

    // --- logical_divide, integer tiler ---
    {
        auto A = make_layout(K<12>, K<1>);
        recs.push_back(emit("ld_int_12_4", "logical_divide", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"int\": 4}", logical_divide(A, K<4>)));
    }
    // --- logical_divide, permutation (layout) tiler ---
    {
        auto A = make_layout(K<8>, K<1>);
        auto T = make_layout(K<4>, K<2>);
        recs.push_back(
            emit("ld_perm_8_4x2", "logical_divide", j_tuple(A.shape()), j_tuple(A.stride()),
                 "{\"layout\": [" + j_tuple(T.shape()) + ", " + j_tuple(T.stride()) + "]}",
                 logical_divide(A, T)));
    }
    // --- logical_divide, tuple tiler on multi-mode ---
    {
        auto A = make_layout(make_shape(K<8>, K<6>), make_stride(K<6>, K<1>));
        auto T = std::make_tuple(K<2>, K<3>);
        recs.push_back(emit("ld_tuple_8x6", "logical_divide", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"tuple\": [2, 3]}", logical_divide(A, T)));
    }
    // --- logical_divide, layout tiler (contiguous) on flat ---
    {
        auto A = make_layout(K<16>, K<1>);
        auto T = make_layout(K<4>, K<1>);
        recs.push_back(
            emit("ld_layout_16_4", "logical_divide", j_tuple(A.shape()), j_tuple(A.stride()),
                 "{\"layout\": [" + j_tuple(T.shape()) + ", " + j_tuple(T.stride()) + "]}",
                 logical_divide(A, T)));
    }

    // --- zipped/tiled/flat divide on the tuple-tiler case ---
    {
        auto A = make_layout(make_shape(K<8>, K<6>), make_stride(K<6>, K<1>));
        auto T = std::make_tuple(K<2>, K<3>);
        recs.push_back(emit("zd_tuple_8x6", "zipped_divide", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"tuple\": [2, 3]}", zipped_divide(A, T)));
        recs.push_back(emit("td_tuple_8x6", "tiled_divide", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 3]}", tiled_divide(A, T)));
        recs.push_back(emit("fd_tuple_8x6", "flat_divide", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 3]}", flat_divide(A, T)));
    }

    // --- zipped_divide trailing-mode passthrough (rank-3 A, rank-2 T) ---
    {
        auto A = make_layout(make_shape(K<4>, K<6>, K<2>), make_stride(K<1>, K<8>, K<48>));
        auto T = std::make_tuple(K<2>, K<3>);
        recs.push_back(emit("zd_trailing", "zipped_divide", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 3]}", zipped_divide(A, T)));
    }

    // --- logical_divide coalesces a hierarchical input first ---
    {
        auto A = make_layout(make_shape(make_shape(K<2>, K<4>), K<3>),
                             make_stride(make_stride(K<1>, K<2>), K<8>));
        recs.push_back(emit("ld_coalesce_first", "logical_divide", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"int\": 4}", logical_divide(A, K<4>)));
    }

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
        std::cerr << "wrote " << recs.size() << " divide records to " << argv[1] << "\n";
    } else {
        std::cout << doc.str();
    }
    return 0;
}
