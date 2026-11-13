// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Golden-vector generator for pymosaic's logical_product family      ║
// ║ (logical_product, zipped_product, tiled_product, flat_product).    ║
// ║                                                                    ║
// ║ COMPILE-TIME, type-changing ops — see the divide/composition       ║
// ║ generators for the rationale. Tiler encoding matches               ║
// ║ generate_divide_golden.cpp: {"layout": [shape, stride]} /          ║
// ║ {"int": N} / {"tuple": [...]}.                                     ║
// ║                                                                    ║
// ║ Build (host-only):                                                 ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/generate_product_golden.cpp \   ║
// ║       -o /tmp/gen_product                                          ║
// ║   /tmp/gen_product python/tests/pymosaic/golden/product_golden.json║
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

template <class R>
std::string emit(std::string const& name, std::string const& op, std::string const& block_shape,
                 std::string const& block_stride, std::string const& tiler_json, R const& r) {
    std::ostringstream out;
    out << "  {\n";
    out << "    \"name\": " << j_str(name) << ",\n";
    out << "    \"op\": " << j_str(op) << ",\n";
    out << "    \"block_shape\": " << block_shape << ",\n";
    out << "    \"block_stride\": " << block_stride << ",\n";
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

    // --- rank-1 Layout x rank-1 Layout ---
    {
        auto A = make_layout(K<2>, K<1>);
        auto B = make_layout(K<3>, K<1>);
        recs.push_back(
            emit("lp_flat_2x3", "logical_product", j_tuple(A.shape()), j_tuple(A.stride()),
                 "{\"layout\": [" + j_tuple(B.shape()) + ", " + j_tuple(B.stride()) + "]}",
                 logical_product(A, B)));
    }
    {
        auto A = make_layout(K<2>, K<2>);
        auto B = make_layout(K<2>, K<1>);
        recs.push_back(
            emit("lp_nonunit", "logical_product", j_tuple(A.shape()), j_tuple(A.stride()),
                 "{\"layout\": [" + j_tuple(B.shape()) + ", " + j_tuple(B.stride()) + "]}",
                 logical_product(A, B)));
    }
    // --- multi-mode block x multi-mode layout tiler ---
    {
        auto A = make_layout(make_shape(K<2>, K<2>), make_stride(K<1>, K<8>));
        auto B = make_layout(make_shape(K<2>, K<2>), make_stride(K<1>, K<2>));
        recs.push_back(
            emit("lp_multi", "logical_product", j_tuple(A.shape()), j_tuple(A.stride()),
                 "{\"layout\": [" + j_tuple(B.shape()) + ", " + j_tuple(B.stride()) + "]}",
                 logical_product(A, B)));
    }
    // --- integer tiler ---
    {
        auto A = make_layout(K<2>, K<1>);
        recs.push_back(emit("lp_int_2_3", "logical_product", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"int\": 3}", logical_product(A, K<3>)));
    }
    // --- tuple tiler (per-mode; size-only postcondition) ---
    {
        auto A = make_layout(make_shape(K<2>, K<2>), make_stride(K<1>, K<2>));
        auto T = std::make_tuple(K<2>, K<2>);
        recs.push_back(emit("lp_tuple", "logical_product", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 2]}", logical_product(A, T)));
    }

    // --- product-family wrappers on the tuple-tiler case ---
    {
        auto A = make_layout(make_shape(K<2>, K<2>), make_stride(K<1>, K<2>));
        auto T = std::make_tuple(K<2>, K<2>);
        recs.push_back(emit("zp_tuple", "zipped_product", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 2]}", zipped_product(A, T)));
        recs.push_back(emit("tp_tuple", "tiled_product", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 2]}", tiled_product(A, T)));
        recs.push_back(emit("fp_tuple", "flat_product", j_tuple(A.shape()), j_tuple(A.stride()),
                            "{\"tuple\": [2, 2]}", flat_product(A, T)));
    }

    // --- trailing block mode past tiler rank (rank-3 block, rank-2 tiler) ---
    {
        auto A = make_layout(make_shape(K<2>, K<2>, K<3>), make_stride(K<1>, K<2>, K<24>));
        auto T = std::make_tuple(K<2>, K<2>);
        recs.push_back(emit("lp_block_gt_tiler", "logical_product", j_tuple(A.shape()),
                            j_tuple(A.stride()), "{\"tuple\": [2, 2]}", logical_product(A, T)));
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
        std::cerr << "wrote " << recs.size() << " product records to " << argv[1] << "\n";
    } else {
        std::cout << doc.str();
    }
    return 0;
}
