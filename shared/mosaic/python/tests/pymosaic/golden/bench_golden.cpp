// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ╔══════════════════════════════════════════════════════════════════╗
// ║ Scale golden generator — ground-truth hashes for many layouts.     ║
// ║                                                                    ║
// ║ Reads a whitespace-delimited spec file (written by scale_spec.py's ║
// ║ write_cpp_input) and, for every layout, computes — straight from   ║
// ║ the mosaic C++ headers — a 64-bit FNV-1a hash of the full offset    ║
// ║ enumeration [L(0)..L(size-1)], plus size, cosize, and the          ║
// ║ "(shape):(stride)" repr.                                           ║
// ║                                                                    ║
// ║ Input  line:  sid k  s0..s{k-1}  d0..d{k-1}                        ║
// ║ Output line:  sid size cosize rank h_eval h_idx2crd repr           ║
// ║   h_eval    = FNV-1a of flat offsets [L(0)..L(n-1)]                ║
// ║   h_idx2crd = FNV-1a of all idx2crd(i,L) coordinate leaves (DFS)   ║
// ║                                                                    ║
// ║ Each `sid` selects a compile-time structure (nesting profile);     ║
// ║ leaf values are runtime — exactly the JIT's "one type, many        ║
// ║ values" model. The sid->structure mapping MUST match STRUCTURES    ║
// ║ in scale_spec.py.                                                  ║
// ║                                                                    ║
// ║ Build (host-only):                                                 ║
// ║   g++ -std=c++20 -O2 -I include \                                  ║
// ║       python/tests/pymosaic/golden/bench_golden.cpp -o /tmp/bench_golden    ║
// ║   /tmp/bench_golden specs.txt out.txt                              ║
// ╚══════════════════════════════════════════════════════════════════╝

#include <cstddef>
#include <cstdint>
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

constexpr std::uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr std::uint64_t FNV_PRIME = 0x100000001b3ULL;

// count_leaves / fill_from_flat — identical in spirit to the JIT template's
// helpers: walk a (possibly nested) tuple type and populate it from a flat
// left-to-right leaf stream.
template <class T>
constexpr size_t count_leaves() {
    if constexpr (mosaic::is_tuple_v<T>) {
        return [&]<size_t... I>(std::index_sequence<I...>) {
            return (count_leaves<std::tuple_element_t<I, T>>() + ... + size_t{0});
        }(std::make_index_sequence<std::tuple_size_v<T>>{});
    } else {
        return 1;
    }
}

template <class T>
void fill_from_flat(T& out, const size_t*& p) {
    if constexpr (mosaic::is_tuple_v<T>) {
        std::apply([&](auto&... elems) { (fill_from_flat(elems, p), ...); }, out);
    } else {
        out = *p++;
    }
}

// Fold one unsigned value into a running FNV-1a state, 8 little-endian bytes.
inline void fnv_step(std::uint64_t& h, std::uint64_t v) {
    for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xffULL;
        h *= FNV_PRIME;
    }
}

// Fold all leaves of a (possibly nested) tuple into FNV in DFS left-to-right
// order — matching mosaic::flatten / _render.flatten on the Python side.
template <class T>
void fnv_leaves(std::uint64_t& h, T const& v) {
    if constexpr (mosaic::is_tuple_v<T>) {
        std::apply([&](auto const&... elems) { (fnv_leaves(h, elems), ...); }, v);
    } else {
        fnv_step(h, static_cast<std::uint64_t>(v));
    }
}

// Build a typed layout L_t from flat shape/stride leaves, compute the offset
// hash + scalars, and write one output line.
template <class L_t>
void run(int sid, const std::vector<size_t>& sh, const std::vector<size_t>& st, std::ostream& out) {
    using S_t = std::remove_cvref_t<decltype(std::declval<L_t>().shape())>;
    using D_t = std::remove_cvref_t<decltype(std::declval<L_t>().stride())>;

    S_t s{};
    D_t d{};
    const size_t* sp = sh.data();
    const size_t* dp = st.data();
    fill_from_flat(s, sp);
    fill_from_flat(d, dp);
    L_t L{s, d};

    const size_t n = static_cast<size_t>(L.size());

    // h_eval     : hash of the flat-eval offsets [L(0)..L(n-1)].
    // h_idx2crd  : hash of every idx2crd(i, L) coordinate's leaves (DFS).
    std::uint64_t h_eval = FNV_OFFSET;
    std::uint64_t h_idx2crd = FNV_OFFSET;
    for (size_t i = 0; i < n; ++i) {
        fnv_step(h_eval, static_cast<std::uint64_t>(L(i)));
        fnv_leaves(h_idx2crd, mosaic::idx2crd(i, L));
    }

    std::ostringstream repr;
    repr << L;

    out << sid << ' ' << n << ' ' << static_cast<long long>(cosize(L)) << ' '
        << static_cast<int>(L_t::rank) << ' ' << h_eval << ' ' << h_idx2crd << ' ' << repr.str()
        << '\n';
}

// sid -> compile-time Layout type. MUST match STRUCTURES in scale_spec.py.
// Each leaf is size_t (runtime value). Shape/Stride share the profile.
template <class... Ts>
using Sh = Shape<Ts...>;
template <class... Ts>
using Sd = Stride<Ts...>;
using z = size_t;

// sid -> compile-time Layout type. MUST match STRUCTURES in scale_spec.py:
// same index, same nesting profile. Each leaf is a runtime size_t.
void dispatch(int sid, const std::vector<size_t>& sh, const std::vector<size_t>& st,
              std::ostream& out) {
    switch (sid) {
        case 0:
            run<Layout<Sh<z>, Sd<z>>>(sid, sh, st, out);
            break;
        case 1:
            run<Layout<Sh<z, z>, Sd<z, z>>>(sid, sh, st, out);
            break;
        case 2:
            run<Layout<Sh<z, z, z>, Sd<z, z, z>>>(sid, sh, st, out);
            break;
        case 3:
            run<Layout<Sh<z, z, z, z>, Sd<z, z, z, z>>>(sid, sh, st, out);
            break;
        case 4:
            run<Layout<Sh<z, z, z, z, z>, Sd<z, z, z, z, z>>>(sid, sh, st, out);
            break;
        case 5:
            run<Layout<Sh<Sh<z, z>, Sh<z, z>>, Sd<Sd<z, z>, Sd<z, z>>>>(sid, sh, st, out);
            break;
        case 6:
            run<Layout<Sh<Sh<z, z, z>, Sh<z, z>>, Sd<Sd<z, z, z>, Sd<z, z>>>>(sid, sh, st, out);
            break;
        case 7:
            run<Layout<Sh<Sh<z, z>, Sh<z, z>, Sh<z, z>>, Sd<Sd<z, z>, Sd<z, z>, Sd<z, z>>>>(
                sid, sh, st, out);
            break;
        case 8:
            run<Layout<Sh<Sh<z, z>, z>, Sd<Sd<z, z>, z>>>(sid, sh, st, out);
            break;
        case 9:
            run<Layout<Sh<z, Sh<z, z>>, Sd<z, Sd<z, z>>>>(sid, sh, st, out);
            break;
        case 10:
            run<Layout<Sh<Sh<z, z>, z, z>, Sd<Sd<z, z>, z, z>>>(sid, sh, st, out);
            break;
        case 11:
            run<Layout<Sh<z, Sh<z, z>, z>, Sd<z, Sd<z, z>, z>>>(sid, sh, st, out);
            break;
        case 12:
            run<Layout<Sh<Sh<z, z>, Sh<z, z>, z>, Sd<Sd<z, z>, Sd<z, z>, z>>>(sid, sh, st, out);
            break;
        case 13:
            run<Layout<Sh<Sh<z>, Sh<z>>, Sd<Sd<z>, Sd<z>>>>(sid, sh, st, out);
            break;
        case 14:
            run<Layout<Sh<Sh<z>>, Sd<Sd<z>>>>(sid, sh, st, out);
            break;
        case 15:
            run<Layout<Sh<Sh<Sh<z, z>, z>, Sh<z>>, Sd<Sd<Sd<z, z>, z>, Sd<z>>>>(sid, sh, st, out);
            break;
        case 16:
            run<Layout<Sh<Sh<z, Sh<z, z>>, z>, Sd<Sd<z, Sd<z, z>>, z>>>(sid, sh, st, out);
            break;
        case 17:
            run<Layout<Sh<Sh<Sh<z, z>, Sh<z, z>>, Sh<z, z>>, Sd<Sd<Sd<z, z>, Sd<z, z>>, Sd<z, z>>>>(
                sid, sh, st, out);
            break;
        case 18:
            run<Layout<Sh<Sh<Sh<Sh<z, z>, z>, z>, z>, Sd<Sd<Sd<Sd<z, z>, z>, z>, z>>>(sid, sh, st,
                                                                                      out);
            break;
        case 19:
            run<Layout<Sh<Sh<Sh<z, z>, z>, Sh<Sh<z, z>, z>>, Sd<Sd<Sd<z, z>, z>, Sd<Sd<z, z>, z>>>>(
                sid, sh, st, out);
            break;
        case 20:
            run<Layout<Sh<z, Sh<Sh<z, z>, z>>, Sd<z, Sd<Sd<z, z>, z>>>>(sid, sh, st, out);
            break;
        case 21:
            run<Layout<Sh<Sh<z, z, z>, z>, Sd<Sd<z, z, z>, z>>>(sid, sh, st, out);
            break;
        case 22:
            run<Layout<Sh<Sh<z, z>, Sh<z, z, z>, z>, Sd<Sd<z, z>, Sd<z, z, z>, z>>>(sid, sh, st,
                                                                                    out);
            break;
        case 23:
            run<Layout<Sh<Sh<Sh<z>>>, Sd<Sd<Sd<z>>>>>(sid, sh, st, out);
            break;
        default:
            throw std::runtime_error("bench_golden: unknown sid " + std::to_string(sid));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: bench_golden <specs_in> <golden_out>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "bench_golden: cannot open input " << argv[1] << "\n";
        return 1;
    }
    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "bench_golden: cannot open output " << argv[2] << "\n";
        return 1;
    }

    long long count = 0;
    in >> count;

    long long done = 0;
    int sid = 0;
    size_t k = 0;
    while (in >> sid >> k) {
        std::vector<size_t> sh(k), st(k);
        for (size_t i = 0; i < k; ++i) in >> sh[i];
        for (size_t i = 0; i < k; ++i) in >> st[i];
        dispatch(sid, sh, st, out);
        ++done;
    }

    std::cerr << "bench_golden: processed " << done << " layouts";
    if (count) std::cerr << " (expected " << count << ")";
    std::cerr << "\n";
    return (count && done != count) ? 1 : 0;
}
