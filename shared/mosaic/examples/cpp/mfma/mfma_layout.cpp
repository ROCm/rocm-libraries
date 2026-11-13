// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Demonstrates expressing AMD MFMA layouts as canonical CuTe-style
// thread-value (TV) layouts in mosaic.
//
// A TV layout maps (tid, vid) -> matrix-coord, where
//   tid is the lane id within the wavefront (0..63 for wave64), and
//   vid is the per-lane value/VGPR id (0..N-1 for an N-element-per-lane
//        operand of a given MFMA instruction).
//
// Hierarchical (shape, stride) tuples capture the lane and per-lane
// decomposition of how the matrix elements are distributed across the
// wavefront. The layout is just a single mosaic::Layout — no
// register-layout / inverse / matrix-layout detour.
//
// This example pins down the TV layouts for the CDNA3 instruction
//
//     v_mfma_f32_32x32x8_f16
//
// TV layouts (column-major flat index into the operand):
//
//     A:  ((32, 2), 4)      :((1, 128), 32)        rows=M=32, cols=K=8
//     B:  ((32, 2), 4)      :((1, 128), 32)        rows=N=32, cols=K=8
//     D:  ((32, 2), (4, 4)) :((32, 4), (1, 8))     rows=M=32, cols=N=32
//
// The matrix-coord that each layout maps to is a column-major flat index
// into the operand: flat = j * rows + i, where (i, j) is (row, col)
// within the operand (with A and B held in M-K and N-K orientation
// respectively).

#include <cassert>
#include <fstream>
#include <iostream>
#include <mosaic/mosaic.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

constexpr std::size_t WAVE = 64;

// One ground-truth entry parsed from a calculator CSV row:
// (tid, vid) → operand cell (i, j, block) using the calculator's own
// (i, j) convention (A[i][j], B[i][j], D[i][j]).
struct CalcCell {
    std::size_t tid, vid, i, j, block;
};

// Parse one of the calculator's --matrix-layout --csv outputs.
// Header: 3 lines (Architecture / Instruction / column header).
// Body: lane,cell_v0,cell_v1,...  where each cell is "X[i][j]" or
// "X[i][j].B<n>"  (X ∈ {A,B,C,D,K}). Empty / non-matching cells are
// skipped (the calculator emits "" for inactive vid slots).
inline std::vector<CalcCell> parse_calc_csv(const std::string& path) {
    static const std::regex cell_re(R"(^[ABCDK]\[(\d+)\]\[(\d+)\](?:\.B(\d+))?$)",
                                    std::regex::icase);

    std::ifstream f(path);
    if (!f) {
        std::cerr << "could not open " << path << "\n";
        std::abort();
    }

    std::vector<CalcCell> out;
    std::string line;
    int header_lines_remaining = 3;
    while (std::getline(f, line)) {
        if (header_lines_remaining > 0) {
            --header_lines_remaining;
            continue;
        }

        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim whitespace
            auto a = tok.find_first_not_of(" \t\r\n");
            auto b = tok.find_last_not_of(" \t\r\n");
            tok = (a == std::string::npos) ? "" : tok.substr(a, b - a + 1);
            fields.push_back(tok);
        }
        if (fields.empty()) continue;

        std::size_t tid;
        try {
            tid = std::stoull(fields[0]);
        } catch (...) {
            continue;
        }

        for (std::size_t vid = 0; vid + 1 < fields.size(); ++vid) {
            std::smatch m;
            if (!std::regex_match(fields[vid + 1], m, cell_re)) continue;
            CalcCell c;
            c.tid = tid;
            c.vid = vid;
            c.i = std::stoull(m[1].str());
            c.j = std::stoull(m[2].str());
            c.block = m[3].matched ? std::stoull(m[3].str()) : 0;
            out.push_back(c);
        }
    }
    return out;
}

int main() {
    using mosaic::Layout;
    using mosaic::make_shape;
    using mosaic::make_stride;

    // ---------- A: 32 (M) x 8 (K), col-major flat ----------
    //   tid: (32 lanes per col, 2 col-groups), strides (1, 128)
    //   vid: 4 VGPRs along K,                  stride 32
    auto A_tv =
        Layout(make_shape(make_shape(32u, 2u), 4u), make_stride(make_stride(1u, 128u), 32u));
    constexpr std::size_t A_rows = 32, A_cols = 8;  // M, K
    constexpr std::size_t A_vals = 4;

    // ---------- B: 32 (N) x 8 (K), col-major flat ----------
    auto B_tv =
        Layout(make_shape(make_shape(32u, 2u), 4u), make_stride(make_stride(1u, 128u), 32u));
    constexpr std::size_t B_rows = 32, B_cols = 8;  // N, K
    constexpr std::size_t B_vals = 4;

    // ---------- D: 32 (M) x 32 (N), col-major flat ----------
    //   tid: (32 lanes per col-block, 2 col-blocks), strides (32, 4)
    //   vid: (4 row-groups, 4 rows per group),       strides (1, 8)
    auto D_tv = Layout(make_shape(make_shape(32u, 2u), make_shape(4u, 4u)),
                       make_stride(make_stride(32u, 4u), make_stride(1u, 8u)));
    constexpr std::size_t D_rows = 32, D_cols = 32;  // M, N
    constexpr std::size_t D_vals = 16;

    std::cout << "v_mfma_f32_32x32x8_f16 — TV layouts (tid, vid) -> col-major flat\n";
    std::cout << "  A_tv = " << A_tv << "   (rows=" << A_rows << ", cols=" << A_cols << ")\n";
    std::cout << "  B_tv = " << B_tv << "   (rows=" << B_rows << ", cols=" << B_cols << ")\n";
    std::cout << "  D_tv = " << D_tv << "   (rows=" << D_rows << ", cols=" << D_cols << ")\n\n";

    // ---------- Worked example: where does (lane=5, vid=2) land in D? ----------
    // L(tid, vid) accepts flat scalar tid/vid; the call internally idx2crd's
    // tid against (32, 2) and vid against (4, 4), then takes the inner
    // product with the strides. See include/mosaic/stride.hpp:79-107.
    {
        const std::size_t lane = 5, vid = 2;
        const std::size_t flat = D_tv(lane, vid);
        const auto rc = mosaic::idx2crd(flat, make_shape(D_rows, D_cols));
        std::cout << "Worked example: D_tv(lane=" << lane << ", vid=" << vid << ") = " << flat
                  << "  ->  (row=" << std::get<0>(rc) << ", col=" << std::get<1>(rc) << ")\n\n";
    }

    // ---------- Bijectivity check: every matrix cell touched exactly once ----------
    auto check = [](auto const& tv, std::size_t rows, std::size_t cols, std::size_t vals,
                    const char* name) {
        std::vector<int> hit(rows * cols, 0);
        for (std::size_t tid = 0; tid < WAVE; ++tid) {
            for (std::size_t vid = 0; vid < vals; ++vid) {
                std::size_t f = tv(tid, vid);
                assert(f < rows * cols && "out of range");
                ++hit[f];
            }
        }
        for (std::size_t f = 0; f < rows * cols; ++f) {
            assert(hit[f] == 1 && "TV layout is not a bijection over the operand");
        }
        std::cout << "  " << name << ": " << (rows * cols) << " cells covered exactly once by "
                  << WAVE << " lanes x " << vals << " VGPRs.\n";
    };

    std::cout << "Bijectivity:\n";
    check(A_tv, A_rows, A_cols, A_vals, "A");
    check(B_tv, B_rows, B_cols, B_vals, "B");
    check(D_tv, D_rows, D_cols, D_vals, "D");

    // ---------- Ground-truth check vs. AMD's matrix calculator ----------
    // For each (tid, vid) the calculator says "this VGPR holds operand cell
    // X[i][j]". We compute the same cell from the TV layout and assert.
    //
    // Note: the calculator labels B as B[k][n] (k-major), but our B_tv
    // canonicalizes to (rows=N, cols=K) so the calculator's (i, j) maps
    // to our (j, i) — pass swap=true for B.
    auto verify_against_calc = [](auto const& tv, std::size_t rows, std::size_t cols,
                                  const std::string& csv_path, bool swap, const char* name) {
        const auto cells = parse_calc_csv(csv_path);
        assert(!cells.empty() && "calculator CSV produced no cells");
        std::size_t mismatches = 0;
        for (auto const& c : cells) {
            const std::size_t our_i = swap ? c.j : c.i;
            const std::size_t our_j = swap ? c.i : c.j;
            const std::size_t expected = c.block * (rows * cols) + our_j * rows + our_i;
            const std::size_t got = tv(c.tid, c.vid);
            if (got != expected) {
                if (mismatches < 5) {
                    std::cerr << "  MISMATCH " << name << "(tid=" << c.tid << ", vid=" << c.vid
                              << "): TV gave " << got << ", calculator says " << name << "[" << c.i
                              << "][" << c.j << "].B" << c.block << " (expected flat " << expected
                              << ")\n";
                }
                ++mismatches;
            }
        }
        std::cout << "  " << name << ": " << cells.size() << " calculator entries verified"
                  << (mismatches ? "" : " (all match)") << "\n";
        assert(mismatches == 0 && "TV layout disagrees with calculator");
    };

#ifndef MFMA_DATA_DIR
#define MFMA_DATA_DIR "matrix_calculator_layouts"
#endif
    const std::string ref_dir =
        std::string(MFMA_DATA_DIR) + "/CDNA3_v_mfma_f32_32x32x8_f16/CDNA3_v_mfma_f32_32x32x8_f16_";
    const std::string a_csv = ref_dir + "A_matrix_layout.csv";
    if (!std::ifstream(a_csv)) {
        std::cout << "\n(skip) calculator CSVs not found under " << MFMA_DATA_DIR << "\n";
        return 0;
    }
    std::cout << "\nGround-truth check vs. AMD matrix calculator:\n";
    verify_against_calc(A_tv, A_rows, A_cols, a_csv,
                        /*swap=*/false, "A");
    verify_against_calc(B_tv, B_rows, B_cols, ref_dir + "B_matrix_layout.csv",
                        /*swap=*/true, "B");
    verify_against_calc(D_tv, D_rows, D_cols, ref_dir + "D_matrix_layout.csv",
                        /*swap=*/false, "D");

    return 0;
}
