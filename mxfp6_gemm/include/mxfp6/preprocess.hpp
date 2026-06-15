#pragma once
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "mxfp6/gemm.hpp"  // K_TILE, kpad
#include "mxfp6/types.hpp"

namespace mxfp6 {

// ---- K-padding helpers: "pad-B-only / compact-A" recipe -----------------------------------
// The kernel reads K in deep tiles, so K must be padded to kpad(K) (Kp). B (weights) is padded
// offline with a ZERO K-tail. A (activations) can then stay in its NATURAL compact layout —
// row stride = fp6_packed_bytes(K), A_row_bytes = fp6_packed_bytes(K) — WITHOUT per-row padding,
// because each row's K-tail read overlaps the next row's real data (or, for the last row, the
// end pad), all nulled by B's zero K-tail. Two obligations remain:
//   (1) over-allocate a_compact_end_pad(K) bytes at the END of the A buffer (the last row's
//       K-tail read; content irrelevant but must be in bounds), and
//   (2) extend A's per-block scales to kpad(K) with a NON-NaN tail — a 0xFF (E8M0 NaN) scale
//       would poison the WHOLE output via 0*NaN. pad_scales_k does this.
// Measured perf-neutral vs per-row padding @8192^3; correctness gated in tests/test_gemm.cpp.

// Extra bytes to over-allocate at the end of a COMPACT A buffer (row stride = packed(K)) so the
// last row's K-tail read stays in bounds.
inline int a_compact_end_pad(int K) {
    return fp6_packed_bytes(kpad(K)) - fp6_packed_bytes(K);
}

// Extend per-block scales [dim][K/32] -> [dim][kpad(K)/32]: real blocks copied, K-tail blocks set
// to a finite (non-NaN) E8M0 code (127 = 2^0). Feed the result to preprocess_scale(..., kpad(K)).
inline std::vector<uint8_t> pad_scales_k(const uint8_t* scales, int dim, int K) {
    int sc = K / 32, scp = kpad(K) / 32;
    std::vector<uint8_t> out((size_t)dim * scp, 127);
    for (int r = 0; r < dim; r++)
        std::memcpy(out.data() + (size_t)r * scp, scales + (size_t)r * sc, (size_t)sc);
    return out;
}

// Quantized MXFP6 matrix: dense-packed FP6 data + per-block E8M0 scales.
// Logical shape: [rows][cols], with one scale per 32-element block along cols.
struct QuantizedMatrix {
    std::vector<uint8_t> packed_data;  // [rows * fp6_packed_bytes(cols)]
    std::vector<uint8_t> scales;       // [rows * (cols/32)]
    int rows, cols;
    int packed_row_bytes;  // fp6_packed_bytes(cols)
    int scale_cols;        // cols / 32
};

// Quantize float matrix → MXFP6.
inline QuantizedMatrix quantize_to_mxfp6(const float* mat, int rows, int cols) {
    assert(cols % 32 == 0);

    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.packed_row_bytes = fp6_packed_bytes(cols);
    q.scale_cols = cols / 32;
    q.packed_data.resize(rows * q.packed_row_bytes, 0);
    q.scales.resize(rows * q.scale_cols);

    for (int r = 0; r < rows; r++) {
        for (int bg = 0; bg < q.scale_cols; bg++) {
            int k_start = bg * 32;

            float max_abs = 0.0f;
            for (int k = 0; k < 32; k++)
                max_abs = std::max(max_abs, std::abs(mat[r * cols + k_start + k]));

            uint8_t scale_code;
            if (max_abs == 0.0f) {
                scale_code = 127;
            } else {
                float needed = max_abs / 7.5f;
                int exp;
                frexpf(needed, &exp);
                float candidate = ldexpf(1.0f, exp - 1);
                if (candidate < needed) { /* use 2^exp */
                } else {
                    exp -= 1;
                }
                scale_code = (uint8_t)std::clamp(exp + 127, 0, 254);
            }

            q.scales[r * q.scale_cols + bg] = scale_code;
            float scale_val = e8m0_to_float(scale_code);

            uint8_t fp6_vals[32];
            for (int k = 0; k < 32; k++)
                fp6_vals[k] = float_to_fp6_e2m3(mat[r * cols + k_start + k] / scale_val);

            pack_fp6(fp6_vals, 32,
                     q.packed_data.data() + r * q.packed_row_bytes + fp6_packed_bytes(k_start));
        }
    }
    return q;
}

// Dequantize MXFP6 → float matrix.
inline void dequantize_mxfp6(const QuantizedMatrix& q, float* out) {
    for (int r = 0; r < q.rows; r++) {
        std::vector<uint8_t> fp6_vals(q.cols);
        unpack_fp6(q.packed_data.data() + r * q.packed_row_bytes, q.cols, fp6_vals.data());
        for (int c = 0; c < q.cols; c++) {
            float scale = e8m0_to_float(q.scales[r * q.scale_cols + c / 32]);
            out[r * q.cols + c] = fp6_e2m3_to_float(fp6_vals[c]) * scale;
        }
    }
}

// Preprocess B: transpose B[K][N] → B_T[N][K], then quantize.
inline QuantizedMatrix preprocess_B(const float* B, int K, int N) {
    assert(K % 32 == 0);
    std::vector<float> B_T(N * K);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) B_T[n * K + k] = B[k * N + n];
    return quantize_to_mxfp6(B_T.data(), N, K);
}

// Preprocess scales to MFMA 32x32x64 lane layout (ISA Page 65).
//
// Per tile (32 rows × K=64):
//   Lane  0-15:  row  0-15,  K= 0..31
//   Lane 16-31:  row 16-31,  K= 0..31
//   Lane 32-47:  row  0-15,  K=32..63
//   Lane 48-63:  row 16-31,  K=32..63
//
// Output: [num_tiles * k64_iters * 64] bytes in lane order.
struct PreprocessedScale {
    std::vector<uint8_t> data;
    int dim, K;
    int num_tiles;  // dim / 32
    int k64_iters;  // K / 64
};

inline PreprocessedScale preprocess_scale(const uint8_t* scales, int dim, int K) {
    assert(dim % 32 == 0 && K % 64 == 0);

    int scale_cols = K / 32;
    PreprocessedScale ps;
    ps.dim = dim;
    ps.K = K;
    ps.num_tiles = dim / 32;
    ps.k64_iters = K / 64;
    ps.data.resize(ps.num_tiles * ps.k64_iters * 64);

    for (int tile = 0; tile < ps.num_tiles; tile++) {
        int row_base = tile * 32;
        for (int k64 = 0; k64 < ps.k64_iters; k64++) {
            int kg0 = k64 * 2, kg1 = k64 * 2 + 1;
            uint8_t* out = ps.data.data() + (tile * ps.k64_iters + k64) * 64;

            for (int i = 0; i < 16; i++) {
                out[i] = scales[(row_base + i) * scale_cols + kg0];
                out[16 + i] = scales[(row_base + 16 + i) * scale_cols + kg0];
                out[32 + i] = scales[(row_base + i) * scale_cols + kg1];
                out[48 + i] = scales[(row_base + 16 + i) * scale_cols + kg1];
            }
        }
    }
    return ps;
}

// Pre-shuffle B for coalesced VMEM loads in the MFMA kernel.
//
// Input: B_q (from preprocess_B), shape B^T[N][K], row-major packed FP6.
//        For a single 32×64 tile: 32 rows × 48 bytes = 1536 bytes.
//
// Output: B_shuffled, 1536 bytes per tile, split into two sections:
//   Section 0 [0..1023]:    tid × 16 bytes = DWORDs 0-3 (for global_load_dwordx4)
//   Section 1 [1024..1535]: tid × 8  bytes = DWORDs 4-5 (for global_load_dwordx2)
//
// Thread tid maps to: n = tid%32 (N-column), khalf = tid/32 (K-half 0 or 1).
// Original data at: B_q.packed_data[n * packed_row_bytes + khalf * 24], 24 bytes.
struct PreshuffledB {
    std::vector<uint8_t> data;  // [n_tiles * k64_iters * 1536]
    int N, K;
    int n_tiles;    // N / 32
    int k64_iters;  // K / 64
};

inline PreshuffledB preshuffle_B(const QuantizedMatrix& B_q) {
    assert(B_q.rows % 32 == 0 && B_q.cols % 64 == 0);

    PreshuffledB pb;
    pb.N = B_q.rows;
    pb.K = B_q.cols;
    pb.n_tiles = B_q.rows / 32;
    pb.k64_iters = B_q.cols / 64;
    pb.data.resize(pb.n_tiles * pb.k64_iters * 1536);

    for (int nt = 0; nt < pb.n_tiles; nt++) {
        for (int ki = 0; ki < pb.k64_iters; ki++) {
            uint8_t* tile = pb.data.data() + (nt * pb.k64_iters + ki) * 1536;

            for (int tid = 0; tid < 64; tid++) {
                int n = nt * 32 + (tid % 32);
                int khalf = tid / 32;
                int k_byte_off = ki * 48 + khalf * 24;  // 48 = fp6_packed_bytes(64)
                const uint8_t* src = B_q.packed_data.data() + n * B_q.packed_row_bytes + k_byte_off;
                memcpy(tile + tid * 16, src, 16);            // section 0
                memcpy(tile + 1024 + tid * 8, src + 16, 8);  // section 1
            }
        }
    }
    return pb;
}

// Tile-grouped scale layout (host): a wave's `group` consecutive 32-blocks AND its `subs`
// k64 sub-slabs of one K-tile become contiguous per lane, so the kernel fetches a whole
// K-tile's scales in ONE dwordx{subs} load. Layout:
//   out[(((g*k_tiles+kt)*64 + lane)*subs + sub)*group_pad + j] = scale(block g*group+j, k64)
struct TiledScale {
    std::vector<uint8_t> data;
};
inline TiledScale tile_scale(const PreprocessedScale& ps, int group, int subs) {
    int group_pad = (group + 3) / 4 * 4;
    int k_tiles = ps.k64_iters / subs;
    int ng = ps.num_tiles / group;
    TiledScale ts;
    ts.data.assign((size_t)ng * k_tiles * 64 * subs * group_pad, 0);
    for (int g = 0; g < ng; g++)
        for (int kt = 0; kt < k_tiles; kt++)
            for (int lane = 0; lane < 64; lane++)
                for (int sub = 0; sub < subs; sub++)
                    for (int j = 0; j < group; j++) {
                        int k64 = kt * subs + sub, tile = g * group + j;
                        ts.data[(((size_t)(g * k_tiles + kt) * 64 + lane) * subs + sub) *
                                    group_pad +
                                j] = ps.data[(size_t)(tile * ps.k64_iters + k64) * 64 + lane];
                    }
    return ts;
}

}  // namespace mxfp6
