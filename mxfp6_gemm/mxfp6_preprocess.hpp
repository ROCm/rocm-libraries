#pragma once
#include "mxfp6_types.hpp"
#include <vector>
#include <cmath>
#include <cassert>

namespace mxfp6 {

// Quantized MXFP6 matrix: dense-packed FP6 data + per-block E8M0 scales.
// Logical shape: [rows][cols], with one scale per 32-element block along cols.
struct QuantizedMatrix {
    std::vector<uint8_t> packed_data;  // [rows * fp6_packed_bytes(cols)]
    std::vector<uint8_t> scales;       // [rows * (cols/32)]
    int rows, cols;
    int packed_row_bytes;              // fp6_packed_bytes(cols)
    int scale_cols;                    // cols / 32
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
                if (candidate < needed) { /* use 2^exp */ } else { exp -= 1; }
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
        for (int n = 0; n < N; n++)
            B_T[n * K + k] = B[k * N + n];
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
    int num_tiles;   // dim / 32
    int k64_iters;   // K / 64
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
                out[i]      = scales[(row_base + i)      * scale_cols + kg0];
                out[16 + i] = scales[(row_base + 16 + i) * scale_cols + kg0];
                out[32 + i] = scales[(row_base + i)      * scale_cols + kg1];
                out[48 + i] = scales[(row_base + 16 + i) * scale_cols + kg1];
            }
        }
    }
    return ps;
}

} // namespace mxfp6
