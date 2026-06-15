#pragma once
#include <cassert>
#include <vector>

#include "mxfp6/preprocess.hpp"
#include "mxfp6/types.hpp"

namespace mxfp6 {

// CPU reference GEMM: D[M][N] = dequant(A_q) × dequant(B_q)^T
//
// A_q: quantized A[M][K]
// B_q: quantized B^T[N][K]  (from preprocess_B)
inline void mxfp6_gemm_ref(const QuantizedMatrix& A_q, const QuantizedMatrix& B_q, float* D, int M,
                           int K, int N) {
    assert(A_q.rows == M && A_q.cols == K);
    assert(B_q.rows == N && B_q.cols == K);

    std::vector<float> A_deq(M * K), B_deq(N * K);
    dequantize_mxfp6(A_q, A_deq.data());
    dequantize_mxfp6(B_q, B_deq.data());

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A_deq[m * K + k] * B_deq[n * K + k];
            D[m * N + n] = acc;
        }
}

}  // namespace mxfp6
