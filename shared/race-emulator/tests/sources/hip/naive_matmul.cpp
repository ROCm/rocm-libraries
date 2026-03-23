// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 naive_matmul.cpp -o
//   ../<arch>/naive_matmul.s
//
// Naive matrix multiply: C = A * B + C (f32)
// Each thread computes one element of C.
// Thread (tx, ty) computes C[ty][tx] = sum_k(A[ty][k] * B[k][tx]) + C[ty][tx]
// blockDim = (N, M), one workgroup computes the full result.
#include <hip/hip_runtime.h>

__global__ void naive_matmul(float *C, const float *A, const float *B, int M,
                             int N, int K) {
  int tx = threadIdx.x; // column index
  int ty = threadIdx.y; // row index
  if (tx < N && ty < M) {
    float sum = C[ty * N + tx];
    for (int k = 0; k < K; ++k) {
      sum += A[ty * K + k] * B[k * N + tx];
    }
    C[ty * N + tx] = sum;
  }
}
