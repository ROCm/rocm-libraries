// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 lds_transpose.cpp -o
//   ../<arch>/lds_transpose.s
//
// Transpose an MxN matrix using LDS.
// Input:  row-major MxN matrix in 'in'
// Output: column-major MxN matrix in 'out' (i.e. NxM row-major)
// blockDim must be >= (N, M). Threads with tx >= N or ty >= M are inactive.
#include <hip/hip_runtime.h>

__global__ void lds_transpose(int *out, const int *in, int M, int N) {
  extern __shared__ int tile[];
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  if (tx < N && ty < M) {
    // Load row-major into LDS
    tile[ty * N + tx] = in[ty * N + tx];
  }
  __syncthreads();

  if (tx < M && ty < N) {
    // Read transposed from LDS, write to output (NxM)
    out[ty * M + tx] = tile[tx * N + ty];
  }
}
