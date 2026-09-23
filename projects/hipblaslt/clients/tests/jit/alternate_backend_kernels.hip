// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

// The two independently compiled symbols deliberately use no library metadata.
extern "C" __global__ void
    alternate_initialize(const float* C, float* workspace, int m, int n, int ldc, float beta)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < m * n)
        workspace[i] = beta * C[(i % m) + (i / m) * ldc];
}

extern "C" __global__ void alternate_gemm(const float* A,
                                          const float* B,
                                          const float* workspace,
                                          float*       D,
                                          int          m,
                                          int          n,
                                          int          k,
                                          int          lda,
                                          int          ldb,
                                          int          ldd,
                                          float        alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= m * n)
        return;
    const int row = i % m;
    const int col = i / m;
    float     sum = 0;
    for(int p = 0; p < k; ++p)
        sum += A[row + p * lda] * B[p + col * ldb];
    D[row + col * ldd] = alpha * sum + workspace[i];
}
