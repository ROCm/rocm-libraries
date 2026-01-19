// Compile with :
//  hipcc --genco --offload-arch=gfx942 -O3 poc_co.cpp -o poc_co.co
// Above assumes MI300X.

#include <hip/hip_runtime.h>

extern "C" __global__ void SimpleGemm(float*       D,
                                      const float* A,
                                      const float* B,
                                      const float* C,
                                      float        alpha,
                                      float        beta,
                                      int          m,
                                      int          n,
                                      int          k)
{
    // Launch a 1D grid covering the output matrix C (M x N)
    int tx  = threadIdx.x;
    int bx  = blockIdx.x;
    int idx = bx * blockDim.x + tx;

    // Boundary check
    if(idx >= m * n)
        return;

    int row = idx % m;
    int col = idx / m;

    float sum = 0.0f;
    for(int i = 0; i < k; ++i)
    {
        float a_val = A[row + i * m];
        float b_val = B[i + col * k];
        sum += a_val * b_val;
    }

    // 4. Write Result
    float c_val = C[idx];
    D[idx]      = alpha * sum + beta * c_val;
}
