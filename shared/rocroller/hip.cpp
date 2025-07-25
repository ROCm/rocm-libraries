#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Error-checking macro
#define HIP_CHECK(cmd) \
    { \
        hipError_t e = cmd; \
        if (e != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(e) << " (" << e << ")\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    }

#define N 256
#define BLOCK_SIZE 64

__global__ void loadToLDS(const float* __restrict__ input, float* __restrict__ output) {
    __shared__ float lds[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    if (tid < BLOCK_SIZE) {
        lds[tid] = input[global_id];
    }

    __syncthreads();

    if (tid < BLOCK_SIZE) {
        output[global_id] = lds[tid];
    }
}

int main() {
    float* h_input = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(i);

    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE);
    hipLaunchKernelGGL(loadToLDS, blocks, threads, 0, 0, d_input, d_output);

    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        if (h_input[i] != h_output[i]) {
            std::cerr << "Mismatch at " << i << ": " << h_input[i]
                      << " vs " << h_output[i] << std::endl;
            return 1;
        }
    }

    std::cout << "LDS load test passed.\n";

    delete[] h_input;
    delete[] h_output;
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    return 0;
}
