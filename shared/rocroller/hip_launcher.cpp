#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t e = cmd; \
        if (e != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(e) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    constexpr int N = 256;
    constexpr int blockSize = 64;

    std::vector<float> h_input(N), h_output(N);
    for (int i = 0; i < N; ++i) h_input[i] = static_cast<float>(i);

    float* d_input = nullptr;
    float* d_output = nullptr;

    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Load the assembly kernel module
    hipModule_t module;
    hipFunction_t kernel;
    HIP_CHECK(hipModuleLoad(&module, "./assembly_kernel.co"));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "load_lds_kernel"));

    // Set up kernel arguments
    struct {
        float* input;
        float* output;
    } args = { d_input, d_output };

    auto argsSize = sizeof(args);

    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &argsSize,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CHECK(hipModuleLaunchKernel(kernel,
        N / blockSize, 1, 1,   // gridDim
        blockSize,     1, 1,   // blockDim
        0, nullptr, config, nullptr)); // sharedMemBytes, stream, params, extra

    HIP_CHECK(hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        if (h_input[i] != h_output[i]) {
            std::cerr << "Mismatch at " << i << ": " << h_input[i] << " vs " << h_output[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Assembly LDS kernel test passed.\n";

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    return 0;
}
