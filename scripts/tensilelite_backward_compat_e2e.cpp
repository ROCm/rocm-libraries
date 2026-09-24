// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{
constexpr int64_t m = 512;
constexpr int64_t n = 512;
constexpr int64_t k = 512;
constexpr size_t  workspace_bytes = 32 * 1024 * 1024;

bool check_hip(hipError_t status, const char* expression, int line)
{
    if(status == hipSuccess)
        return true;

    std::cerr << "HIP failure at line " << line << ": " << expression << ": "
              << hipGetErrorString(status) << " (" << static_cast<int>(status) << ")\n";
    return false;
}

bool check_hipblaslt(hipblasStatus_t status, const char* expression, int line)
{
    if(status == HIPBLAS_STATUS_SUCCESS)
        return true;

    std::cerr << "hipBLASLt failure at line " << line << ": " << expression << ": status "
              << static_cast<int>(status) << "\n";
    return false;
}

#define CHECK_HIP(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if(!check_hip((call), #call, __LINE__))                                                    \
            return 20;                                                                             \
    } while(false)

#define CHECK_HIPBLASLT(call)                                                                      \
    do                                                                                             \
    {                                                                                              \
        if(!check_hipblaslt((call), #call, __LINE__))                                              \
            return 30;                                                                             \
    } while(false)

std::string loaded_hipblaslt_path()
{
    void* symbol = dlsym(RTLD_DEFAULT, "hipblasLtCreate");
    Dl_info info{};
    if(symbol == nullptr || dladdr(symbol, &info) == 0 || info.dli_fname == nullptr)
        return "<dladdr-failed>";

    std::error_code ec;
    auto            canonical = std::filesystem::canonical(info.dli_fname, ec);
    return ec ? std::string(info.dli_fname) : canonical.string();
}
} // namespace

int main()
{
    std::cout << "hipblaslt_library=" << loaded_hipblaslt_path() << "\n";

    int device_count = 0;
    auto device_status = hipGetDeviceCount(&device_count);
    if(device_status != hipSuccess || device_count == 0)
    {
        std::cerr << "gpu_unavailable=" << hipGetErrorString(device_status)
                  << " device_count=" << device_count << "\n";
        return 10;
    }

    hipDeviceProp_t properties{};
    CHECK_HIP(hipGetDeviceProperties(&properties, 0));
    std::cout << "gpu_arch=" << properties.gcnArchName << " device_count=" << device_count << "\n";

    hipblasLtHandle_t handle{};
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));

    int version = 0;
    CHECK_HIPBLASLT(hipblasLtGetVersion(handle, &version));
    std::cout << "hipblaslt_version_integer=" << version << "\n";

    std::vector<float> host_a(k * m);
    std::vector<float> host_b(k * n);
    std::vector<float> host_c(m * n);
    std::vector<float> host_d(m * n);
    std::vector<float> reference(m * n);

    for(int64_t col = 0; col < m; ++col)
        for(int64_t row = 0; row < k; ++row)
            host_a[row + col * k] = static_cast<float>((row * 3 + col * 5) % 17 - 8) / 16.0f;

    for(int64_t col = 0; col < n; ++col)
        for(int64_t row = 0; row < k; ++row)
            host_b[row + col * k] = static_cast<float>((row * 7 + col * 2) % 19 - 9) / 16.0f;

    for(int64_t col = 0; col < n; ++col)
        for(int64_t row = 0; row < m; ++row)
            host_c[row + col * m] = static_cast<float>((row + col * 11) % 13 - 6) / 16.0f;

    constexpr float alpha = 1.0f;
    constexpr float beta  = 1.0f;
    for(int64_t col = 0; col < n; ++col)
    {
        for(int64_t row = 0; row < m; ++row)
        {
            double sum = 0.0;
            for(int64_t inner = 0; inner < k; ++inner)
                sum += static_cast<double>(host_a[inner + row * k])
                       * static_cast<double>(host_b[inner + col * k]);
            reference[row + col * m]
                = alpha * static_cast<float>(sum) + beta * host_c[row + col * m];
        }
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr, *device_d = nullptr;
    void* device_workspace = nullptr;
    CHECK_HIP(hipMalloc(&device_a, host_a.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_b, host_b.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_c, host_c.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_d, host_d.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&device_workspace, workspace_bytes));

    hipStream_t stream{};
    CHECK_HIP(hipStreamCreate(&stream));
    CHECK_HIP(hipMemcpyAsync(device_a,
                             host_a.data(),
                             host_a.size() * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));
    CHECK_HIP(hipMemcpyAsync(device_b,
                             host_b.data(),
                             host_b.size() * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));
    CHECK_HIP(hipMemcpyAsync(device_c,
                             host_c.data(),
                             host_c.size() * sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));

    hipblasLtMatrixLayout_t layout_a{}, layout_b{}, layout_c{}, layout_d{};
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_32F, k, m, k));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_32F, k, n, k));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_32F, m, n, m));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layout_d, HIP_R_32F, m, n, m));

    hipblasLtMatmulDesc_t operation{};
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t transpose_a = HIPBLAS_OP_T;
    hipblasOperation_t transpose_b = HIPBLAS_OP_N;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operation, HIPBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)));

    hipblasLtMatmulPreference_t preference{};
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_bytes,
                                                          sizeof(workspace_bytes)));

    hipblasLtMatmulHeuristicResult_t heuristic{};
    int                              returned_algorithms = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    operation,
                                                    layout_a,
                                                    layout_b,
                                                    layout_c,
                                                    layout_d,
                                                    preference,
                                                    1,
                                                    &heuristic,
                                                    &returned_algorithms));
    if(returned_algorithms != 1 || heuristic.state != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "no_valid_solution returned_algorithms=" << returned_algorithms
                  << " heuristic_state=" << static_cast<int>(heuristic.state) << "\n";
        return 40;
    }
    if(heuristic.workspaceSize > workspace_bytes)
    {
        std::cerr << "workspace_too_large requested=" << heuristic.workspaceSize
                  << " available=" << workspace_bytes << "\n";
        return 41;
    }

    CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                    operation,
                                    &alpha,
                                    device_a,
                                    layout_a,
                                    device_b,
                                    layout_b,
                                    &beta,
                                    device_c,
                                    layout_c,
                                    device_d,
                                    layout_d,
                                    &heuristic.algo,
                                    device_workspace,
                                    heuristic.workspaceSize,
                                    stream));
    CHECK_HIP(hipMemcpyAsync(host_d.data(),
                             device_d,
                             host_d.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    size_t incorrect_values   = 0;
    for(size_t index = 0; index < host_d.size(); ++index)
    {
        double absolute_error
            = std::abs(static_cast<double>(host_d[index]) - static_cast<double>(reference[index]));
        double tolerance = 1.0e-3 + 2.0e-4 * std::abs(static_cast<double>(reference[index]));
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error
            = std::max(max_relative_error,
                       absolute_error / std::max(1.0, std::abs(static_cast<double>(reference[index]))));
        if(!std::isfinite(host_d[index]) || absolute_error > tolerance)
            ++incorrect_values;
    }

    std::cout << "problem=m512_n512_batch1_k512_transA_T_transB_N_f32\n"
              << "workspace_used=" << heuristic.workspaceSize << "\n"
              << "incorrect_values=" << incorrect_values << " total_values=" << host_d.size()
              << " max_absolute_error=" << max_absolute_error
              << " max_relative_error=" << max_relative_error << "\n";

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatmulDescDestroy(operation);
    hipblasLtMatrixLayoutDestroy(layout_a);
    hipblasLtMatrixLayoutDestroy(layout_b);
    hipblasLtMatrixLayoutDestroy(layout_c);
    hipblasLtMatrixLayoutDestroy(layout_d);
    static_cast<void>(hipStreamDestroy(stream));
    static_cast<void>(hipFree(device_workspace));
    static_cast<void>(hipFree(device_d));
    static_cast<void>(hipFree(device_c));
    static_cast<void>(hipFree(device_b));
    static_cast<void>(hipFree(device_a));
    hipblasLtDestroy(handle);

    if(incorrect_values != 0)
    {
        std::cerr << "FAIL numerical_validation\n";
        return 50;
    }

    std::cout << "PASS numerical_validation\n";
    return 0;
}
