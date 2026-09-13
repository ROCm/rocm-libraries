// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    try {
        setenv("ROCM_INTERFACES_BLAS_V2_PROVIDER", REAL_NARROW_PROVIDER_PATH, 1);
        setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", NARROW_BACKEND_PATH, 1);
        rocblas_handle handle = nullptr;
        require(rocblas_create_handle(&handle) == rocblas_status_success && handle,
                "real narrow provider did not create a backend handle");

        float alpha = 2.0f;
        std::array<float, 3> x{1.0f, 2.0f, 3.0f};
        std::array<float, 3> y{4.0f, 5.0f, 6.0f};
        require(
            rocblas_saxpy(handle, 3, &alpha, x.data(), 1, y.data(), 1) == rocblas_status_success,
            "AXPY did not cross the narrow vector-transform provider");
        require((y == std::array<float, 3>{6.0f, 9.0f, 12.0f}), "AXPY result mismatch");

        require(rocblas_sscal(handle, 3, &alpha, x.data(), 1) == rocblas_status_success,
                "SCAL did not cross the narrow vector-transform provider");
        require((x == std::array<float, 3>{2.0f, 4.0f, 6.0f}), "SCAL result mismatch");

        std::array<float, 3> copied{};
        require(rocblas_scopy(handle, 3, x.data(), 1, copied.data(), 1) == rocblas_status_success,
                "COPY did not cross the narrow vector-transform provider");
        require(copied == x, "COPY result mismatch");

        require(rocblas_sswap(handle, 3, x.data(), 1, y.data(), 1) == rocblas_status_success,
                "SWAP did not cross the narrow vector-transform provider");
        require((x == std::array<float, 3>{6.0f, 9.0f, 12.0f}), "SWAP x result mismatch");
        require((y == std::array<float, 3>{2.0f, 4.0f, 6.0f}), "SWAP y result mismatch");

        std::array<float, 3> x64{1.0f, 2.0f, 3.0f};
        std::array<float, 3> y64{4.0f, 5.0f, 6.0f};
        require(rocblas_saxpy_64(handle, 3, &alpha, x64.data(), 1, y64.data(), 1) ==
                    rocblas_status_success,
                "64-bit AXPY did not cross the narrow vector-transform provider");
        require(rocblas_sscal_64(handle, 3, &alpha, x64.data(), 1) == rocblas_status_success,
                "64-bit SCAL did not cross the narrow vector-transform provider");
        std::array<float, 3> copied64{};
        require(rocblas_scopy_64(handle, 3, x64.data(), 1, copied64.data(), 1) ==
                    rocblas_status_success,
                "64-bit COPY did not cross the narrow vector-transform provider");
        require(rocblas_sswap_64(handle, 3, x64.data(), 1, y64.data(), 1) == rocblas_status_success,
                "64-bit SWAP did not cross the narrow vector-transform provider");
        require((x64 == std::array<float, 3>{6.0f, 9.0f, 12.0f}), "64-bit SWAP x result mismatch");
        require((y64 == std::array<float, 3>{2.0f, 4.0f, 6.0f}), "64-bit SWAP y result mismatch");
        require(copied64 == y64, "64-bit COPY result mismatch");

        require(rocblas_destroy_handle(handle) == rocblas_status_success,
                "real narrow provider did not destroy the backend handle");
        std::cout << "real narrow-v2 vector-transform migration passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
