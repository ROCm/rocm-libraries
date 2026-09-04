// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocblas/rocblas.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    setenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", REAL_PROVIDER_PATH, 1);
    setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", REAL_ROCBLAS_PATH, 1);
    constexpr int thread_count = 16;
    constexpr int iterations = 100;
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&] {
            for (int iteration = 0; iteration < iterations; ++iteration) {
                size_t size = 0;
                if (rocblas_get_version_string_size(&size) != rocblas_status_success || !size)
                    failed = true;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (failed) {
        std::cerr << "real provider failed under concurrent initialization/use\n";
        return EXIT_FAILURE;
    }
    std::cout << "real provider concurrency passed\n";
    return EXIT_SUCCESS;
}
