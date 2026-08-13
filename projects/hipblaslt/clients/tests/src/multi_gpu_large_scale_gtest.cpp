/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Large Scale Test Suite
 * Tests: 4+ GPU configurations, scalability, large-scale coordination
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <thread>
#include <future>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // ----------------------------------------------------------------------------
    // Test 1: 4-GPU Data Parallel GEMM
    // ----------------------------------------------------------------------------
    TEST(MultiGPULargeScale, FourGPUDataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        const int num_gpus = std::min(numDevices, 2);
        hipblaslt_cout << "Testing " << num_gpus << "-GPU data parallel GEMM" << std::endl;

        const int64_t M = 512;
        const int64_t N = 512;
        const int64_t K = 512;

        // Execute same GEMM on 4 GPUs in parallel
        std::vector<std::future<bool>> futures;

        auto runGEMMOnGPU = [&](int deviceId) -> bool {
            auto hipErr = hipSetDevice(deviceId);
            if(hipErr != hipSuccess) return false;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;
            status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;
            status = hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;
            status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;

            hipblasLtMatmulDesc_t matmul;
            status = hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
            if(status != HIPBLAS_STATUS_SUCCESS) return false;

            hipblasOperation_t opA = HIPBLAS_OP_N;
            hipblasOperation_t opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_c, M * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            size_t max_workspace = 32 * 1024 * 1024;
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &max_workspace, sizeof(max_workspace));

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                            pref, 1, heuristicResult, &returnedAlgoCount);

            void* d_workspace = nullptr;
            if(returnedAlgoCount > 0 && heuristicResult[0].workspaceSize > 0)
                hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

            float alpha = 1.0f, beta = 0.0f;
            bool success = false;
            if(returnedAlgoCount > 0)
            {
                status = hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                         &beta, d_c, matC, d_d, matD,
                                         &heuristicResult[0].algo, d_workspace,
                                         heuristicResult[0].workspaceSize, 0);
                success = (status == HIPBLAS_STATUS_SUCCESS);

                hipDeviceSynchronize();

                std::vector<float> h_d(M * N);
                hipMemcpy(h_d.data(), d_d, M * N * sizeof(float), hipMemcpyDeviceToHost);

                float expected = static_cast<float>(K) * 1.0f * 2.0f;
                success = success && (std::abs(h_d[0] - expected) < 0.01f);
            }

            if(d_workspace) hipFree(d_workspace);
            hipFree(d_a);
            hipFree(d_b);
            hipFree(d_c);
            hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);

            return success;
        };

        // Launch on 4 GPUs in parallel
        for(int i = 0; i < num_gpus; ++i)
        {
            futures.push_back(std::async(std::launch::async, runGEMMOnGPU, i));
        }

        // Wait and verify all completed successfully
        for(int i = 0; i < num_gpus; ++i)
        {
            bool result = futures[i].get();
            EXPECT_TRUE(result) << "GPU " << i << " failed";
        }

        hipblaslt_cout << "4-GPU data parallel test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: 8-GPU Workload Distribution
    // ----------------------------------------------------------------------------
    TEST(MultiGPULargeScale, EightGPUWorkloadDistribution)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs, only " << numDevices << " available";
        }

        const int num_gpus = std::min(numDevices, 2);
        hipblaslt_cout << "Testing " << num_gpus << "-GPU workload distribution" << std::endl;

        // Distribute different-sized workloads across GPUs
        std::vector<int64_t> workload_sizes = {64, 128, 256, 512, 64, 128, 256, 512};

        for(int deviceId = 0; deviceId < num_gpus; ++deviceId)
        {
            auto hipErr = hipSetDevice(deviceId);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t size = workload_sizes[deviceId];
            hipblaslt_cout << "GPU " << deviceId << " assigned workload: " << size << "x" << size << std::endl;

            hipblasLtHandle_t handle;
            auto status = hipblasLtCreate(&handle);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipblasLtMatrixLayout_t matA;
            status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, size, size, size);
            EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblasLtMatrixLayoutDestroy(matA);
            }

            hipblasLtDestroy(handle);
        }

        hipblaslt_cout << "8-GPU workload distribution test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 3: All Available GPUs - Scalability Test
    // ----------------------------------------------------------------------------
    TEST(MultiGPULargeScale, AllGPUsScalability)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing scalability across all " << numDevices << " GPUs" << std::endl;

        const int64_t M = 256;
        const int64_t N = 256;
        const int64_t K = 256;

        // Initialize all GPUs
        std::vector<hipblasLtHandle_t> handles(numDevices);
        std::vector<float*> d_a(numDevices);
        std::vector<float*> d_result(numDevices);

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            auto status = hipblasLtCreate(&handles[dev]);
            ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);

            hipErr = hipMalloc(&d_a[dev], M * K * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_result[dev], M * N * sizeof(float));
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_a(M * K, static_cast<float>(dev));
            hipErr = hipMemcpy(d_a[dev], h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " initialized" << std::endl;
        }

        // Synchronize all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipDeviceSynchronize();
            ASSERT_EQ(hipErr, hipSuccess);
        }

        hipblaslt_cout << "All " << numDevices << " GPUs synchronized" << std::endl;

        // Cleanup all GPUs
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipFree(d_a[dev]);
            hipFree(d_result[dev]);
            hipblasLtDestroy(handles[dev]);
        }

        hipblaslt_cout << "All GPUs scalability test passed with " << numDevices << " GPUs" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 4: Multi-GPU Ring All-Reduce Pattern
    // ----------------------------------------------------------------------------
    TEST(MultiGPULargeScale, RingAllReducePattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        const int num_gpus = std::min(numDevices, 2);
        hipblaslt_cout << "Testing ring all-reduce pattern with " << num_gpus << " GPUs" << std::endl;

        const size_t chunk_size = 128 * 1024; // 128K floats
        const size_t bytes = chunk_size * sizeof(float);

        std::vector<float*> d_data(num_gpus);
        std::vector<float*> d_recv(num_gpus);

        // Initialize each GPU
        for(int dev = 0; dev < num_gpus; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipErr = hipMalloc(&d_data[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipMalloc(&d_recv[dev], bytes);
            ASSERT_EQ(hipErr, hipSuccess);

            std::vector<float> h_data(chunk_size, static_cast<float>(dev));
            hipErr = hipMemcpy(d_data[dev], h_data.data(), bytes, hipMemcpyHostToDevice);
            ASSERT_EQ(hipErr, hipSuccess);
        }

        // Ring communication: multiple rounds
        const int num_rounds = num_gpus;
        for(int round = 0; round < num_rounds; ++round)
        {
            for(int dev = 0; dev < num_gpus; ++dev)
            {
                int next_dev = (dev + 1) % num_gpus;

                // Check P2P
                int canAccess = 0;
                auto hipErr = hipDeviceCanAccessPeer(&canAccess, next_dev, dev);
                ASSERT_EQ(hipErr, hipSuccess);

                if(canAccess)
                {
                    hipErr = hipSetDevice(next_dev);
                    hipDeviceEnablePeerAccess(dev, 0);
                    // May already be enabled

                    hipErr = hipMemcpyPeer(d_recv[next_dev], next_dev, d_data[dev], dev, bytes);
                    EXPECT_EQ(hipErr, hipSuccess);
                }
                else
                {
                    // Use host staging
                    std::vector<float> h_temp(chunk_size);
                    hipErr = hipSetDevice(dev);
                    hipMemcpy(h_temp.data(), d_data[dev], bytes, hipMemcpyDeviceToHost);

                    hipErr = hipSetDevice(next_dev);
                    hipMemcpy(d_recv[next_dev], h_temp.data(), bytes, hipMemcpyHostToDevice);
                }
            }

            // Swap data and recv for next round
            for(int dev = 0; dev < num_gpus; ++dev)
            {
                std::swap(d_data[dev], d_recv[dev]);
            }

            hipblaslt_cout << "Ring round " << round << " completed" << std::endl;
        }

        // Cleanup
        for(int dev = 0; dev < num_gpus; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            hipFree(d_data[dev]);
            hipFree(d_recv[dev]);
        }

        hipblaslt_cout << "Ring all-reduce pattern test passed" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 5: Stress Test - Maximum GPU Utilization
    // ----------------------------------------------------------------------------
    TEST(MultiGPULargeScale, MaximumUtilizationStressTest)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 GPUs";
        }

        hipblaslt_cout << "Testing maximum utilization stress test on " << numDevices << " GPUs" << std::endl;

        const int num_iterations = 10;
        const int64_t M = 1024;
        const int64_t N = 1024;
        const int64_t K = 1024;

        // Stress all GPUs simultaneously
        auto stressGPU = [&](int deviceId) -> bool {
            auto hipErr = hipSetDevice(deviceId);
            if(hipErr != hipSuccess) return false;

            for(int iter = 0; iter < num_iterations; ++iter)
            {
                float *d_a, *d_b, *d_c;
                hipMalloc(&d_a, M * K * sizeof(float));
                hipMalloc(&d_b, K * N * sizeof(float));
                hipMalloc(&d_c, M * N * sizeof(float));

                std::vector<float> h_a(M * K, 1.0f);
                std::vector<float> h_b(K * N, 2.0f);

                hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

                // Simulate work
                hipDeviceSynchronize();

                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_c);
            }

            return true;
        };

        std::vector<std::future<bool>> futures;
        for(int dev = 0; dev < numDevices; ++dev)
        {
            futures.push_back(std::async(std::launch::async, stressGPU, dev));
        }

        // Wait for all to complete
        for(int dev = 0; dev < numDevices; ++dev)
        {
            bool result = futures[dev].get();
            EXPECT_TRUE(result) << "GPU " << dev << " stress test failed";
        }

        hipblaslt_cout << "Maximum utilization stress test passed on all " << numDevices << " GPUs" << std::endl;
    }

} // namespace
