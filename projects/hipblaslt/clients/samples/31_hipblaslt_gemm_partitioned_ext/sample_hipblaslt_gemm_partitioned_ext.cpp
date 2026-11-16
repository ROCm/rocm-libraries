// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Runs one GEMM on the partitions of a partitioned GPU (for example DPX) with
// hipblaslt_ext::PartitionedGemm.
//
//   - one hipBLASLt handle, stream, workspace and set of matrices per partition (device)
//   - the problem is the full m x n x k GEMM; it is split along N, so partition i computes
//     m x n_i x k and writes its columns of D at their final place in its own full-size D
//   - every matrix has one pointer per partition. Here each partition owns a whole copy; passing
//     the same pointer for every partition would share one matrix. The library adds the slice
//     offset to each pointer.
//
// A and B hold ones, so every element of D must equal k. The sample checks that.
//
// Usage: sample_hipblaslt_gemm_partitioned_ext [partitions]    (default 2)

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "helper.h"

int main(int argc, char** argv)
{
    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
    const int partitions = argc > 1 ? std::atoi(argv[1]) : 2;
    if(partitions < 1 || partitions > deviceCount)
    {
        std::cerr << "This sample needs " << partitions << " devices, found " << deviceCount
                  << ". On a partitioned GPU each partition is one device." << std::endl;
        return EXIT_SUCCESS; // not an error: there is nothing to run on this machine
    }

    // The full problem: D = A * B with A (m x k) and B (k x n), column-major, no transpose.
    const int64_t m = 1024, n = 2048, k = 512;
    const int64_t lda = m, ldb = k, ldc = m, ldd = m;
    const size_t  workspaceBytes = 32 * 1024 * 1024;
    float         alpha = 1.f, beta = 0.f;

    // ---- one handle, stream, workspace and set of matrices per partition ----
    std::vector<hipblasLtHandle_t> handles(partitions);
    std::vector<hipStream_t>       streams(partitions);
    std::vector<void*>             workspaces(partitions);
    std::vector<const void*>       a(partitions), b(partitions), c(partitions);
    std::vector<void*>             d(partitions);
    const uint16_t                 half_one = 0x3C00; // 1.0 in fp16
    std::vector<uint16_t>          hostA(lda * k, half_one), hostB(ldb * n, half_one);
    for(int i = 0; i < partitions; i++)
    {
        CHECK_HIP_ERROR(hipSetDevice(i)); // the handle belongs to the device that is current
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handles[i]));
        CHECK_HIP_ERROR(hipStreamCreate(&streams[i]));
        CHECK_HIP_ERROR(hipMalloc(&workspaces[i], workspaceBytes));
        void *da, *db, *dc, *dd;
        CHECK_HIP_ERROR(hipMalloc(&da, hostA.size() * 2));
        CHECK_HIP_ERROR(hipMalloc(&db, hostB.size() * 2));
        CHECK_HIP_ERROR(hipMalloc(&dc, ldc * n * 2));
        CHECK_HIP_ERROR(hipMalloc(&dd, ldd * n * 2));
        CHECK_HIP_ERROR(hipMemcpy(da, hostA.data(), hostA.size() * 2, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hostB.data(), hostB.size() * 2, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(dc, 0, ldc * n * 2));
        CHECK_HIP_ERROR(hipMemset(dd, 0, ldd * n * 2));
        a[i] = da, b[i] = db, c[i] = dc, d[i] = dd;
    }

    // ---- the partitioned GEMM ----
    hipblaslt_ext::PartitionedGemm gemm(handles,
                                        HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        HIP_R_16F,
                                        HIP_R_16F,
                                        HIP_R_16F,
                                        HIP_R_16F,
                                        HIPBLAS_COMPUTE_32F);

    // One GemmInputs per partition. A matrix that is shared has the same pointer in every
    // element; a replicated one has its own copy in each.
    std::vector<hipblaslt_ext::GemmInputs> inputs(partitions);
    for(int i = 0; i < partitions; i++)
    {
        inputs[i].setA(a[i]);
        inputs[i].setB(b[i]);
        inputs[i].setC(c[i]);
        inputs[i].setD(d[i]);
        inputs[i].setAlpha(&alpha);
        inputs[i].setBeta(&beta);
    }

    CHECK_HIPBLASLT_ERROR(gemm.setProblem(m,
                                          n,
                                          k,
                                          1,
                                          lda,
                                          ldb,
                                          ldc,
                                          ldd,
                                          0,
                                          0,
                                          0,
                                          0,
                                          hipblaslt_ext::SplitDim::N,
                                          inputs));

    // Only algorithms that every partition accepts for its slice come back.
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(workspaceBytes);
    gemm.setMaxWorkspaceBytes(workspaceBytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(1, pref, heuristic));
    if(heuristic.empty())
    {
        std::cerr << "No algorithm accepted by every partition!" << std::endl;
        return EXIT_FAILURE;
    }

    // initialize again whenever the algorithm or a pointer changes (call setProblem first
    // to change a pointer)
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristic[0].algo, workspaces, true, streams));

    // Launches slice i on streams[i] and returns. The streams run at the same time.
    CHECK_HIPBLASLT_ERROR(gemm.run(streams));
    for(int i = 0; i < partitions; i++)
    {
        CHECK_HIP_ERROR(hipSetDevice(i));
        CHECK_HIP_ERROR(hipStreamSynchronize(streams[i]));
    }

    // ---- check: every column of the slice of partition i must hold k ----
    int errors = 0;
    for(int i = 0; i < partitions; i++)
    {
        int64_t offset = 0, size = 0;
        CHECK_HIPBLASLT_ERROR(gemm.getSlice(i, offset, size));
        std::vector<uint16_t> out(ldd * n);
        CHECK_HIP_ERROR(hipSetDevice(i));
        CHECK_HIP_ERROR(hipMemcpy(out.data(), d[i], out.size() * 2, hipMemcpyDeviceToHost));
        const uint16_t half_k = 0x6000; // 512.0 in fp16
        for(int64_t col = offset; col < offset + size; col++)
            for(int64_t row = 0; row < m; row++)
                if(out[row + col * ldd] != half_k && ++errors < 5)
                    std::cerr << "partition " << i << " D(" << row << "," << col << ") = 0x"
                              << std::hex << out[row + col * ldd] << std::dec << ", expected " << k
                              << std::endl;
        std::cout << "partition " << i << " computed columns [" << offset << ", " << offset + size
                  << ")" << std::endl;
    }
    std::cout << (errors ? "FAILED" : "PASSED") << std::endl;

    for(int i = 0; i < partitions; i++)
    {
        CHECK_HIP_ERROR(hipSetDevice(i));
        hipFree(const_cast<void*>(a[i]));
        hipFree(const_cast<void*>(b[i]));
        hipFree(const_cast<void*>(c[i]));
        hipFree(d[i]);
        hipFree(workspaces[i]);
        hipStreamDestroy(streams[i]);
        hipblasLtDestroy(handles[i]);
    }
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
