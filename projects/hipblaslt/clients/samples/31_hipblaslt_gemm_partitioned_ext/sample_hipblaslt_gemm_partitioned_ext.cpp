// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Runs one GEMM on the partitions of a partitioned GPU (for example DPX) with
// hipblaslt_experimental::PartitionedGemm, an experimental API: it is declared only when
// ROCM_ENABLE_EXPERIMENTAL_APIS is 1, and the sample checks both the revision it was written
// for (at compile time) and the revision of the loaded library (at run time).
//
//   - one hipBLASLt handle, stream, workspace and set of matrices per partition (device)
//   - the problem is the full m x n x k GEMM; it is split along N, so partition i computes
//     m x n_i x k and writes its columns of D at their final place in its own full-size D
//   - every matrix has one pointer per partition. By default each partition owns a whole copy.
//     With "shared", A and B are one buffer each on device 0 and every partition gets the same
//     pointer; the other partitions read them over peer access, which the caller enables. The
//     library adds the slice offset to each pointer.
//
// A and B hold ones, so every element of D must equal k. The sample checks that.
//
// Usage: sample_hipblaslt_gemm_partitioned_ext [partitions] [shared]    (default 2, replicated)
//
// TODO: add a variant with a shared A or B and storePeerD = true (every partition also stores
// its slice into the other partition's D). It needs OutputPeerD kernels in the device library;
// with the current library algoGetHeuristic returns no algorithm for it.

// A CMake target gets the switch by linking roc::hipblaslt_experimental, as this sample does;
// other builds define it before the include.
#ifndef ROCM_ENABLE_EXPERIMENTAL_APIS
#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#endif
#include <hipblaslt/hipblaslt-experimental.h>

#if !defined(HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM) \
    || HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM != 1
#error "This sample is written for revision 1 of hipBLASLt PARTITIONED_GEMM; see the CHANGELOG"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "helper.h"

int main(int argc, char** argv)
{
    // The header says revision 1; the loaded library must have the same one. A program with a
    // non-experimental path would fall back to it here instead of stopping.
    if(hipblasLtExperimentalGetFeatureRevision("PARTITIONED_GEMM") != 1)
    {
        std::cerr << "The loaded hipBLASLt does not have revision 1 of PARTITIONED_GEMM"
                  << std::endl;
        return EXIT_FAILURE;
    }

    int deviceCount = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
    const int  partitions = argc > 1 ? std::atoi(argv[1]) : 2;
    const bool shared     = argc > 2 && std::strcmp(argv[2], "shared") == 0;
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
    // One counter per partition, in host memory that every device can reach. A partition's stream
    // writes the number of the launch into its counter after its slice.
    std::vector<uint64_t*>         done(partitions);
    uint64_t                       launch = 0;
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
        // Created here, with the stream, so that nothing is allocated after run().
        CHECK_HIP_ERROR(hipHostMalloc(reinterpret_cast<void**>(&done[i]),
                                      sizeof(uint64_t),
                                      hipHostMallocCoherent | hipHostMallocPortable));
        *done[i] = 0;
        CHECK_HIP_ERROR(hipMalloc(&workspaces[i], workspaceBytes));
        void *dc, *dd;
        CHECK_HIP_ERROR(hipMalloc(&dc, ldc * n * 2));
        CHECK_HIP_ERROR(hipMalloc(&dd, ldd * n * 2));
        CHECK_HIP_ERROR(hipMemset(dc, 0, ldc * n * 2));
        CHECK_HIP_ERROR(hipMemset(dd, 0, ldd * n * 2));
        c[i] = dc, d[i] = dd;
        if(shared && i > 0)
        {
            // Shared A and B: the same pointer as partition 0. The library does not enable peer
            // access; this partition's device must be able to read device 0's memory.
            hipError_t peer = hipDeviceEnablePeerAccess(0, 0);
            if(peer != hipSuccess && peer != hipErrorPeerAccessAlreadyEnabled)
                CHECK_HIP_ERROR(peer);
            static_cast<void>(hipGetLastError());
            a[i] = a[0], b[i] = b[0];
            continue;
        }
        void *da, *db;
        CHECK_HIP_ERROR(hipMalloc(&da, hostA.size() * 2));
        CHECK_HIP_ERROR(hipMalloc(&db, hostB.size() * 2));
        CHECK_HIP_ERROR(hipMemcpy(da, hostA.data(), hostA.size() * 2, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hostB.data(), hostB.size() * 2, hipMemcpyHostToDevice));
        a[i] = da, b[i] = db;
    }
    // Shared inputs were written on device 0: they must be complete before another device reads.
    CHECK_HIP_ERROR(hipSetDevice(0));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // The join of stream j, built once: write its own counter, then wait for every other one.
    // After run() only the launch number is filled in.
    std::vector<std::vector<hipStreamBatchMemOpParams>> join(partitions);
    for(int j = 0; j < partitions; j++)
    {
        join[j].resize(partitions);
        join[j][0].writeValue.operation = hipStreamMemOpWriteValue64;
        join[j][0].writeValue.address   = done[j];
        int at                          = 1;
        for(int i = 0; i < partitions; i++)
        {
            if(i == j)
                continue;
            join[j][at].waitValue.operation = hipStreamMemOpWaitValue64;
            join[j][at].waitValue.address   = done[i];
            join[j][at].waitValue.flags     = hipStreamWaitValueGte;
            at++;
        }
    }

    // ---- the partitioned GEMM ----
    hipblaslt_experimental::PartitionedGemm gemm(handles,
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

    CHECK_HIPBLASLT_ERROR(gemm.setProblem(
        m, n, k, 1, lda, ldb, ldc, ldd, 0, 0, 0, 0, hipblaslt_experimental::SplitDim::N, inputs));

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

    // ---- join the partitions ----
    // run() does not order anything after the partitions. To queue work on a partition's stream
    // that needs the whole result, every stream writes the launch number into its own counter
    // after its slice, then waits until the other partitions' counters have reached it. After
    // this, work queued on any of the streams starts only when every partition has finished.
    // One hipStreamBatchMemOp per stream, built at setup; the write comes first, so no stream can
    // block another.
    // Both operations are recorded by stream capture, so the join also exists in a graph.
    ++launch;
    for(int j = 0; j < partitions; j++)
    {
        join[j][0].writeValue.value64 = launch;
        for(int i = 1; i < partitions; i++)
            join[j][i].waitValue.value64 = launch;
        CHECK_HIP_ERROR(hipStreamBatchMemOp(streams[j], partitions, join[j].data(), 0));
    }

    // The host reads D next, so it waits for the streams.
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
    std::cout << (shared ? "A and B: shared" : "A and B: replicated") << std::endl;
    std::cout << (errors ? "FAILED" : "PASSED") << std::endl;

    for(int i = 0; i < partitions; i++)
    {
        CHECK_HIP_ERROR(hipSetDevice(i));
        if(!shared || i == 0)
        {
            hipFree(const_cast<void*>(a[i]));
            hipFree(const_cast<void*>(b[i]));
        }
        hipFree(const_cast<void*>(c[i]));
        hipFree(d[i]);
        hipFree(workspaces[i]);
        hipHostFree(done[i]);
        hipStreamDestroy(streams[i]);
        hipblasLtDestroy(handles[i]);
    }
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
