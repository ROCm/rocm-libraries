// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Single-GPU bring-up arms for the A2A-GEMM shard loop (FusedA2AMode=1),
// dispatched from main.cpp when the a2a-prefill option is set. Allocates and
// validates its own tensors rather than using DataInitialization /
// ReferenceValidator: B is W separate [nToken, k_local] segments
// (ldb = k_local), W times larger than B's TensorDescriptor accounts for.
//
// Two arms share that setup:
//   prefill  -- the host lays all W gathered segments out in B up front.
//   loopback -- the host fills segment 0 only and points every peer group at
//               this device; the kernel's own SDMA packets deliver segments
//               1..W-1. Enabled with a2a-loopback.
//
// The loopback arm gives peer[s] a distinct source slice, never one shared
// slice for every peer.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes_BFloat16.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "FusedA2ACounterSentinel.hpp"
#include "ProgramOptions.hpp"
#include "SolutionIterator.hpp"

// SdmaQueue.hpp is header-only and pulls in hsakmt, which is only on the
// build path when TENSILELITE_ENABLE_SDMA is set.
#ifdef TENSILELITE_ENABLE_SDMA
#include "SdmaQueue.hpp"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
            // Multiples of 0.25 in [-1, 1]: exact in bf16.
            inline float a2aPrefillValue(uint64_t seed)
            {
                uint64_t h = seed * 0x9E3779B97F4A7C15ull;
                h ^= h >> 29;
                h *= 0xBF58476D1CE4E5B9ull;
                h ^= h >> 32;
                return (float)(int)(h % 9u) * 0.25f - 1.0f;
            }

            struct DeviceBuffer
            {
                void* ptr = nullptr;

                ~DeviceBuffer()
                {
                    if(ptr)
                        (void)hipFree(ptr);
                }
                hipError_t allocate(size_t bytes)
                {
                    return hipMalloc(&ptr, bytes);
                }
                // For buffers an SDMA engine writes while the kernel reads them.
                hipError_t allocateFineGrained(size_t bytes)
                {
                    return hipExtMallocWithFlags(&ptr, bytes, hipDeviceMallocFinegrained);
                }
            };

            // A: [M, K] contiguous at lda. B: W segments of [nToken, k_local],
            // each filled from its own value stream.
            void a2aFillHost(size_t                 M,
                             size_t                 N,
                             size_t                 K,
                             size_t                 lda,
                             size_t                 kLoc,
                             int                    W,
                             std::vector<BFloat16>& hostA,
                             std::vector<BFloat16>& hostB)
            {
#pragma omp parallel for
                for(size_t m = 0; m < M; m++)
                    for(size_t k = 0; k < K; k++)
                        hostA[m * lda + k] = BFloat16(a2aPrefillValue(m * K + k));

#pragma omp parallel for
                for(size_t r = 0; r < (size_t)W; r++)
                    for(size_t n = 0; n < N; n++)
                        for(size_t kl = 0; kl < kLoc; kl++)
                            hostB[r * N * kLoc + n * kLoc + kl]
                                = BFloat16(a2aPrefillValue(0x5A2Aull + r * N * kLoc + n * kLoc + kl));
            }

            // Every D element against a shard-aware reference: shard r pairs
            // A[m, r*k_local ..] with B segment r.
            size_t a2aCheckD(size_t                       M,
                             size_t                       N,
                             size_t                       lda,
                             size_t                       ldd,
                             size_t                       kLoc,
                             int                          W,
                             std::vector<BFloat16> const& hostA,
                             std::vector<BFloat16> const& hostB,
                             std::vector<BFloat16> const& hostD,
                             double&                      worstRel)
            {
                size_t mismatches = 0;
                worstRel          = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : mismatches) reduction(max : worstRel)
                for(size_t m = 0; m < M; m++)
                {
                    for(size_t n = 0; n < N; n++)
                    {
                        double acc = 0.0;
                        for(size_t r = 0; r < (size_t)W; r++)
                        {
                            BFloat16 const* aRow = &hostA[m * lda + r * kLoc];
                            BFloat16 const* bRow = &hostB[r * N * kLoc + n * kLoc];
                            for(size_t kl = 0; kl < kLoc; kl++)
                                acc += (double)(float)aRow[kl] * (double)(float)bRow[kl];
                        }
                        const double got = (double)(float)hostD[m + n * ldd];
                        const double rel = std::abs(got - acc) / std::max(1.0, std::abs(acc));
                        if(rel > worstRel)
                            worstRel = rel;
                        if(rel > 1e-2)
                            mismatches++;
                    }
                }
                return mismatches;
            }

            int runA2APrefillForWorld(hip::SolutionAdapter&                     adapter,
                                      std::shared_ptr<SolutionIterator>         solutionIterator,
                                      std::shared_ptr<Hardware>                 hardware,
                                      ContractionProblemGemm const&             base,
                                      int                                       W)
            {
                const size_t M     = base.freeSizeA(0);
                const size_t N     = base.freeSizeB(0);
                const size_t K     = base.boundSize(0);
                const size_t lda   = base.a().strides()[1];
                const size_t ldd   = base.d().strides()[1];
                const size_t kLoc  = K / (size_t)W;
                const auto   dtype = base.a().dataType();

                std::cout << "[a2a-prefill] W=" << W << " M=" << M << " N=" << N << " K=" << K
                          << " k_local=" << kLoc << std::endl;

                ContractionProblemGemm problem = base;
                problem.resetTensor(ContractionProblemGemm::TENSOR::B,
                                    dtype,
                                    {K, N, 1},
                                    {1, kLoc, N * kLoc});
                problem.setFusedA2AWorld((uint32_t)W);

                solutionIterator->preProblem(&problem);
                if(!solutionIterator->moreSolutionsInProblem())
                {
                    std::cerr << "[a2a-prefill] ERROR: no solution accepts the problem at W=" << W
                              << std::endl;
                    return 1;
                }
                std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
                if(!solution)
                {
                    std::cerr << "[a2a-prefill] ERROR: getSolution returned null" << std::endl;
                    return 1;
                }
                std::cout << "[a2a-prefill] solution: " << solution->name() << std::endl;

                const size_t aElems = M * K;
                const size_t bElems = (size_t)W * N * kLoc;
                const size_t cdElems = M * N;

                std::vector<BFloat16> hostA(aElems);
                std::vector<BFloat16> hostB(bElems);
                std::vector<BFloat16> hostD(cdElems);

                a2aFillHost(M, N, K, lda, kLoc, W, hostA, hostB);

                DeviceBuffer devA, devB, devC, devD;
                HIP_CHECK_EXC(devA.allocate(aElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devB.allocate(bElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devC.allocate(cdElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devD.allocate(cdElems * sizeof(BFloat16)));

                HIP_CHECK_EXC(hipMemcpy(devA.ptr,
                                        hostA.data(),
                                        aElems * sizeof(BFloat16),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemcpy(devB.ptr,
                                        hostB.data(),
                                        bElems * sizeof(BFloat16),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemset(devC.ptr, 0, cdElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(hipMemset(devD.ptr, 0, cdElems * sizeof(BFloat16)));

                ContractionInputs inputs;
                inputs.a     = devA.ptr;
                inputs.b     = devB.ptr;
                inputs.c     = devC.ptr;
                inputs.d     = devD.ptr;
                inputs.alpha = 1.0f;
                inputs.beta  = 0.0f;

                if(solution->requiredWorkspaceSize(problem, *hardware) != 0)
                {
                    std::cerr << "[a2a-prefill] ERROR: solution wants a workspace" << std::endl;
                    return 1;
                }

                hipStream_t stream = nullptr;
                HIP_CHECK_EXC(hipStreamCreate(&stream));
                auto kernels = solution->solve(problem, inputs, *hardware, nullptr, 0, stream);
                HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
                HIP_CHECK_EXC(hipStreamDestroy(stream));

                HIP_CHECK_EXC(hipMemcpy(hostD.data(),
                                        devD.ptr,
                                        cdElems * sizeof(BFloat16),
                                        hipMemcpyDeviceToHost));

                double       worstRel = 0.0;
                const size_t mismatches
                    = a2aCheckD(M, N, lda, ldd, kLoc, W, hostA, hostB, hostD, worstRel);

                std::cout << "[a2a-prefill] W=" << W << " mismatches=" << mismatches
                          << " worst-rel=" << worstRel << std::endl;
                return mismatches == 0 ? 0 : 1;
            }

#ifdef TENSILELITE_ENABLE_SDMA
            // Byte pattern written over gathered segments 1..W-1 before each
            // launch. Outside the value set a2aPrefillValue draws from.
            constexpr int A2A_LOOPBACK_POISON = 0xC7;

            int runA2ALoopbackForWorld(hip::SolutionAdapter&             adapter,
                                       std::shared_ptr<SolutionIterator> solutionIterator,
                                       std::shared_ptr<Hardware>         hardware,
                                       ContractionProblemGemm const&     base,
                                       int                               W,
                                       int                               launches)
            {
                const size_t M     = base.freeSizeA(0);
                const size_t N     = base.freeSizeB(0);
                const size_t K     = base.boundSize(0);
                const size_t lda   = base.a().strides()[1];
                const size_t ldd   = base.d().strides()[1];
                const size_t kLoc  = K / (size_t)W;
                const auto   dtype = base.a().dataType();

                std::cout << "[a2a-loopback] W=" << W << " M=" << M << " N=" << N << " K=" << K
                          << " k_local=" << kLoc << " launches=" << launches << std::endl;

                ContractionProblemGemm problem = base;
                problem.resetTensor(ContractionProblemGemm::TENSOR::B,
                                    dtype,
                                    {K, N, 1},
                                    {1, kLoc, N * kLoc});
                problem.setFusedA2AWorld((uint32_t)W);

                solutionIterator->preProblem(&problem);
                if(!solutionIterator->moreSolutionsInProblem())
                {
                    std::cerr << "[a2a-loopback] ERROR: no solution accepts the problem at W=" << W
                              << std::endl;
                    return 1;
                }
                std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
                if(!solution)
                {
                    std::cerr << "[a2a-loopback] ERROR: getSolution returned null" << std::endl;
                    return 1;
                }
                std::cout << "[a2a-loopback] solution: " << solution->name() << std::endl;

                if(solution->requiredWorkspaceSize(problem, *hardware) != 0)
                {
                    std::cerr << "[a2a-loopback] ERROR: solution wants a workspace" << std::endl;
                    return 1;
                }

                auto const* amd = dynamic_cast<AMDGPU const*>(hardware.get());
                if(!amd || amd->computeUnitCount == 0)
                {
                    std::cerr << "[a2a-loopback] ERROR: no CU count to size the counter region"
                              << std::endl;
                    return 1;
                }
                const size_t numCu = amd->computeUnitCount;

                // The kernel's iB spans [0, batches): a2aBatchSpan computes it as
                // (b * F) / numCu over b in [0, tokenBlocks).
                const size_t mt0         = solution->sizeMapping.macroTile.x;
                const size_t mt1         = solution->sizeMapping.macroTile.y;
                const size_t F           = (M + mt0 - 1) / mt0;
                const size_t tokenBlocks = (N + mt1 - 1) / mt1;
                const size_t batches     = (tokenBlocks - 1) * F / numCu + 1;

                const size_t flagBytes
                    = fusedA2AMode1FlagBytes((uint32_t)W, (uint32_t)tokenBlocks);
                const size_t payloadBytes = fusedA2AMode1PayloadBytes(
                    (uint32_t)W, (uint32_t)tokenBlocks, (uint32_t)batches);
                const size_t allocBytes = fusedA2AMode1AllocBytes(
                    (uint32_t)W, (uint32_t)tokenBlocks, (uint32_t)batches);

                std::cout << "[a2a-loopback] tokenBlocks=" << tokenBlocks << " F=" << F
                          << " numCu=" << numCu << " batches=" << batches
                          << " flagBytes=" << flagBytes << " counterAlloc=" << allocBytes
                          << std::endl;

                const size_t aElems   = M * K;
                const size_t bElems   = (size_t)W * N * kLoc;
                const size_t cdElems  = M * N;
                const size_t bBytes   = bElems * sizeof(BFloat16);
                const size_t segBytes = N * kLoc * sizeof(BFloat16);

                std::vector<BFloat16> hostA(aElems), hostB(bElems), hostD(cdElems), gotB(bElems);
                a2aFillHost(M, N, K, lda, kLoc, W, hostA, hostB);

                DeviceBuffer devA, devB, devC, devD, devX, devCounter;
                HIP_CHECK_EXC(devA.allocate(aElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devB.allocateFineGrained(bBytes));
                HIP_CHECK_EXC(devC.allocate(cdElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devD.allocate(cdElems * sizeof(BFloat16)));
                HIP_CHECK_EXC(devX.allocate(bBytes));
                HIP_CHECK_EXC(devCounter.allocateFineGrained(allocBytes));

                HIP_CHECK_EXC(hipMemcpy(
                    devA.ptr, hostA.data(), aElems * sizeof(BFloat16), hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemcpy(devX.ptr, hostB.data(), bBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemset(devC.ptr, 0, cdElems * sizeof(BFloat16)));
                // Once, over the whole payload. The per-launch reset below covers
                // the flag region only.
                HIP_CHECK_EXC(hipMemset(devCounter.ptr, 0, payloadBytes));
                std::vector<uint32_t> guard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                fusedA2ACounterSentinelFill(guard.data());
                HIP_CHECK_EXC(hipMemcpy((char*)devCounter.ptr + payloadBytes,
                                        guard.data(),
                                        FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                        hipMemcpyHostToDevice));

                const uint32_t                          node = sdmaNodeIdForDevice(0);
                std::vector<std::unique_ptr<SdmaQueue>> queues;
                for(int j = 0; j < W; j++)
                    queues.push_back(
                        std::make_unique<SdmaQueue>(node, sdmaSelectEngine(node, node)));

                ContractionInputs inputs;
                inputs.a               = devA.ptr;
                inputs.b               = devB.ptr;
                inputs.c               = devC.ptr;
                inputs.d               = devD.ptr;
                inputs.alpha           = 1.0f;
                inputs.beta            = 0.0f;
                inputs.fusedA2ACounter = devCounter.ptr;
                inputs.fusedA2AMyRank  = 0;
                inputs.fusedA2ADrain   = 0;
                // peer[s].x rides in the mode-0 recvPtr slot. At myRank 0 the
                // queue that fills gathered segment s reads slice s of X.
                for(int s = 0; s < W; s++)
                {
                    const HsaQueueResource& r = queues[s]->queueResource();
                    inputs.fusedA2APeers.push_back(
                        {nullptr,
                         (void*)((char*)devX.ptr + (size_t)s * segBytes),
                         queues[s]->ringBase(),
                         (void*)r.Queue_read_ptr_aql,
                         (void*)r.Queue_write_ptr_aql,
                         (void*)r.Queue_DoorBell_aql});
                }

                hipStream_t stream = nullptr;
                HIP_CHECK_EXC(hipStreamCreate(&stream));

                {
                    auto probe = solution->solve(problem, inputs, *hardware, nullptr, 0, stream);
                    if(probe.size() != 1 || probe[0].numWorkGroups.x != F
                       || probe[0].numWorkGroups.y != tokenBlocks)
                    {
                        std::cerr << "[a2a-loopback] ERROR: launch geometry is "
                                  << probe.size() << " kernel(s) of "
                                  << (probe.empty() ? 0 : probe[0].numWorkGroups.x) << "x"
                                  << (probe.empty() ? 0 : probe[0].numWorkGroups.y)
                                  << " work-groups, but the counter region was sized for 1 kernel of "
                                  << F << "x" << tokenBlocks << std::endl;
                        (void)hipStreamDestroy(stream);
                        return 1;
                    }
                }

                int rc = 0;
                for(int it = 0; it < launches; it++)
                {
                    HIP_CHECK_EXC(
                        hipMemsetAsync(devD.ptr, 0, cdElems * sizeof(BFloat16), stream));
                    HIP_CHECK_EXC(hipMemcpyAsync(
                        devB.ptr, devX.ptr, segBytes, hipMemcpyDeviceToDevice, stream));
                    HIP_CHECK_EXC(hipMemsetAsync((char*)devB.ptr + segBytes,
                                                 A2A_LOOPBACK_POISON,
                                                 bBytes - segBytes,
                                                 stream));
                    HIP_CHECK_EXC(hipMemsetAsync((char*)devCounter.ptr
                                                     + FUSED_A2A_MODE1_FLAG_OFFSET,
                                                 0,
                                                 flagBytes,
                                                 stream));

                    auto kernels = solution->solve(problem, inputs, *hardware, nullptr, 0, stream);
                    HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
                    HIP_CHECK_EXC(hipStreamSynchronize(stream));

                    HIP_CHECK_EXC(
                        hipMemcpy(gotB.data(), devB.ptr, bBytes, hipMemcpyDeviceToHost));
                    HIP_CHECK_EXC(hipMemcpy(hostD.data(),
                                            devD.ptr,
                                            cdElems * sizeof(BFloat16),
                                            hipMemcpyDeviceToHost));

                    int    badSeg = -1;
                    size_t badIdx = 0;
                    for(size_t r = 0; r < (size_t)W && badSeg < 0; r++)
                    {
                        BFloat16 const* got  = gotB.data() + r * N * kLoc;
                        BFloat16 const* want = hostB.data() + r * N * kLoc;
                        for(size_t i = 0; i < N * kLoc; i++)
                        {
                            if(std::memcmp(&got[i], &want[i], sizeof(BFloat16)) != 0)
                            {
                                badSeg = (int)r;
                                badIdx = i;
                                break;
                            }
                        }
                    }

                    double       worstRel = 0.0;
                    const size_t mismatches
                        = a2aCheckD(M, N, lda, ldd, kLoc, W, hostA, hostB, hostD, worstRel);

                    std::vector<uint32_t> gotGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                    HIP_CHECK_EXC(hipMemcpy(gotGuard.data(),
                                            (const char*)devCounter.ptr + payloadBytes,
                                            FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                            hipMemcpyDeviceToHost));
                    const int badGuard = fusedA2ACounterSentinelFirstBad(gotGuard.data());

                    std::cout << "[a2a-loopback] W=" << W << " launch=" << it
                              << " gathered=" << (badSeg < 0 ? "exact" : "DIFFERS")
                              << " mismatches=" << mismatches << " worst-rel=" << worstRel
                              << " guard=" << (badGuard < 0 ? "intact" : "CORRUPT") << std::endl;
                    if(badSeg >= 0)
                    {
                        std::cerr << "[a2a-loopback] ERROR: gathered segment " << badSeg
                                  << " first differs at element " << badIdx << std::endl;
                    }
                    if(badGuard >= 0)
                    {
                        std::cerr << "[a2a-loopback] ERROR: counter guard word " << badGuard
                                  << " overwritten; counter[iB] ran past " << payloadBytes
                                  << " bytes" << std::endl;
                    }
                    if(badSeg >= 0 || mismatches != 0 || badGuard >= 0)
                        rc = 1;
                }

                HIP_CHECK_EXC(hipStreamDestroy(stream));
                return rc;
            }
#endif
        } // namespace

        int runA2APrefill(po::variables_map const&                                       args,
                          std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
                          std::shared_ptr<Hardware>                                      hardware,
                          hip::SolutionAdapter&                                          adapter,
                          ContractionProblem*                                            problemIn)
        {
            auto* base = dynamic_cast<ContractionProblemGemm*>(problemIn);
            if(!base)
            {
                std::cerr << "[a2a-prefill] ERROR: problem is not a plain GEMM" << std::endl;
                return 1;
            }

            const auto dtype = base->a().dataType();
            if(dtype != rocisa::DataType::BFloat16 || base->b().dataType() != dtype
               || base->d().dataType() != dtype)
            {
                std::cerr << "[a2a-prefill] ERROR: this arm only handles bf16 A/B/D" << std::endl;
                return 1;
            }
            if(base->batchSize(0) != 1)
            {
                std::cerr << "[a2a-prefill] ERROR: batch count must be 1, got "
                          << base->batchSize(0) << std::endl;
                return 1;
            }
            if(base->a().strides()[1] != base->boundSize(0))
            {
                std::cerr << "[a2a-prefill] ERROR: A must be one contiguous [M, K]; lda="
                          << base->a().strides()[1] << " K=" << base->boundSize(0) << std::endl;
                return 1;
            }

            std::vector<int> worlds = args["a2a-prefill-worlds"].as<std::vector<int>>();
            if(worlds.empty())
                worlds = {1, 4};

            const bool loopback = args["a2a-loopback"].as<bool>();
            const int  launches = std::max(1, args["a2a-loopback-launches"].as<int>());
#ifndef TENSILELITE_ENABLE_SDMA
            if(loopback)
            {
                std::cerr << "[a2a-loopback] ERROR: this client was built without "
                             "TENSILELITE_ENABLE_SDMA; the loopback arm has no queues to hand "
                             "the kernel. Rebuild with -DTENSILELITE_ENABLE_SDMA=ON."
                          << std::endl;
                return 1;
            }
#endif

            auto solutionIterator = SolutionIterator::Default(library, hardware, args);

            int rc = 0;
            for(int W : worlds)
            {
                if(W < 1 || base->boundSize(0) % (size_t)W != 0)
                {
                    std::cerr << "[a2a-prefill] ERROR: K=" << base->boundSize(0)
                              << " is not a multiple of W=" << W << std::endl;
                    return 1;
                }
                int one = runA2APrefillForWorld(adapter, solutionIterator, hardware, *base, W);
                if(one != 0)
                    rc = one;
#ifdef TENSILELITE_ENABLE_SDMA
                if(loopback)
                {
                    if(!fusedA2AWorldSizeValid(W))
                    {
                        std::cerr << "[a2a-loopback] ERROR: W=" << W
                                  << " has no peer group; the segment holds "
                                  << FUSED_A2A_MAX_RANKS << std::endl;
                        return 1;
                    }
                    one = runA2ALoopbackForWorld(
                        adapter, solutionIterator, hardware, *base, W, launches);
                    if(one != 0)
                        rc = one;
                }
#endif
            }

            std::cout << "[a2a-prefill] overall " << (rc == 0 ? "PASSED" : "FAILED") << std::endl;
            return rc;
        }

    } // namespace Client
} // namespace TensileLite
