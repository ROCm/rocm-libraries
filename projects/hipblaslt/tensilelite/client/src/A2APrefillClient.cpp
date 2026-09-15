// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Bring-up arms for the A2A-GEMM shard loop (FusedA2AMode=1), dispatched from
// main.cpp when a2a-prefill or a2a-multigpu is set. Allocate and validate their
// own tensors rather than using DataInitialization / ReferenceValidator: B is W
// separate [nToken, k_local] segments (ldb = k_local), W times larger than B's
// TensorDescriptor accounts for.
//
// Three arms share that setup:
//   prefill  -- one device; the host lays all W gathered segments out in B up
//               front.
//   loopback -- one device; the host fills segment 0 only and points every peer
//               group at this device; the kernel's own SDMA packets deliver
//               segments 1..W-1. Enabled with a2a-loopback.
//   multigpu -- W devices in one process, rank d on device d, peer groups
//               pointing at the other devices' x. Enabled with a2a-multigpu.
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

            // A: [M, K] contiguous at lda.
            void a2aFillA(size_t M, size_t K, size_t lda, std::vector<BFloat16>& hostA)
            {
#pragma omp parallel for
                for(size_t m = 0; m < M; m++)
                    for(size_t k = 0; k < K; k++)
                        hostA[m * lda + k] = BFloat16(a2aPrefillValue(m * K + k));
            }

            // B: W segments of [nToken, k_local], each filled from its own value
            // stream.
            void a2aFillHost(size_t                 M,
                             size_t                 N,
                             size_t                 K,
                             size_t                 lda,
                             size_t                 kLoc,
                             int                    W,
                             std::vector<BFloat16>& hostA,
                             std::vector<BFloat16>& hostB)
            {
                a2aFillA(M, K, lda, hostA);

#pragma omp parallel for
                for(size_t r = 0; r < (size_t)W; r++)
                    for(size_t n = 0; n < N; n++)
                        for(size_t kl = 0; kl < kLoc; kl++)
                            hostB[r * N * kLoc + n * kLoc + kl]
                                = BFloat16(a2aPrefillValue(0x5A2Aull + r * N * kLoc + n * kLoc + kl));
            }

            // Every D element against a shard-aware reference: shard r pairs
            // A[m, r*k_local ..] with B segment r. segStride is the element gap
            // between consecutive segments of hostB; rowBase picks which token
            // block of each segment this rank owns.
            size_t a2aCheckD(size_t                       M,
                             size_t                       N,
                             size_t                       lda,
                             size_t                       ldd,
                             size_t                       kLoc,
                             int                          W,
                             size_t                       segStride,
                             size_t                       rowBase,
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
                            BFloat16 const* bRow = &hostB[r * segStride + (rowBase + n) * kLoc];
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

            struct A2ACounterLayout
            {
                size_t F            = 0;
                size_t tokenBlocks  = 0;
                size_t batches      = 0;
                size_t flagBytes    = 0;
                size_t payloadBytes = 0;
                size_t allocBytes   = 0;
            };

            // Sizes the mode-1 counter region, then checks that derivation against
            // the grid solve() reports. probeInputs is read for pointer values the
            // invocation records; nothing is launched.
            bool a2aCounterLayout(ContractionSolution&          solution,
                                  Hardware const&               hardware,
                                  ContractionProblemGemm const& problem,
                                  ContractionInputs const&      probeInputs,
                                  size_t                        M,
                                  size_t                        N,
                                  int                           W,
                                  char const*                   tag,
                                  A2ACounterLayout&             out)
            {
                auto const* amd = dynamic_cast<AMDGPU const*>(&hardware);
                if(!amd || amd->computeUnitCount == 0)
                {
                    std::cerr << tag << " ERROR: no CU count to size the counter region"
                              << std::endl;
                    return false;
                }
                if(problem.transposeC01())
                {
                    std::cerr << tag
                              << " ERROR: transposeC01 swaps the two free sizes before the tile "
                                 "divide; this arm's counter layout assumes it is off"
                              << std::endl;
                    return false;
                }
                const size_t numCu = amd->computeUnitCount;
                const size_t mt0   = solution.sizeMapping.macroTile.x;
                const size_t mt1   = solution.sizeMapping.macroTile.y;

                // a2aBatchSpan computes iB as (b * F) / numCu over b in [0, tokenBlocks).
                out.F            = (M + mt0 - 1) / mt0;
                out.tokenBlocks  = (N + mt1 - 1) / mt1;
                out.batches      = (out.tokenBlocks - 1) * out.F / numCu + 1;
                out.flagBytes    = fusedA2AMode1FlagBytes((uint32_t)W, (uint32_t)out.tokenBlocks);
                out.payloadBytes = fusedA2AMode1PayloadBytes(
                    (uint32_t)W, (uint32_t)out.tokenBlocks, (uint32_t)out.batches);
                out.allocBytes = fusedA2AMode1AllocBytes(
                    (uint32_t)W, (uint32_t)out.tokenBlocks, (uint32_t)out.batches);

                // ContractionSolution folds y and z into x unless clusters are on.
                // The comparison is on the work-group total, not the per-axis counts.
                auto         probe    = solution.solve(problem, probeInputs, hardware, nullptr, 0, nullptr);
                const size_t launched = probe.size() != 1
                                            ? 0
                                            : (size_t)probe[0].numWorkGroups.x
                                                  * probe[0].numWorkGroups.y
                                                  * probe[0].numWorkGroups.z;
                if(launched != out.F * out.tokenBlocks)
                {
                    std::cerr << tag << " ERROR: " << probe.size() << " kernel(s) launching "
                              << launched << " work-groups; the counter region was sized for 1 "
                              << "kernel of " << out.F << "*" << out.tokenBlocks << "="
                              << out.F * out.tokenBlocks << std::endl;
                    return false;
                }
                std::cout << tag << " tokenBlocks=" << out.tokenBlocks << " F=" << out.F
                          << " numCu=" << numCu << " batches=" << out.batches
                          << " flagBytes=" << out.flagBytes << " counterAlloc=" << out.allocBytes
                          << std::endl;
                return true;
            }

            // Zeroes the whole payload and arms the guard tail past it.
            hipError_t a2aAllocCounter(DeviceBuffer& buf, A2ACounterLayout const& layout)
            {
                hipError_t err = buf.allocateFineGrained(layout.allocBytes);
                if(err != hipSuccess)
                    return err;
                err = hipMemset(buf.ptr, 0, layout.payloadBytes);
                if(err != hipSuccess)
                    return err;
                std::vector<uint32_t> guard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                fusedA2ACounterSentinelFill(guard.data());
                return hipMemcpy((char*)buf.ptr + layout.payloadBytes,
                                 guard.data(),
                                 FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                 hipMemcpyHostToDevice);
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

                // At W>=2 the mode-1 prologue reaches the peer queue pointers, which
                // this arm leaves null. The loopback arm is the W>=2 path.
                if(W >= 2)
                {
                    std::cout << "[a2a-prefill] W=" << W
                              << " skipped: this arm has no peer queues; run it with "
                                 "--a2a-loopback for W>=2"
                              << std::endl;
                    return 0;
                }

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

                // The election runs at every W and increments counter[iB].
                A2ACounterLayout layout;
                if(!a2aCounterLayout(
                       *solution, *hardware, problem, inputs, M, N, W, "[a2a-prefill]", layout))
                    return 1;
                DeviceBuffer devCounter;
                HIP_CHECK_EXC(a2aAllocCounter(devCounter, layout));
                inputs.fusedA2ACounter = devCounter.ptr;

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
                const size_t mismatches = a2aCheckD(
                    M, N, lda, ldd, kLoc, W, N * kLoc, 0, hostA, hostB, hostD, worstRel);

                std::cout << "[a2a-prefill] W=" << W << " mismatches=" << mismatches
                          << " worst-rel=" << worstRel << std::endl;
                return mismatches == 0 ? 0 : 1;
            }

#ifdef TENSILELITE_ENABLE_SDMA
            // Byte pattern written over gathered segments 1..W-1 before each
            // launch. Outside the value set a2aPrefillValue draws from.
            constexpr int A2A_LOOPBACK_POISON = 0xC7;

            // First gathered segment differing from its host-side source, or -1.
            // want[r] is the source for segment r.
            int a2aFirstBadSegment(BFloat16 const*                     got,
                                   std::vector<BFloat16 const*> const& want,
                                   size_t                              segElems,
                                   size_t&                             badIdx)
            {
                for(size_t r = 0; r < want.size(); r++)
                {
                    BFloat16 const* seg = got + r * segElems;
                    for(size_t i = 0; i < segElems; i++)
                    {
                        if(std::memcmp(&seg[i], &want[r][i], sizeof(BFloat16)) != 0)
                        {
                            badIdx = i;
                            return (int)r;
                        }
                    }
                }
                return -1;
            }

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

                HIP_CHECK_EXC(hipMemcpy(
                    devA.ptr, hostA.data(), aElems * sizeof(BFloat16), hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemcpy(devX.ptr, hostB.data(), bBytes, hipMemcpyHostToDevice));
                HIP_CHECK_EXC(hipMemset(devC.ptr, 0, cdElems * sizeof(BFloat16)));

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

                A2ACounterLayout layout;
                if(!a2aCounterLayout(
                       *solution, *hardware, problem, inputs, M, N, W, "[a2a-loopback]", layout))
                    return 1;
                HIP_CHECK_EXC(a2aAllocCounter(devCounter, layout));
                inputs.fusedA2ACounter = devCounter.ptr;

                hipStream_t stream = nullptr;
                HIP_CHECK_EXC(hipStreamCreate(&stream));

                std::vector<BFloat16 const*> wantSeg(W);
                for(int s = 0; s < W; s++)
                    wantSeg[s] = hostB.data() + (size_t)s * N * kLoc;

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
                                                 layout.flagBytes,
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

                    size_t    badIdx = 0;
                    const int badSeg
                        = a2aFirstBadSegment(gotB.data(), wantSeg, N * kLoc, badIdx);

                    double       worstRel = 0.0;
                    const size_t mismatches = a2aCheckD(
                        M, N, lda, ldd, kLoc, W, N * kLoc, 0, hostA, hostB, hostD, worstRel);

                    std::vector<uint32_t> gotGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                    HIP_CHECK_EXC(hipMemcpy(gotGuard.data(),
                                            (const char*)devCounter.ptr + layout.payloadBytes,
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
                                  << " overwritten; counter[iB] ran past " << layout.payloadBytes
                                  << " bytes" << std::endl;
                    }
                    if(badSeg >= 0 || mismatches != 0 || badGuard >= 0)
                        rc = 1;
                }

                HIP_CHECK_EXC(hipStreamDestroy(stream));
                return rc;
            }

            // One process holding W devices, one rank each. peer[j].x is rank j's
            // whole x allocation, not pre-offset: the kernel adds the
            // myRank*nToken row offset itself. Peer group index is the device
            // ordinal, unpermuted.
            int runA2AMultiGpuForWorld(po::variables_map const&          args,
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

                int deviceCount = 0;
                HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));
                if(deviceCount < W)
                {
                    std::cerr << "[a2a-multigpu] ERROR: W=" << W << " needs " << W
                              << " devices, found " << deviceCount << std::endl;
                    return 1;
                }

                std::cout << "[a2a-multigpu] W=" << W << " M=" << M << " N=" << N << " K=" << K
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
                    std::cerr << "[a2a-multigpu] ERROR: no solution accepts the problem at W=" << W
                              << std::endl;
                    return 1;
                }
                std::shared_ptr<ContractionSolution> solution = solutionIterator->getSolution();
                if(!solution)
                {
                    std::cerr << "[a2a-multigpu] ERROR: getSolution returned null" << std::endl;
                    return 1;
                }
                std::cout << "[a2a-multigpu] solution: " << solution->name() << std::endl;

                if(solution->requiredWorkspaceSize(problem, *hardware) != 0)
                {
                    std::cerr << "[a2a-multigpu] ERROR: solution wants a workspace" << std::endl;
                    return 1;
                }

                const size_t aElems   = M * K;
                const size_t segElems = N * kLoc;
                const size_t xElems   = (size_t)W * segElems;
                const size_t cdElems  = M * N;
                const size_t segBytes = segElems * sizeof(BFloat16);
                const size_t xBytes   = xElems * sizeof(BFloat16);

                // All W ranks' x laid out back to back; a2aCheckD walks them with
                // segStride = xElems.
                std::vector<BFloat16> hostA(aElems), hostX((size_t)W * xElems);
                std::vector<BFloat16> hostD(cdElems), gotB(xElems);
                a2aFillA(M, K, lda, hostA);
#pragma omp parallel for
                for(size_t i = 0; i < (size_t)W * xElems; i++)
                    hostX[i] = BFloat16(a2aPrefillValue(0x5A2Aull + i));

                std::vector<DeviceBuffer> devA(W), devB(W), devC(W), devD(W), devX(W),
                    devCounter(W);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(devA[d].allocate(aElems * sizeof(BFloat16)));
                    HIP_CHECK_EXC(devB[d].allocateFineGrained(xBytes));
                    HIP_CHECK_EXC(devC[d].allocate(cdElems * sizeof(BFloat16)));
                    HIP_CHECK_EXC(devD[d].allocate(cdElems * sizeof(BFloat16)));
                    // Read by remote SDMA engines.
                    HIP_CHECK_EXC(devX[d].allocateFineGrained(xBytes));
                    HIP_CHECK_EXC(hipMemcpy(devA[d].ptr,
                                            hostA.data(),
                                            aElems * sizeof(BFloat16),
                                            hipMemcpyHostToDevice));
                    HIP_CHECK_EXC(hipMemcpy(devX[d].ptr,
                                            hostX.data() + (size_t)d * xElems,
                                            xBytes,
                                            hipMemcpyHostToDevice));
                    HIP_CHECK_EXC(hipMemset(devC[d].ptr, 0, cdElems * sizeof(BFloat16)));
                }

                for(int s = 0; s < W; s++)
                {
                    HIP_CHECK_EXC(hipSetDevice(s));
                    for(int t = 0; t < W; t++)
                    {
                        if(t == s)
                            continue;
                        int canAccess = 0;
                        HIP_CHECK_EXC(hipDeviceCanAccessPeer(&canAccess, s, t));
                        if(!canAccess)
                        {
                            std::cerr << "[a2a-multigpu] ERROR: device " << s
                                      << " cannot P2P device " << t << std::endl;
                            return 1;
                        }
                        hipError_t pe = hipDeviceEnablePeerAccess(t, 0);
                        if(pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                            HIP_CHECK_EXC(pe);
                    }
                }

                // Created after P2P enable. The self entry (j == d) is never
                // enqueued.
                std::vector<std::vector<std::unique_ptr<SdmaQueue>>> queues(W);
                {
                    std::vector<uint32_t> nodes(W);
                    for(int j = 0; j < W; j++)
                        nodes[j] = sdmaNodeIdForDevice(j);
                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        for(int j = 0; j < W; j++)
                            queues[d].push_back(std::make_unique<SdmaQueue>(
                                nodes[d], sdmaSelectEngine(nodes[d], nodes[j])));
                    }
                }

                // One adapter per device; the caller's binds its modules to one
                // device only.
                auto        filename = args["library-file"].as<std::string>();
                size_t      dirPos   = filename.rfind('/');
                std::string libraryDirectory
                    = (dirPos != std::string::npos) ? filename.substr(0, dirPos + 1)
                                                    : std::string(".");
                auto const& codeObjectFiles = args["code-object"].as<std::vector<std::string>>();

                std::vector<std::shared_ptr<hip::SolutionAdapter>> adapters(W);
                std::vector<hipStream_t>                           streams(W, nullptr);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipStreamCreate(&streams[d]));
                    adapters[d] = std::make_shared<hip::SolutionAdapter>();
                    for(auto const& co : codeObjectFiles)
                        (void)adapters[d]->loadCodeObjectFile(co);
                    (void)adapters[d]->initializeLazyLoading(hardware->archName(),
                                                             libraryDirectory);
                }

                std::vector<ContractionInputs> inputs(W);
                for(int d = 0; d < W; d++)
                {
                    inputs[d].a              = devA[d].ptr;
                    inputs[d].b              = devB[d].ptr;
                    inputs[d].c              = devC[d].ptr;
                    inputs[d].d              = devD[d].ptr;
                    inputs[d].alpha          = 1.0f;
                    inputs[d].beta           = 0.0f;
                    inputs[d].fusedA2AMyRank = (uint32_t)d;
                    inputs[d].fusedA2ADrain  = 0;
                    for(int j = 0; j < W; j++)
                    {
                        const HsaQueueResource& r = queues[d][j]->queueResource();
                        inputs[d].fusedA2APeers.push_back({nullptr,
                                                           devX[j].ptr,
                                                           queues[d][j]->ringBase(),
                                                           (void*)r.Queue_read_ptr_aql,
                                                           (void*)r.Queue_write_ptr_aql,
                                                           (void*)r.Queue_DoorBell_aql});
                    }
                }

                A2ACounterLayout layout;
                if(!a2aCounterLayout(
                       *solution, *hardware, problem, inputs[0], M, N, W, "[a2a-multigpu]", layout))
                    return 1;
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(a2aAllocCounter(devCounter[d], layout));
                    inputs[d].fusedA2ACounter = devCounter[d].ptr;
                }

                std::vector<std::vector<KernelInvocation>> perDeviceKernels(W);
                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    perDeviceKernels[d]
                        = solution->solve(problem, inputs[d], *hardware, nullptr, 0, streams[d]);
                }

                // Rank d's gathered segment j carries rank (d+j) mod W's x, rows
                // [d*nToken, +nToken).
                std::vector<std::vector<BFloat16 const*>> wantSeg(
                    W, std::vector<BFloat16 const*>(W));
                for(int d = 0; d < W; d++)
                    for(int j = 0; j < W; j++)
                        wantSeg[d][j] = hostX.data() + (size_t)((d + j) % W) * xElems
                                        + (size_t)d * segElems;

                int rc = 0;
                for(int it = 0; it < launches; it++)
                {
                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(hipMemsetAsync(
                            devD[d].ptr, 0, cdElems * sizeof(BFloat16), streams[d]));
                        HIP_CHECK_EXC(hipMemcpyAsync(devB[d].ptr,
                                                     (char*)devX[d].ptr + (size_t)d * segBytes,
                                                     segBytes,
                                                     hipMemcpyDeviceToDevice,
                                                     streams[d]));
                        HIP_CHECK_EXC(hipMemsetAsync((char*)devB[d].ptr + segBytes,
                                                     A2A_LOOPBACK_POISON,
                                                     xBytes - segBytes,
                                                     streams[d]));
                        HIP_CHECK_EXC(hipMemsetAsync((char*)devCounter[d].ptr
                                                         + FUSED_A2A_MODE1_FLAG_OFFSET,
                                                     0,
                                                     layout.flagBytes,
                                                     streams[d]));
                    }

                    // Stands in for the boundary barrier.
                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(hipDeviceSynchronize());
                    }

                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(adapters[d]->launchKernels(
                            perDeviceKernels[d], streams[d], nullptr, nullptr));
                    }
                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(hipStreamSynchronize(streams[d]));
                    }

                    for(int d = 0; d < W; d++)
                    {
                        HIP_CHECK_EXC(hipSetDevice(d));
                        HIP_CHECK_EXC(
                            hipMemcpy(gotB.data(), devB[d].ptr, xBytes, hipMemcpyDeviceToHost));
                        HIP_CHECK_EXC(hipMemcpy(hostD.data(),
                                                devD[d].ptr,
                                                cdElems * sizeof(BFloat16),
                                                hipMemcpyDeviceToHost));

                        size_t    badIdx = 0;
                        const int badSeg
                            = a2aFirstBadSegment(gotB.data(), wantSeg[d], segElems, badIdx);

                        double       worstRel  = 0.0;
                        const size_t mismatches = a2aCheckD(M,
                                                            N,
                                                            lda,
                                                            ldd,
                                                            kLoc,
                                                            W,
                                                            xElems,
                                                            (size_t)d * N,
                                                            hostA,
                                                            hostX,
                                                            hostD,
                                                            worstRel);

                        std::vector<uint32_t> gotGuard(FUSED_A2A_COUNTER_SENTINEL_WORDS);
                        HIP_CHECK_EXC(hipMemcpy(gotGuard.data(),
                                                (const char*)devCounter[d].ptr
                                                    + layout.payloadBytes,
                                                FUSED_A2A_COUNTER_SENTINEL_BYTES,
                                                hipMemcpyDeviceToHost));
                        const int badGuard = fusedA2ACounterSentinelFirstBad(gotGuard.data());

                        std::cout << "[a2a-multigpu] W=" << W << " launch=" << it << " dev=" << d
                                  << " gathered=" << (badSeg < 0 ? "exact" : "DIFFERS")
                                  << " mismatches=" << mismatches << " worst-rel=" << worstRel
                                  << " guard=" << (badGuard < 0 ? "intact" : "CORRUPT")
                                  << std::endl;
                        if(badSeg >= 0)
                        {
                            std::cerr << "[a2a-multigpu] ERROR: device " << d
                                      << " gathered segment " << badSeg
                                      << " first differs at element " << badIdx << std::endl;
                        }
                        if(badGuard >= 0)
                        {
                            std::cerr << "[a2a-multigpu] ERROR: device " << d
                                      << " counter guard word " << badGuard << " overwritten; "
                                      << "counter[iB] ran past " << layout.payloadBytes << " bytes"
                                      << std::endl;
                        }
                        if(badSeg >= 0 || mismatches != 0 || badGuard >= 0)
                            rc = 1;
                    }
                }

                for(int d = 0; d < W; d++)
                {
                    HIP_CHECK_EXC(hipSetDevice(d));
                    HIP_CHECK_EXC(hipStreamDestroy(streams[d]));
                }
                HIP_CHECK_EXC(hipSetDevice(0));
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

            const bool prefill  = args["a2a-prefill"].as<bool>();
            const bool loopback = args["a2a-loopback"].as<bool>();
            const bool multigpu = args["a2a-multigpu"].as<bool>();
            const int  launches = std::max(1, args["a2a-loopback-launches"].as<int>());
#ifndef TENSILELITE_ENABLE_SDMA
            if(loopback || multigpu)
            {
                std::cerr << "[a2a-prefill] ERROR: this client was built without "
                             "TENSILELITE_ENABLE_SDMA; the loopback and multigpu arms have no "
                             "queues to hand the kernel. Rebuild with "
                             "-DTENSILELITE_ENABLE_SDMA=ON."
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
                if(prefill)
                {
                    int one = runA2APrefillForWorld(adapter, solutionIterator, hardware, *base, W);
                    if(one != 0)
                        rc = one;
                }
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
                    int one = runA2ALoopbackForWorld(
                        adapter, solutionIterator, hardware, *base, W, launches);
                    if(one != 0)
                        rc = one;
                }
                if(multigpu)
                {
                    if(!fusedA2AWorldSizeValid(W))
                    {
                        std::cerr << "[a2a-multigpu] ERROR: W=" << W
                                  << " has no peer group; the segment holds "
                                  << FUSED_A2A_MAX_RANKS << std::endl;
                        return 1;
                    }
                    int one = runA2AMultiGpuForWorld(
                        args, solutionIterator, hardware, *base, W, launches);
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
