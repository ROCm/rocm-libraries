// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Single-GPU bring-up arm for the A2A-GEMM shard loop (FusedA2AMode=1),
// dispatched from main.cpp when the a2a-prefill option is set. Allocates and
// validates its own tensors rather than using DataInitialization /
// ReferenceValidator: B is W separate [nToken, k_local] segments
// (ldb = k_local), W times larger than B's TensorDescriptor accounts for.

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes_BFloat16.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "ProgramOptions.hpp"
#include "SolutionIterator.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
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
            };

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

                size_t mismatches = 0;
                double worstRel   = 0.0;
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

                std::cout << "[a2a-prefill] W=" << W << " mismatches=" << mismatches
                          << " worst-rel=" << worstRel << std::endl;
                return mismatches == 0 ? 0 : 1;
            }
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
            }

            std::cout << "[a2a-prefill] overall " << (rc == 0 ? "PASSED" : "FAILED") << std::endl;
            return rc;
        }

    } // namespace Client
} // namespace TensileLite
