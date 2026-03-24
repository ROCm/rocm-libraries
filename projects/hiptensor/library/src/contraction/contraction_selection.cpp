/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <numeric>

#ifndef CHECK_HIP_ALLOC
#define CHECK_HIP_ALLOC(status)               \
    if(status != hipSuccess)                  \
    {                                         \
        return HIPTENSOR_STATUS_ALLOC_FAILED; \
    }
#endif

#include <ck/stream_config.hpp>

#include "contraction_selection.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "performance.hpp"
#include "util.hpp"

namespace hiptensor
{
    hiptensorStatus_t bruteForceModel(ContractionSolution**              winner,
                                      std::vector<ContractionSolution*>& candidates,
                                      hiptensorDataType_t                typeA,
                                      std::vector<std::size_t> const&    a_ms_ks_lengths,
                                      std::vector<std::size_t> const&    a_ms_ks_strides,
                                      std::vector<int32_t> const&        a_ms_ks_modes,
                                      hiptensorDataType_t                typeB,
                                      std::vector<std::size_t> const&    b_ns_ks_lengths,
                                      std::vector<std::size_t> const&    b_ns_ks_strides,
                                      std::vector<int32_t> const&        b_ns_ks_modes,
                                      hiptensorDataType_t                typeD,
                                      std::vector<std::size_t> const&    d_ms_ns_lengths,
                                      std::vector<std::size_t> const&    d_ms_ns_strides,
                                      std::vector<int32_t> const&        d_ms_ns_modes,
                                      hiptensorDataType_t                typeE,
                                      std::vector<std::size_t> const&    e_ms_ns_lengths,
                                      std::vector<std::size_t> const&    e_ms_ns_strides,
                                      std::vector<int32_t> const&        e_ms_ns_modes,
                                      hiptensorComputeDescriptor_t       computeType,
                                      ContractionUnaryOps const&         unaryOps,
                                      const uint64_t                     workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementsFromLengths(a_ms_ks_lengths) * hiptensorDataTypeSize(typeA);
        auto sizeB = elementsFromLengths(b_ns_ks_lengths) * hiptensorDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementsFromLengths(d_ms_ns_lengths) * hiptensorDataTypeSize(typeD);
        }
        auto sizeE = elementsFromLengths(e_ms_ns_lengths) * hiptensorDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *wspace;

        /*
         * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
         * ```
         * alphaF = hiptensor::readVal<float>(
         *      alpha, convertToComputeType(HipTensorDataType_v<typename Traits::ComputeDataT>));
         * ```
         * Hence, the `alpha` and `bete` need to point to a ComputeData value
         */
        ScalarData alpha;
        ScalarData beta;
        if(computeType == HIPTENSOR_COMPUTE_DESC_C32F || computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            writeVal(&alpha, computeType, {computeType, 1.02, 1.03});
            writeVal(&beta, computeType, {computeType, 1.04, 1.05});
        }
        else
        {
            writeVal(&alpha, computeType, ScalarData(computeType, 1.02));
            writeVal(&beta, computeType, ScalarData(computeType, 1.03));
        }

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

        std::string          best_op_name;
        ContractionSolution* bestSolution = nullptr;
        PerfMetrics          bestMetrics  = {
            0,
            "",
            0,
            0,
            0,
        };

        std::vector<float> sol_times(candidates.size(), std::numeric_limits<float>::max());
        std::vector<int>   indices(candidates.size());
        std::iota(indices.begin(), indices.end(), 0);
        int idx = 0;
        for(auto* solution : candidates)
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();

            auto [errorCode, time] = (*solution)(&alpha,
                                                 A_d,
                                                 B_d,
                                                 &beta,
                                                 D_d,
                                                 E_d,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 e_ms_ns_modes,
                                                 unaryOps,
                                                 wspace,
                                                 workspaceSize,
                                                 StreamConfig{
                                                     nullptr, // stream id
                                                     true, // time_kernel
                                                     0, // log_level
                                                     options->coldRuns(), // cold_niters
                                                     options->hotRuns(), // nrepeat
                                                 });
            if(errorCode == HIPTENSOR_STATUS_SUCCESS && time > 0)
            {
                // Make sure to time the kernels
                int32_t m, n, k;
                std::tie(m, n, k) = solution->problemDims();
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = solution->problemBytes();

                PerfMetrics metrics = {
                    solution->uid(), // id
                    solution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                using hiptensor::Logger;
                auto& logger = Logger::instance();

                // Log brute force timings for actor critic training
                if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE)
                {
                    // Log Kernel performances access
                    char msg[256];
                    snprintf(msg,
                             sizeof(msg),
                             "KernelId: %zu, KernelName: %s, AvgTime: %0.3f ms",
                             solution->uid(),
                             solution->kernelName().c_str(),
                             time);

                    logger->logHeuristics("BRUTE_FORCE_KERNEL_PERF", msg);
                }

                if(metrics > bestMetrics)
                {
                    bestSolution = solution;
                    bestMetrics  = metrics;
                }

                sol_times[idx] = time;
            }

            idx++;
        }

        CHECK_HIP_ALLOC(hipFree(A_d));
        CHECK_HIP_ALLOC(hipFree(B_d));
        CHECK_HIP_ALLOC(hipFree(D_d));
        CHECK_HIP_ALLOC(hipFree(E_d));
        CHECK_HIP_ALLOC(hipFree(wspace));

        *winner = bestSolution;

        //Sort candidates based on performance (from fastest to slowest)
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return sol_times[i] < sol_times[j];
        });
        std::vector<ContractionSolution*> tmpCandidates = candidates;
        candidates.clear();
        for(auto idx : indices)
            candidates.push_back(tmpCandidates[idx]);

        if(bestSolution == nullptr)
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
        else
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16046312426561516674ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5651259715737336589ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5651259715737336589ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 16046312426561516674ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17447143014665713887ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16046312426561516674ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 9021620837589482599ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10097482900535040320ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6053663486226699267ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16046312426561516674ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 16046312426561516674ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5651259715737336589ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 16046312426561516674ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17447143014665713887ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16046312426561516674ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6053663486226699267ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10097482900535040320ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6053663486226699267ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12241437837959333440ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2317674114976786230ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2317674114976786230ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2317674114976786230ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12241437837959333440ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 11152060091307708334ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 872672380373754190ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 872672380373754190ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16476891743625221381ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 16476891743625221381ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 16476891743625221381ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16476891743625221381ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 58303249112943560ull;
                }
                // m1n1k1
                else if(rank == 1)
                // if (rank == 1 || (rank == 1 && (a_ms_ks_lengths[3] == 1 || b_ns_ks_lengths[3] == 1)))
                {
                    unique_id = 58303249112943560ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2303552229010777601ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 58303249112943560ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 58303249112943560ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 58303249112943560ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2303552229010777601ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8024078432480148721ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12894894255582471185ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17760782256758115565ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13058678487168027ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 9333825291905548205ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8024078432480148721ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12894894255582471185ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12894894255582471185ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17760782256758115565ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13058678487168027ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 9333825291905548205ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 9333825291905548205ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15452087623356707112ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 9967477699864925937ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14071475272156866885ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14071475272156866885ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15452087623356707112ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8307633941691601884ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 9344798352708026060ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 9344798352708026060ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 9344798352708026060ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 9344798352708026060ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 9344798352708026060ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 9344798352708026060ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 16299024124514902126ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 378062791888302715ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 76527422265261696ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 378062791888302715ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 378062791888302715ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 378062791888302715ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 378062791888302715ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 13825918879176996502ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13825918879176996502ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17141562253969597117ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17141562253969597117ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6384780398804323250ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 11208787066124811014ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 11208787066124811014ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14522095938220523368ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14522095938220523368ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14522095938220523368ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14522095938220523368ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 8251132190088736039ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 2897979232477761524ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2897979232477761524ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2897979232477761524ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2897979232477761524ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2897979232477761524ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2897979232477761524ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 13613206280884761703ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13613206280884761703ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13613206280884761703ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 4373449368168185126ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 4373449368168185126ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 4373449368168185126ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 4373449368168185126ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 4373449368168185126ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2008216990064456310ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 4373449368168185126ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13613206280884761703ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15116758930810193332ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float,
                                float,
                                float,
                                float,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15864809842584901464ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8067958629699904967ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15864809842584901464ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6775599605174985174ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 6775599605174985174ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5326563676026437938ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 8067958629699904967ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 8116863550692548667ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8116863550692548667ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8116863550692548667ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8116863550692548667ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 8116863550692548667ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8116863550692548667ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5794367356792942822ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17939389824758640014ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10640128726648594287ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5794367356792942822ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13933081369664111675ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2224053047801499357ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 3431382583157381293ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5422513160360085353ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3431382583157381293ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 3431382583157381293ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14915761978535949477ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14915761978535949477ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16870758234615651290ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14901158961446820896ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16870758234615651290ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8188562791036959263ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 16870758234615651290ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16870758234615651290ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 18207091374964962208ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 16948282955506101335ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16870758234615651290ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15355329505248522280ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14642257549075851915ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14642257549075851915ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12057130050439892271ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13038089902448627981ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12057130050439892271ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11269655469469274301ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11269655469469274301ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12057130050439892271ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 11269655469469274301ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 2143493311543532856ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2143493311543532856ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2143493311543532856ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2143493311543532856ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2143493311543532856ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2143493311543532856ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6406117030749216765ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8021137963958390646ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 3248584345341330494ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3879892272436099392ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7950787545240972863ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 4041813994497895944ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 4041813994497895944ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 4041813994497895944ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 4041813994497895944ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 4041813994497895944ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7591632339673577634ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 2054609181761357786ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 14145390177844245465ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14145390177844245465ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14145390177844245465ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14145390177844245465ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14145390177844245465ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14145390177844245465ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1688099565795560288ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 4348837698146370003ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1688099565795560288ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 4363356859752806590ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15330878641001915472ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 15330878641001915472ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 11537900932066889768ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8338926107119209426ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 11537900932066889768ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11537900932066889768ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11537900932066889768ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 11537900932066889768ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12959721676360111684ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10254320286859648634ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15705829219230515535ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12959721676360111684ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10254320286859648634ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10254320286859648634ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10254320286859648634ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1322366267556764247ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1322366267556764247ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1322366267556764247ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1322366267556764247ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1322366267556764247ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1322366267556764247ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 14051358583041094215ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 8503926755447648324ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8503926755447648324ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8503926755447648324ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8503926755447648324ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 8503926755447648324ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8503926755447648324ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    hiptensorStatus_t
        actorCriticModel(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         hiptensorComputeDescriptor_t                            computeType,
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
           && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == HIPTENSOR_C_32F
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == HIPTENSOR_C_64F
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14408154905494010489ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 6974095321173130927ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6974095321173130927ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16253689926652332996ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14408154905494010489ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14408154905494010489ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 6974095321173130927ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6974095321173130927ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 16253689926652332996ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16253689926652332996ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    // Acotor-Critic model for unary ops
    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7538467540961049787ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7538467540961049787ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7538467540961049787ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7538467540961049787ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7538467540961049787ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7538467540961049787ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5838574146849536063ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 9929579717278416296ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 9929579717278416296ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 9929579717278416296ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 9929579717278416296ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15431985258997757854ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 3415378554316130654ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 3415378554316130654ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 3415378554316130654ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 3415378554316130654ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3415378554316130654ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 3415378554316130654ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17177484621493380088ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17177484621493380088ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17177484621493380088ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8974719589220802400ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 8974719589220802400ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17177484621493380088ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 3753568868518805000ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3753568868518805000ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12765469494343230674ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16194744748728374153ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14950476381367017410ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12765469494343230674ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3753568868518805000ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12765469494343230674ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16194744748728374153ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 16194744748728374153ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14950476381367017410ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14950476381367017410ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10572640329933031377ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15510068347583301452ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10572640329933031377ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15510068347583301452ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10572640329933031377ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10572640329933031377ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 6276769470110182113ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5827523263153508957ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5827523263153508957ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5827523263153508957ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5827523263153508957ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6276769470110182113ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12481385650722299445ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 4574287328754767068ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12481385650722299445ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12481385650722299445ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12481385650722299445ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 4574287328754767068ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2817642069473326068ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1139053823940322184ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2817642069473326068ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1139053823940322184ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1139053823940322184ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2817642069473326068ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10482334011498030239ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8472864432528069731ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8472864432528069731ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8472864432528069731ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 8472864432528069731ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8472864432528069731ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 4617465351495394743ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1663480608868680521ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1663480608868680521ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1663480608868680521ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1663480608868680521ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1663480608868680521ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14762504979677123854ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14762504979677123854ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14762504979677123854ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14762504979677123854ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14762504979677123854ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14762504979677123854ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1158968124405133622ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 6681600076438905506ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1685602420862754469ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6681600076438905506ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 6681600076438905506ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6681600076438905506ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 4677594781166299396ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 4677594781166299396ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 4677594781166299396ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 4677594781166299396ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 4677594781166299396ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 4677594781166299396ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5675370401297283233ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 9469902973241888595ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 9469902973241888595ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 9469902973241888595ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 9469902973241888595ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 9469902973241888595ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5704881916563684892ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2062110410148341538ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2062110410148341538ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2062110410148341538ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2062110410148341538ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2062110410148341538ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1580714540240191281ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1580714540240191281ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1580714540240191281ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1580714540240191281ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1580714540240191281ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1580714540240191281ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12160779730984340837ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 12160779730984340837ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 12160779730984340837ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12160779730984340837ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 12160779730984340837ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12160779730984340837ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 6103881444342374783ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 3492198639380540417ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 6103881444342374783ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 3492198639380540417ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 3492198639380540417ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6103881444342374783ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5447850794718128443ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7828346908127446539ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5447850794718128443ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7828346908127446539ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5447850794718128443ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5447850794718128443ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15770210374284736011ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15770210374284736011ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15770210374284736011ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6463641982677566422ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 6463641982677566422ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15770210374284736011ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2158644421080291960ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2158644421080291960ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2158644421080291960ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1106420844725934876ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2158644421080291960ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2158644421080291960ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 770569218298543735ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5213105944240361468ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5213105944240361468ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5213105944240361468ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5213105944240361468ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5213105944240361468ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15689293195218612641ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15689293195218612641ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15689293195218612641ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15689293195218612641ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15689293195218612641ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15689293195218612641ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12362723444598556025ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2980379434373265690ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2980379434373265690ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2980379434373265690ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2980379434373265690ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2980379434373265690ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1818797200832435030ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 6509874686188425969ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 6509874686188425969ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 6509874686188425969ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 6509874686188425969ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 6509874686188425969ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10063938818899116719ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 16235788987998995137ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 16235788987998995137ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10063938818899116719ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 16235788987998995137ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16235788987998995137ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12567580538174379913ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15173184787179566326ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15173184787179566326ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15173184787179566326ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15173184787179566326ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15173184787179566326ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12567580538174379913ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14694634098178137439ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14694634098178137439ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 12567580538174379913ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14694634098178137439ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 12567580538174379913ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    hiptensorStatus_t
        actorCriticModelUnaryOps(ContractionSolution**                                   winner,
                                 std::unordered_map<size_t, ContractionSolution*> const& candidates,
                                 hiptensorDataType_t                                     typeA,
                                 std::vector<std::size_t> const& a_ms_ks_lengths,
                                 std::vector<std::size_t> const& a_ms_ks_strides,
                                 std::vector<int32_t> const&     a_ms_ks_modes,
                                 hiptensorDataType_t             typeB,
                                 std::vector<std::size_t> const& b_ns_ks_lengths,
                                 std::vector<std::size_t> const& b_ns_ks_strides,
                                 std::vector<int32_t> const&     b_ns_ks_modes,
                                 hiptensorDataType_t             typeD,
                                 std::vector<std::size_t> const& d_ms_ns_lengths,
                                 std::vector<std::size_t> const& d_ms_ns_strides,
                                 std::vector<int32_t> const&     d_ms_ns_modes,
                                 hiptensorDataType_t             typeE,
                                 std::vector<std::size_t> const& e_ms_ns_lengths,
                                 std::vector<std::size_t> const& e_ms_ns_strides,
                                 std::vector<int32_t> const&     e_ms_ns_modes,
                                 hiptensorComputeDescriptor_t    computeType,
                                 const uint64_t                  workspaceSize)
    {
        if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
           && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<_Float16,
                                                _Float16,
                                                _Float16,
                                                _Float16,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<_Float16,
                                                _Float16,
                                                _Float16,
                                                _Float16,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                _Float16>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                _Float16>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                hip_bfloat16>::selectWinner(winner,
                                                                            candidates,
                                                                            typeA,
                                                                            a_ms_ks_lengths,
                                                                            a_ms_ks_strides,
                                                                            a_ms_ks_modes,
                                                                            typeB,
                                                                            b_ns_ks_lengths,
                                                                            b_ns_ks_strides,
                                                                            b_ns_ks_modes,
                                                                            typeD,
                                                                            d_ms_ns_lengths,
                                                                            d_ms_ns_strides,
                                                                            d_ms_ns_modes,
                                                                            typeE,
                                                                            e_ms_ns_lengths,
                                                                            e_ms_ns_strides,
                                                                            e_ms_ns_modes,
                                                                            workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                hip_bfloat16>::selectWinner(winner,
                                                                            candidates,
                                                                            typeA,
                                                                            a_ms_ks_lengths,
                                                                            a_ms_ks_strides,
                                                                            a_ms_ks_modes,
                                                                            typeB,
                                                                            b_ns_ks_lengths,
                                                                            b_ns_ks_strides,
                                                                            b_ns_ks_modes,
                                                                            typeD,
                                                                            d_ms_ns_lengths,
                                                                            d_ms_ns_strides,
                                                                            d_ms_ns_modes,
                                                                            typeE,
                                                                            e_ms_ns_lengths,
                                                                            e_ms_ns_strides,
                                                                            e_ms_ns_modes,
                                                                            workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::SCALE,
                                                double>::selectWinner(winner,
                                                                      candidates,
                                                                      typeA,
                                                                      a_ms_ks_lengths,
                                                                      a_ms_ks_strides,
                                                                      a_ms_ks_modes,
                                                                      typeB,
                                                                      b_ns_ks_lengths,
                                                                      b_ns_ks_strides,
                                                                      b_ns_ks_modes,
                                                                      typeD,
                                                                      d_ms_ns_lengths,
                                                                      d_ms_ns_strides,
                                                                      d_ms_ns_modes,
                                                                      typeE,
                                                                      e_ms_ns_lengths,
                                                                      e_ms_ns_strides,
                                                                      e_ms_ns_modes,
                                                                      workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::BILINEAR,
                                                double>::selectWinner(winner,
                                                                      candidates,
                                                                      typeA,
                                                                      a_ms_ks_lengths,
                                                                      a_ms_ks_strides,
                                                                      a_ms_ks_modes,
                                                                      typeB,
                                                                      b_ns_ks_lengths,
                                                                      b_ns_ks_strides,
                                                                      b_ns_ks_modes,
                                                                      typeD,
                                                                      d_ms_ns_lengths,
                                                                      d_ms_ns_strides,
                                                                      d_ms_ns_modes,
                                                                      typeE,
                                                                      e_ms_ns_lengths,
                                                                      e_ms_ns_strides,
                                                                      e_ms_ns_modes,
                                                                      workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
