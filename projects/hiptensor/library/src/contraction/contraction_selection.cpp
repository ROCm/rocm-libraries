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
#include <string>

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
    // Find the first solution in candidates whose kernel name matches the given string.
    // Used by ActorCriticSelection to perform a cross-platform-stable lookup.
    static ContractionSolution*
        findByKernelName(std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         std::string const&                                      kernelName)
    {
        for(auto const& [uid, solution] : candidates)
        {
            if(solution->kernelName() == kernelName)
                return solution;
        }

        return nullptr;
    }

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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 1, 2, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                // if (rank == 1 || (rank == 1 && (a_ms_ks_lengths[3] == 1 || b_ns_ks_lengths[3] == 1)))
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 32, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 1, 2, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 128, 128, 32, "
                                "16, 32, 32, 4, 4, 2, 2, 2, 2, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 64, 64, 32, 16, "
                                "32, 32, 4, 4, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 2, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 1, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 1, 1, 1, 1, 1>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 32, 32, 4, 4, 2, 2, 4, 4, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 64, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 128, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 1, 1, 1, 1, 1>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 1, 1, 1, 1, 1>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 2, 2, 1, 1, 1, 1>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
            auto        rank = getRank(a_ms_ks_strides);
            std::string unique_id;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 1, 1, 1, 1, 1, 1, 0, 0>";
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 2, 1, 2, 1, 1, 1, 1>";
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = "DeviceContractionMultipleD_Xdl_CShuffle<6, 6, 6, 256, 128, 64, "
                                "16, 16, 16, 2, 1, 2, 1, 1, 1, 1, 0>";
                }
            }

            *winner = findByKernelName(candidates, unique_id);
            return (*winner != nullptr) ? HIPTENSOR_STATUS_SUCCESS
                                        : HIPTENSOR_STATUS_EXECUTION_FAILED;
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
