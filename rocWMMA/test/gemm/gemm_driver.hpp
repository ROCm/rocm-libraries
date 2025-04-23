/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GEMM_DRIVER_HPP
#define GEMM_DRIVER_HPP

namespace rocwmma
{
    /* GemmDriver class:
    * This class implements a generic GEMM workflow to be used within kernel
    * device code. It implements workflow steps using meta-data information
    * from Global, Lds and Scheduler classes. It enables either Block-Level,
    * Wave-Level or Workgroup-Level kernel workflows based on ADL and input
    * fragment types.
    */

    namespace CooperativeGemm
    {
        template <typename GlobalMapping,
                  typename LdsMapping,
                  typename CoopSchedulerA,
                  typename CoopSchedulerB>
        struct GemmDriver
        {
            // Global fragment types
            using GRFragA = typename GlobalMapping::GRFragA;
            using GRFragB = typename GlobalMapping::GRFragB;

            // Mfma fragment types
            using MfmaFragA   = typename GlobalMapping::MfmaFragA;
            using MfmaFragB   = typename GlobalMapping::MfmaFragB;
            using MfmaFragC   = typename GlobalMapping::MfmaFragC;
            using MfmaFragD   = typename GlobalMapping::MfmaFragD;
            using MfmaFragAcc = typename GlobalMapping::MfmaFragAcc;

            // Local fragment types
            using LWFragA = typename LdsMapping::LWFragA;
            using LWFragB = typename LdsMapping::LWFragB;
            using LRFragA = typename LdsMapping::LRFragA;
            using LRFragB = typename LdsMapping::LRFragB;

            template <typename FragT>
            using MappingUtil = GetMappingUtil_t<FragT>;

            template <typename FragT>
            using GetIOTraitsFragA =
                typename GetCoopIOConfig_t<FragT, CoopSchedulerA::waveCount()>::IOTraits;

            template <typename FragT>
            using GetIOTraitsFragB =
                typename GetCoopIOConfig_t<FragT, CoopSchedulerB::waveCount()>::IOTraits;

            // Ensure that splitCounts are the same on both sides of
            // global fetch and local writes to match fragment data locality.
            constexpr static auto splitCountA
                = std::min((uint32_t)GetIOTraitsFragA<GRFragA>::IOCount,
                           (uint32_t)GetIOTraitsFragA<LWFragA>::IOCount);

            constexpr static auto splitCountB
                = std::min((uint32_t)GetIOTraitsFragB<GRFragB>::IOCount,
                           (uint32_t)GetIOTraitsFragB<LWFragB>::IOCount);

            static_assert(
                ((uint32_t)GetIOTraitsFragA<GRFragA>::IOCount % splitCountA == 0u)
                    && ((uint32_t)GetIOTraitsFragA<LWFragA>::IOCount % splitCountA == 0u),
                "splitCount A is not common divisor of GlobalRead and LocalWrite IOCounts");

            static_assert(
                ((uint32_t)GetIOTraitsFragB<GRFragB>::IOCount % splitCountB == 0u)
                    && ((uint32_t)GetIOTraitsFragB<LWFragB>::IOCount % splitCountB == 0u),
                "splitCount B is not common divisor of GlobalRead and LocalWrite IOCounts");

            ///
            /// Broadcast (fill) value
            ///

            // Broadcast value to fragment
            // Single or BlocksX * BlocksY frags
            template <typename FragT>
            __device__ static inline void fill(FragT& frag, GetDataType_t<FragT> value);
            template <typename FragT, uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void fill(FragT (&frags)[BlocksX][BlocksY],
                                               GetDataType_t<FragT> value);

            ///
            /// Global R/W
            ///

            // Global A/B reads in cooperative mode
            template <uint32_t BlocksX>
            __device__ static inline void globalReadCoopA(GRFragA (&fragsA)[BlocksX],
                                                          GetDataType_t<GRFragA> const* gAddrA,
                                                          uint32_t                      lda);
            __device__ static inline void globalReadCoopA(GRFragA&                      grFragA,
                                                          GetDataType_t<GRFragA> const* gAddrA,
                                                          uint32_t                      lda);

            template <uint32_t BlocksY>
            __device__ static inline void globalReadCoopB(GRFragB (&fragsB)[BlocksY],
                                                          GetDataType_t<GRFragB> const* gAddrB,
                                                          uint32_t                      ldb);
            __device__ static inline void globalReadCoopB(GRFragB&                      grFragB,
                                                          GetDataType_t<GRFragB> const* gAddrB,
                                                          uint32_t                      ldb);

            // Global C reads non-cooperative
            // Single or BlocksX * BlocksY frags
            template <uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void globalReadC(MfmaFragC (&fragC)[BlocksX][BlocksY],
                                                      GetDataType_t<MfmaFragC> const* gAddrC,
                                                      uint32_t                        ldc);
            __device__ static inline void
                globalReadC(MfmaFragC& fragC, GetDataType_t<MfmaFragC> const* gAddrC, uint32_t ldc);

            // Global D writes non-cooperative
            // Single or BlocksX * BlocksY frags
            template <uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void globalWriteD(GetDataType_t<MfmaFragD>* gAddrD,
                                                       MfmaFragD const (&fragsD)[BlocksX][BlocksY],
                                                       uint32_t ldd);
            __device__ static inline void globalWriteD(GetDataType_t<MfmaFragD>* gAddrD,
                                                       MfmaFragD const&          fragD,
                                                       uint32_t                  ldd);

            ///
            /// Local R/W
            ///

            // Local A/B writes in cooperative mode
            template <uint32_t BlocksX>
            __device__ static inline void localWriteCoopA(GetDataType_t<GRFragA>* ldsAddr,
                                                          GRFragA const (&grFragsA)[BlocksX],
                                                          uint32_t ldlds);
            __device__ static inline void localWriteCoopA(GetDataType_t<GRFragA>* ldsAddr,
                                                          GRFragA const&          grFragA,
                                                          uint32_t                ldlds);

            template <uint32_t BlocksY>
            __device__ static inline void localWriteCoopB(GetDataType_t<GRFragB>* ldsAddr,
                                                          GRFragB const (&grFragsB)[BlocksY],
                                                          uint32_t ldlds);
            __device__ static inline void localWriteCoopB(GetDataType_t<GRFragB>* ldsAddr,
                                                          GRFragB const&          grFragB,
                                                          uint32_t                ldlds);

            // Local A read non-cooperative
            // Single or BlocksX frags
            template <uint32_t BlocksX>
            __device__ static inline void localReadA(MfmaFragA (&fragsA)[BlocksX],
                                                     GetDataType_t<MfmaFragA> const* ldsAddrA,
                                                     uint32_t                        ldlds);
            __device__ static inline void localReadA(MfmaFragA&                      fragsA,
                                                     GetDataType_t<MfmaFragA> const* ldsAddrA,
                                                     uint32_t                        ldlds);

            // Local B read non-cooperative
            // Single or BlocksY frags
            template <uint32_t BlocksY>
            __device__ static inline void localReadB(MfmaFragB (&fragsB)[BlocksY],
                                                     GetDataType_t<MfmaFragB> const* ldsAddrB,
                                                     uint32_t                        ldlds);
            __device__ static inline void localReadB(MfmaFragB&                      fragsB,
                                                     GetDataType_t<MfmaFragB> const* ldsAddrB,
                                                     uint32_t                        ldlds);

            ///
            /// MFMA
            ///

            // Performs mfma
            // Single block, or BlocksX * BlocksY frags
            template <uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void mfma(MfmaFragAcc (&fragAccOut)[BlocksX][BlocksY],
                                               MfmaFragA const (&fragA)[BlocksX],
                                               MfmaFragB const (&fragB)[BlocksY],
                                               MfmaFragAcc const (&fragAccIn)[BlocksX][BlocksY]);
            __device__ static inline void mfma(MfmaFragAcc&       fragAccOut,
                                               MfmaFragA const&   fragA,
                                               MfmaFragB const&   fragB,
                                               MfmaFragAcc const& fragAccIn);

            ///
            /// Uniform fused multiply - add (FMA)
            ///

            // Performs D = alpha * acc + beta * C, where alpha, beta are uniform scalars
            template <uint32_t BlocksX, uint32_t BlocksY>
            __device__ static inline void
                                          uniformFma(MfmaFragD (&fragsD)[BlocksX][BlocksY],
                                                     GetDataType_t<MfmaFragAcc> alpha,
                                                     MfmaFragAcc const (&fragsAcc)[BlocksX][BlocksY],
                                                     GetDataType_t<MfmaFragAcc> beta,
                                                     MfmaFragC const (&fragsC)[BlocksX][BlocksY]);
            __device__ static inline void uniformFma(MfmaFragD&                 fragD,
                                                     GetDataType_t<MfmaFragAcc> alpha,
                                                     MfmaFragAcc const&         fragAcc,
                                                     GetDataType_t<MfmaFragAcc> beta,
                                                     MfmaFragC const&           fragC);

            ///
            /// Wave synchronization
            ///
            __device__ static inline void syncWorkgroup();

            template <int32_t priority = 0>
            __device__ static inline void prioritize_wavefront();

            template <int32_t mask = 0>
            __device__ static inline void sched_barrier();

            template <int32_t vmcnt = 0, int32_t lgkmcnt = 0>
            __device__ static inline void mem_barrier();

            template <int32_t vmcnt = 0>
            __device__ static inline void vector_mem_barrier();

            template <int32_t lgkmcnt = 0>
            __device__ static inline void lds_mem_barrier();

            __device__ static inline void scheduleKLoop();
        };

    } // namespace CooperativeGemm

} // namespace rocwmma

#include "gemm_driver_impl.hpp"

#endif // GEMM_DRIVER_HPP
