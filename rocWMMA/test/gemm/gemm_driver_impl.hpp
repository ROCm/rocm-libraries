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
#ifndef GEMM_DRIVER_IMPL_HPP
#define GEMM_DRIVER_IMPL_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{

    namespace CooperativeGemm
    {

        namespace detail
        {
            template <typename CoopSchedulerA,
                      typename CoopSchedulerB,
                      uint32_t SplitCountA,
                      uint32_t SplitCountB,
                      bool     UseCompileTimeConstants
                      = Schedule::WaveCountIsConstexpr<CoopSchedulerA>::value&&
                          Schedule::WaveCountIsConstexpr<CoopSchedulerB>::value>
            struct CoopApiSelector;

            template <typename CoopSchedulerA,
                      typename CoopSchedulerB,
                      uint32_t SplitCountA,
                      uint32_t SplitCountB>
            struct CoopApiSelector<CoopSchedulerA, CoopSchedulerB, SplitCountA, SplitCountB, true>
            {
                template <typename GRFragA>
                __device__ static inline void globalReadCoopA(GRFragA&                      grFragA,
                                                              GetDataType_t<GRFragA> const* gAddrA,
                                                              uint32_t                      lda)
                {
                    rocwmma::template load_matrix_coop_sync<CoopSchedulerA::waveCount(),
                                                            SplitCountA>(
                        grFragA, gAddrA, lda, CoopSchedulerA::waveIndex());
                }

                template <typename GRFragB>
                __device__ static inline void globalReadCoopB(GRFragB&                      grFragB,
                                                              GetDataType_t<GRFragB> const* gAddrB,
                                                              uint32_t                      ldb)
                {
                    rocwmma::template load_matrix_coop_sync<CoopSchedulerB::waveCount(),
                                                            SplitCountB>(
                        grFragB, gAddrB, ldb, CoopSchedulerB::waveIndex());
                }

                template <typename LWFragA>
                __device__ static inline void localWriteCoopA(GetDataType_t<LWFragA>* ldsAddr,
                                                              LWFragA const&          lwFragA,
                                                              uint32_t                ldlds)
                {
                    rocwmma::template store_matrix_coop_sync<CoopSchedulerA::waveCount(),
                                                             SplitCountA>(
                        ldsAddr, lwFragA, ldlds, CoopSchedulerA::waveIndex());
                }

                template <typename LWFragB>
                __device__ static inline void localWriteCoopB(GetDataType_t<LWFragB>* ldsAddr,
                                                              LWFragB const&          lwFragB,
                                                              uint32_t                ldlds)
                {
                    rocwmma::template store_matrix_coop_sync<CoopSchedulerB::waveCount(),
                                                             SplitCountB>(
                        ldsAddr, lwFragB, ldlds, CoopSchedulerB::waveIndex());
                }
            };

            template <typename CoopSchedulerA,
                      typename CoopSchedulerB,
                      uint32_t SplitCountA,
                      uint32_t SplitCountB>
            struct CoopApiSelector<CoopSchedulerA, CoopSchedulerB, SplitCountA, SplitCountB, false>
            {
                template <typename GRFragA>
                __device__ static inline void globalReadCoopA(GRFragA&                      grFragA,
                                                              GetDataType_t<GRFragA> const* gAddrA,
                                                              uint32_t                      lda)
                {
                    rocwmma::load_matrix_coop_sync(grFragA,
                                                   gAddrA,
                                                   lda,
                                                   CoopSchedulerA::waveIndex(),
                                                   CoopSchedulerA::waveCount(),
                                                   SplitCountA);
                }

                template <typename GRFragB>
                __device__ static inline void globalReadCoopB(GRFragB&                      grFragB,
                                                              GetDataType_t<GRFragB> const* gAddrB,
                                                              uint32_t                      ldb)
                {
                    rocwmma::load_matrix_coop_sync(grFragB,
                                                   gAddrB,
                                                   ldb,
                                                   CoopSchedulerB::waveIndex(),
                                                   CoopSchedulerB::waveCount(),
                                                   SplitCountB);
                }

                template <typename LWFragA>
                __device__ static inline void localWriteCoopA(GetDataType_t<LWFragA>* ldsAddr,
                                                              LWFragA const&          lwFragA,
                                                              uint32_t                ldlds)
                {
                    rocwmma::store_matrix_coop_sync(ldsAddr,
                                                    lwFragA,
                                                    ldlds,
                                                    CoopSchedulerA::waveIndex(),
                                                    CoopSchedulerA::waveCount(),
                                                    SplitCountA);
                }

                template <typename LWFragB>
                __device__ static inline void localWriteCoopB(GetDataType_t<LWFragB>* ldsAddr,
                                                              LWFragB const&          lwFragB,
                                                              uint32_t                ldlds)
                {
                    rocwmma::store_matrix_coop_sync(ldsAddr,
                                                    lwFragB,
                                                    ldlds,
                                                    CoopSchedulerB::waveIndex(),
                                                    CoopSchedulerB::waveCount(),
                                                    SplitCountB);
                }
            };
        }

#define GemmDriverT \
    typename GlobalMapping, typename LdsMapping, typename CoopSchedulerA, typename CoopSchedulerB

#define GemmDriverT_impl GlobalMapping, LdsMapping, CoopSchedulerA, CoopSchedulerB

        template <GemmDriverT>
        template <uint32_t BlocksX>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopA(
            GRFragA (&grFragsA)[BlocksX], GetDataType_t<GRFragA> const* gAddrA, uint32_t lda)
        {
            auto blockOffset = MappingUtil<GRFragA>::dataOffset(GlobalMapping::blockOffsetA(), lda);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                globalReadCoopA(grFragsA[i], gAddrA + i * blockOffset, lda);
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopA(
            GRFragA& grFragA, GetDataType_t<GRFragA> const* gAddrA, uint32_t lda)
        {
            using CoopApiSelector
                = detail::CoopApiSelector<CoopSchedulerA, CoopSchedulerB, splitCountA, splitCountB>;
            CoopApiSelector::globalReadCoopA(grFragA, gAddrA, lda);
        }

        template <GemmDriverT>
        template <uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopB(
            GRFragB (&grFragsB)[BlocksY], GetDataType_t<GRFragB> const* gAddrB, uint32_t ldb)
        {
            auto blockOffset = MappingUtil<GRFragB>::dataOffset(GlobalMapping::blockOffsetB(), ldb);
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                globalReadCoopB(grFragsB[i], gAddrB + i * blockOffset, ldb);
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadCoopB(
            GRFragB& grFragB, GetDataType_t<GRFragB> const* gAddrB, uint32_t ldb)
        {
            using CoopApiSelector
                = detail::CoopApiSelector<CoopSchedulerA, CoopSchedulerB, splitCountA, splitCountB>;
            CoopApiSelector::globalReadCoopB(grFragB, gAddrB, ldb);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopA(
            GetDataType_t<GRFragA>* ldsAddr, GRFragA const (&grFragsA)[BlocksX], uint32_t ldlds)
        {
            auto blockOffset = MappingUtil<LWFragA>::dataOffset(LdsMapping::blockOffsetA(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                localWriteCoopA(ldsAddr + i * blockOffset, grFragsA[i], ldlds);
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopA(
            GetDataType_t<GRFragA>* ldsAddr, GRFragA const& grFragA, uint32_t ldlds)
        {
            using CoopApiSelector
                = detail::CoopApiSelector<CoopSchedulerA, CoopSchedulerB, splitCountA, splitCountB>;
            CoopApiSelector::localWriteCoopA(
                ldsAddr,
                LdsMapping::template formatLWFragA<CoopSchedulerA::waveCount()>(grFragA),
                ldlds);
        }

        template <GemmDriverT>
        template <uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopB(
            GetDataType_t<GRFragB>* ldsAddr, GRFragB const (&grFragsB)[BlocksY], uint32_t ldlds)
        {
            auto blockOffset = MappingUtil<LWFragB>::dataOffset(LdsMapping::blockOffsetB(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                localWriteCoopB(ldsAddr + i * blockOffset, grFragsB[i], ldlds);
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localWriteCoopB(
            GetDataType_t<GRFragB>* ldsAddr, GRFragB const& grFragB, uint32_t ldlds)
        {
            using CoopApiSelector
                = detail::CoopApiSelector<CoopSchedulerA, CoopSchedulerB, splitCountA, splitCountB>;
            CoopApiSelector::localWriteCoopB(
                ldsAddr,
                LdsMapping::template formatLWFragB<CoopSchedulerB::waveCount()>(grFragB),
                ldlds);
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadA(
            MfmaFragA& fragsA, GetDataType_t<MfmaFragA> const* ldsAddrA, uint32_t ldlds)
        {
            LRFragA lrFragA;
            rocwmma::load_matrix_sync(lrFragA, ldsAddrA, ldlds);
            fragsA.mStorage = LdsMapping::formatLRFragA(lrFragA).mStorage;
        }

        template <GemmDriverT>
        template <uint32_t BlocksX>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadA(
            MfmaFragA (&fragsA)[BlocksX], GetDataType_t<MfmaFragA> const* ldsAddrA, uint32_t ldlds)
        {
            auto blockStep = MappingUtil<LRFragA>::dataOffset(LdsMapping::blockOffsetA(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                localReadA(fragsA[i], ldsAddrA, ldlds);
                ldsAddrA += blockStep;
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadB(
            MfmaFragB& fragsB, GetDataType_t<MfmaFragB> const* ldsAddrB, uint32_t ldlds)
        {
            LRFragB lrFragB;
            rocwmma::load_matrix_sync(lrFragB, ldsAddrB, ldlds);
            fragsB.mStorage = LdsMapping::formatLRFragB(lrFragB).mStorage;
        }

        template <GemmDriverT>
        template <uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::localReadB(
            MfmaFragB (&fragsB)[BlocksY], GetDataType_t<MfmaFragB> const* ldsAddrB, uint32_t ldlds)
        {
            auto blockStep = MappingUtil<LRFragB>::dataOffset(LdsMapping::blockOffsetB(), ldlds);
#pragma unroll
            for(int i = 0; i < BlocksY; i++)
            {
                localReadB(fragsB[i], ldsAddrB, ldlds);
                ldsAddrB += blockStep;
            }
        }

        template <GemmDriverT>
        template <typename FragT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::fill(FragT&               frag,
                                                                  GetDataType_t<FragT> value)
        {
            rocwmma::fill_fragment(frag, value);
        }

        template <GemmDriverT>
        template <typename FragT, uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::fill(FragT (&frags)[BlocksX][BlocksY],
                                                                  GetDataType_t<FragT> value)
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    fill(frags[i][j], value);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::mfma(MfmaFragAcc&       fragAccOut,
                                                                  MfmaFragA const&   fragA,
                                                                  MfmaFragB const&   fragB,
                                                                  MfmaFragAcc const& fragAccIn)
        {
            rocwmma::mma_sync(fragAccOut, fragA, fragB, fragAccIn);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void
            GemmDriver<GemmDriverT_impl>::mfma(MfmaFragAcc (&fragAccOut)[BlocksX][BlocksY],
                                               MfmaFragA const (&fragA)[BlocksX],
                                               MfmaFragB const (&fragB)[BlocksY],
                                               MfmaFragAcc const (&fragAccIn)[BlocksX][BlocksY])
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    mfma(fragAccOut[i][j], fragA[i], fragB[j], fragAccIn[i][j]);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalReadC(
            MfmaFragC& fragC, GetDataType_t<MfmaFragC> const* gAddrC, uint32_t ldc)
        {
            rocwmma::load_matrix_sync(fragC, gAddrC, ldc);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void
            GemmDriver<GemmDriverT_impl>::globalReadC(MfmaFragC (&fragC)[BlocksX][BlocksY],
                                                      GetDataType_t<MfmaFragC> const* gAddrC,
                                                      uint32_t                        ldc)
        {
            auto blockStepX
                = MappingUtil<MfmaFragC>::dataOffset(GlobalMapping::blockOffsetA(), ldc);
            auto blockStepY
                = MappingUtil<MfmaFragC>::dataOffset(GlobalMapping::blockOffsetB(), ldc);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                auto offsetY = 0u;
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    globalReadC(fragC[i][j], gAddrC + offsetY, ldc);

                    offsetY += blockStepY;
                }
                gAddrC += blockStepX;
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::globalWriteD(
            GetDataType_t<MfmaFragD>* gAddrD, MfmaFragD const& fragD, uint32_t ldd)
        {
            rocwmma::store_matrix_sync(gAddrD, fragD, ldd);
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void
            GemmDriver<GemmDriverT_impl>::globalWriteD(GetDataType_t<MfmaFragD>* gAddrD,
                                                       MfmaFragD const (&fragsD)[BlocksX][BlocksY],
                                                       uint32_t ldd)
        {
            auto blockStepX
                = MappingUtil<MfmaFragD>::dataOffset(GlobalMapping::blockOffsetA(), ldd);
            auto blockStepY
                = MappingUtil<MfmaFragD>::dataOffset(GlobalMapping::blockOffsetB(), ldd);
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
                auto offsetY = 0u;
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    globalWriteD(gAddrD + offsetY, fragsD[i][j], ldd);

                    offsetY += blockStepY;
                }
                gAddrD += blockStepX;
            }
        }

        template <GemmDriverT>
        __device__ inline void
            GemmDriver<GemmDriverT_impl>::uniformFma(MfmaFragD&                 fragD,
                                                     GetDataType_t<MfmaFragAcc> alpha,
                                                     MfmaFragAcc const&         fragAcc,
                                                     GetDataType_t<MfmaFragAcc> beta,
                                                     MfmaFragC const&           fragC)
        {
            // Split into chunks to not overwhelm the register capacity
            constexpr uint32_t ChunkSize = min(8u, MfmaFragD::size());
            for(int i = 0; i < MfmaFragD::size() / ChunkSize; i++)
            {
                for(int j = 0; j < ChunkSize; j++)
                {
                    auto idx = i * ChunkSize + j;
                    // Perform computation in ComputeT and cast back to OutputT
                    fragD.x[idx] = static_cast<GetDataType_t<MfmaFragD>>(
                        alpha * fragAcc.x[idx]
                        + beta * static_cast<GetDataType_t<MfmaFragAcc>>(fragC.x[idx]));
                }

            }
        }

        template <GemmDriverT>
        template <uint32_t BlocksX, uint32_t BlocksY>
        __device__ inline void GemmDriver<GemmDriverT_impl>::uniformFma(
            MfmaFragD (&fragsD)[BlocksX][BlocksY],
            GetDataType_t<MfmaFragAcc> alpha,
            MfmaFragAcc const (&fragsAcc)[BlocksX][BlocksY],
            GetDataType_t<MfmaFragAcc> beta,
            MfmaFragC const (&fragsC)[BlocksX][BlocksY])
        {
#pragma unroll
            for(int i = 0; i < BlocksX; i++)
            {
#pragma unroll
                for(int j = 0; j < BlocksY; j++)
                {
                    uniformFma(fragsD[i][j], alpha, fragsAcc[i][j], beta, fragsC[i][j]);
                }
            }
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::syncWorkgroup()
        {
            rocwmma::synchronize_workgroup();
        }

        template <GemmDriverT>
        template <int32_t priority>
        __device__ inline void GemmDriver<GemmDriverT_impl>::prioritize_wavefront()
        {
            using SetPrio = SetPrio<priority>;
            SetPrio::exec();
        }

        template <GemmDriverT>
        template <int32_t mask>
        __device__ inline void GemmDriver<GemmDriverT_impl>::sched_barrier()
        {
            using SchedBarrier = SchedBarrier<mask>;
            SchedBarrier::exec();
        }

        template <GemmDriverT>
        template <int32_t vmcnt, int32_t lgkmcnt>
        __device__ inline void GemmDriver<GemmDriverT_impl>::mem_barrier()
        {
            using Waitcnt = Waitcnt<vmcnt, lgkmcnt>;
            Waitcnt::exec();
        }

        template <GemmDriverT>
        template <int32_t vmcnt>
        __device__ inline void GemmDriver<GemmDriverT_impl>::vector_mem_barrier()
        {
            using WaitVmcnt = WaitVmcnt<vmcnt>;
            WaitVmcnt::exec();
        }

        template <GemmDriverT>
        template <int32_t lgkmcnt>
        __device__ inline void GemmDriver<GemmDriverT_impl>::lds_mem_barrier()
        {
            using WaitLgkmcnt = WaitLgkmcnt<lgkmcnt>;
            WaitLgkmcnt::exec();
        }

        template <GemmDriverT>
        __device__ inline void GemmDriver<GemmDriverT_impl>::scheduleKLoop()
        {
            // K-loop rough scheduling:
            // 1. GR
            // 2. LR
            // 3. Mma
            // 4. LW
            SchedBarrierVmemRead::exec();
            SchedBarrierDsRead::exec();
            SchedBarrierMma::exec();
            SchedBarrierDsWrite::exec();
        }

#undef GemmDriverT
#undef GemmDriverT_impl

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_DRIVER_IMPL_HPP
