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

#ifndef ROCWMMA_DETAIL_LOAD_STORE_MATRIX_COOP_SYNC_HPP
#define ROCWMMA_DETAIL_LOAD_STORE_MATRIX_COOP_SYNC_HPP

#include "../helper_macros.hpp"
#include "device/load_store_matrix_coop_sync.hpp"
#include "load_store_matrix_sync_test/detail/load_store_matrix_sync.hpp"

namespace rocwmma
{
    // Some adaptor structs to select and dispatch
    // schedulers in the kernel selection.
    enum IOScheduler_t : uint32_t
    {
        ROW_MAJOR_2D = 0u,
        COL_MAJOR_2D = 1u,
        ROW_SLICE_2D = 2u,
        COL_SLICE_2D = 3u,
        SINGLE       = 4u
    };

    // Wrap a selector with the given ID so that we can dispatch from the IOScheduler_t
    template <uint32_t TBlockX, uint32_t TBlockY, uint32_t>
    struct SchedulerSelector;

    template <uint32_t TBlockX, uint32_t TBlockY>
    struct SchedulerSelector<TBlockX, TBlockY, IOScheduler_t::ROW_MAJOR_2D>
    {
        using type = IOScheduler::RowMajor2d<TBlockX, TBlockY>;
    };

    template <uint32_t TBlockX, uint32_t TBlockY>
    struct SchedulerSelector<TBlockX, TBlockY, IOScheduler_t::COL_MAJOR_2D>
    {
        using type = IOScheduler::ColMajor2d<TBlockX, TBlockY>;
    };

    template <uint32_t TBlockX, uint32_t TBlockY>
    struct SchedulerSelector<TBlockX, TBlockY, IOScheduler_t::ROW_SLICE_2D>
    {
        using type = IOScheduler::RowSlice2d<TBlockX, TBlockY>;
    };

    template <uint32_t TBlockX, uint32_t TBlockY>
    struct SchedulerSelector<TBlockX, TBlockY, IOScheduler_t::COL_SLICE_2D>
    {
        using type = IOScheduler::ColSlice2d<TBlockX, TBlockY>;
    };

    template <uint32_t TBlockX, uint32_t TBlockY>
    struct SchedulerSelector<TBlockX, TBlockY, IOScheduler_t::SINGLE>
    {
        using type = IOScheduler::
            Single<TBlockX, TBlockY, IOScheduler::Single<TBlockX, TBlockY>::waveCount() - 1u>;
    };

    template <uint32_t TBlockX, uint32_t TBlockY, IOScheduler_t Scheduler>
    using SchedulerSelector_t = typename SchedulerSelector<TBlockX, TBlockY, Scheduler>::type;

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout, uint32_t Scheduler>
    struct LoadStoreMatrixCoopSyncKernelA final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {

            // Generate a nested switch case
            // TBlockY = [1, 2, 4, 8]
            // TBlockX = [32, 64, 128, 256, 512]
#define CASE_IMPL_ASSIGN2(TBLOCK_X, TBLOCK_Y) \
    return typename Base::KernelFunc(         \
        LoadStoreMatrixCoopSyncA<             \
            BlockM,                           \
            BlockN,                           \
            DataT,                            \
            Layout,                           \
            SchedulerSelector_t<TBLOCK_X, TBLOCK_Y, (IOScheduler_t)Scheduler>>);

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y) \
    ROCWMMA_SWITCH_BODY5_ARG2(         \
        Base::mTBlockX, CASE_IMPL_ASSIGN2, 32u, 64u, 128u, 256u, 512u, TBLOCK_Y)

#define DISPATCH_BODY \
    ROCWMMA_SWITCH_BODY4_ARG1(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, 8u)

            // Invoke dispatch
            DISPATCH_BODY

            // Silence warnings
            return typename Base::KernelFunc(nullptr);

// Clean up macro space
#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_TBLOCK_X
#undef DISPATCH_BODY
        }

    public:
        bool checkQuirks() const final
        {
            // Generate a nested switch case
            // TBlockY = [1, 2, 4, 8]
            // TBlockX = [32, 64, 128, 256, 512]
#define CASE_IMPL_ASSIGN2(TBLOCK_X, TBLOCK_Y) \
    return Base::checkQuirks()                \
           && CooperativePredicates<          \
               matrix_a,                      \
               BlockM,                        \
               BlockN,                        \
               DataT,                         \
               SchedulerSelector_t<TBLOCK_X, TBLOCK_Y, (IOScheduler_t)Scheduler>>::enable();

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y) \
    ROCWMMA_SWITCH_BODY5_ARG2(         \
        Base::mTBlockX, CASE_IMPL_ASSIGN2, 32u, 64u, 128u, 256u, 512u, TBLOCK_Y)

#define DISPATCH_BODY \
    ROCWMMA_SWITCH_BODY4_ARG1(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, 8u)

            // Invoke dispatch
            DISPATCH_BODY

            // Silence warnings
            return false;

// Clean up macro space
#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_TBLOCK_X
#undef DISPATCH_BODY
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout, uint32_t Scheduler>
    struct LoadStoreMatrixCoopSyncKernelB final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {

            // Generate a nested switch case
            // TBlockY = [1, 2, 4, 8]
            // TBlockX = [32, 64, 128, 256, 512]
#define CASE_IMPL_ASSIGN2(TBLOCK_X, TBLOCK_Y) \
    return typename Base::KernelFunc(         \
        LoadStoreMatrixCoopSyncB<             \
            BlockM,                           \
            BlockN,                           \
            DataT,                            \
            Layout,                           \
            SchedulerSelector_t<TBLOCK_X, TBLOCK_Y, (IOScheduler_t)Scheduler>>);

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y) \
    ROCWMMA_SWITCH_BODY5_ARG2(         \
        Base::mTBlockX, CASE_IMPL_ASSIGN2, 32u, 64u, 128u, 256u, 512u, TBLOCK_Y)

#define DISPATCH_BODY \
    ROCWMMA_SWITCH_BODY4_ARG1(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, 8u)

            // Invoke dispatch
            DISPATCH_BODY

            // Silence warnings
            return typename Base::KernelFunc(nullptr);

// Clean up macro space
#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_TBLOCK_X
#undef DISPATCH_BODY
        }

    public:
        bool checkQuirks() const final
        {
            // Generate a nested switch case
            // TBlockY = [1, 2, 4, 8]
            // TBlockX = [32, 64, 128, 256, 512]
#define CASE_IMPL_ASSIGN2(TBLOCK_X, TBLOCK_Y) \
    return Base::checkQuirks()                \
           && CooperativePredicates<          \
               matrix_b,                      \
               BlockM,                        \
               BlockN,                        \
               DataT,                         \
               SchedulerSelector_t<TBLOCK_X, TBLOCK_Y, (IOScheduler_t)Scheduler>>::enable();

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y) \
    ROCWMMA_SWITCH_BODY5_ARG2(         \
        Base::mTBlockX, CASE_IMPL_ASSIGN2, 32u, 64u, 128u, 256u, 512u, TBLOCK_Y)

#define DISPATCH_BODY \
    ROCWMMA_SWITCH_BODY4_ARG1(Base::mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, 8u)

            // Invoke dispatch
            DISPATCH_BODY

            // Silence warnings
            return false;

// Clean up macro space
#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_TBLOCK_X
#undef DISPATCH_BODY
        }
    };

    template <template <uint32_t, uint32_t, typename, typename, uint32_t> class KernelClass>
    struct LoadStoreMatrixCoopSyncGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT     = 0,
            BlockM    = 1,
            BlockN    = 2,
            Layout    = 3,
            Scheduler = 4
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = KernelClass<std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                              std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                              std::tuple_element_t<DataT, TestParamsT>, // DataT
                              std::tuple_element_t<Layout, TestParamsT>, // Layout
                              std::tuple_element_t<Scheduler, TestParamsT>::value // Scheduler
                              >;

            return std::make_shared<KernelT>();
        }
    };

    using LoadStoreMatrixCoopSyncGeneratorA
        = LoadStoreMatrixCoopSyncGenerator<LoadStoreMatrixCoopSyncKernelA>;
    using LoadStoreMatrixCoopSyncGeneratorB
        = LoadStoreMatrixCoopSyncGenerator<LoadStoreMatrixCoopSyncKernelB>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_LOAD_STORE_MATRIX_COOP_SYNC_HPP
