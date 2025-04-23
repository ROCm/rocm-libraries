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

#include "device/load_store_matrix_coop_sync.hpp"
#include "load_store_matrix_sync_test/detail/load_store_matrix_sync.hpp"

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixCoopSyncKernelA final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return
                typename Base::KernelFunc(LoadStoreMatrixCoopSyncA<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixCoopSyncKernelB final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return
                typename Base::KernelFunc(LoadStoreMatrixCoopSyncB<BlockM, BlockN, DataT, Layout>);
        }
    };

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename Layout>
    struct LoadStoreMatrixCoopSyncKernelAcc final
        : public LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>
    {
    private:
        using Base = LoadStoreMatrixSyncKernel<BlockM, BlockN, DataT, Layout>;

    protected:
        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(
                LoadStoreMatrixCoopSyncAcc<BlockM, BlockN, DataT, Layout>);
        }
    };

    using LoadStoreMatrixCoopSyncGeneratorA
        = LoadStoreMatrixSyncGenerator<LoadStoreMatrixCoopSyncKernelA>;
    using LoadStoreMatrixCoopSyncGeneratorB
        = LoadStoreMatrixSyncGenerator<LoadStoreMatrixCoopSyncKernelB>;
    using LoadStoreMatrixCoopSyncGeneratorAcc
        = LoadStoreMatrixSyncGenerator<LoadStoreMatrixCoopSyncKernelAcc>;

} // namespace rocwmma

#endif // ROCWMMA_DETAIL_LOAD_STORE_MATRIX_COOP_SYNC_HPP
