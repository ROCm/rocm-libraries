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

#ifndef DLRM_DOT_DETAIL_HPP
#define DLRM_DOT_DETAIL_HPP

#include "device/dlrm_dot_bwd.hpp"
#include "device/dlrm_dot_fwd.hpp"
#include "dlrm_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t TileSize, typename DataT>
    struct DlrmDotKernel final : public DlrmKernelBase<TileSize, DataT>
    {
    private:
        using Base = DlrmKernelBase<TileSize, DataT>;

    public:
        DlrmDotKernel() {}
        ~DlrmDotKernel() final {}

        typename Base::KernelFwdFunc kernelFwdImpl() const final
        {
            return typename Base::KernelFwdFunc(dlrmDotFwd<DataT, TileSize>);
        }

        typename Base::KernelBwdFunc kernelBwdImpl() const final
        {
            return typename Base::KernelBwdFunc(dlrmDotBwd<DataT, TileSize>);
        }

        typename Base::KernelTrilFunc kernelTrilImpl() const final
        {
            return typename Base::KernelTrilFunc(trilReconstruct<DataT>);
        }
    };

    // This is the GeneratorImpl class
    struct DlrmDotGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            DataT    = 0,
            TileSize = 1
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = DlrmDotKernel<std::tuple_element_t<TileSize, TestParamsT>::value,
                                          std::tuple_element_t<DataT, TestParamsT>>;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // DLRM_DOT_DETAIL_HPP
