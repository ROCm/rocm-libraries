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

#ifndef ROCWMMA_GEMM_TEST_DETAIL_KERNEL
#define ROCWMMA_GEMM_TEST_DETAIL_KERNEL

#include "device/kernel_device_func.hpp"
#include "gemm_kernel_base.hpp"
#include "helper_macros.hpp"

namespace rocwmma
{

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD = LayoutC>
    struct Kernel_PGR0_LB0_MP0_SB_NC final : public GemmKernelBase<BlockM,
                                                                   BlockN,
                                                                   BlockK,
                                                                   InputT,
                                                                   OutputT,
                                                                   ComputeT,
                                                                   LayoutA,
                                                                   LayoutB,
                                                                   LayoutC,
                                                                   LayoutD>
    {
    private:
        using Base = GemmKernelBase<BlockM,
                                    BlockN,
                                    BlockK,
                                    InputT,
                                    OutputT,
                                    ComputeT,
                                    LayoutA,
                                    LayoutB,
                                    LayoutC,
                                    LayoutD>;

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveSize, uint32_t ArchId>
        using TestGuard = gemm_PGR0_LB0_MP0_SB_NC_guard<BlockM,
                                                        BlockN,
                                                        BlockK,
                                                        InputT,
                                                        OutputT,
                                                        ComputeT,
                                                        TBlockX,
                                                        TBlockY,
                                                        WaveSize,
                                                        ArchId>;

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveSize, uint32_t ArchId>
        struct TestKernelFunc
        {
            static constexpr auto generate()
            {
                // Avoid attempting to reference kernel functions that haven't passed
                // predicate tests, as they won't be built!
                if constexpr(TestGuard<TBlockX, TBlockY, WaveSize, ArchId>::enableRun())
                {
                    return typename Base::KernelFunc(gemm_PGR0_LB0_MP0_SB_NC<BlockM,
                                                                             BlockN,
                                                                             BlockK,
                                                                             InputT,
                                                                             OutputT,
                                                                             ComputeT,
                                                                             LayoutA,
                                                                             LayoutB,
                                                                             LayoutC,
                                                                             LayoutD,
                                                                             TBlockX,
                                                                             TBlockY,
                                                                             WaveSize,
                                                                             ArchId>);
                }
                else
                {
                    return typename Base::KernelFunc(nullptr);
                }
            }
        };

    public:
        Kernel_PGR0_LB0_MP0_SB_NC() {}
        ~Kernel_PGR0_LB0_MP0_SB_NC() final {}

        bool checkQuirks() const final
        {
            return Base::checkQuirks() && Base::template dispatchGuard<TestGuard>();
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return Base::template dispatchKernelFunc<TestKernelFunc>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DETAIL_KERNEL
