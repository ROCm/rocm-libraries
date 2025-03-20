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

#ifndef ROCWMMA_LAYOUT_TRAITS_INT_TEST_DETAIL_HPP
#define ROCWMMA_LAYOUT_TRAITS_INT_TEST_DETAIL_HPP

#include "device/layout_traits_int.hpp"
#include "helper_macros.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    // Wrapper into the actual device function
    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct LayoutTraitsIntKernel final : public UnitKernelBase<BlockM, BlockN, DataT, DataLayoutT>
    {
    private:
        using Base = UnitKernelBase<BlockM, BlockN, DataT, DataLayoutT>;

        template <uint32_t WaveSize, uint32_t ArchId>
        using TestGuard = FragSize_guard<BlockM, BlockN, DataT, DataLayoutT, WaveSize, ArchId>;

    public:
        LayoutTraitsIntKernel()        = default;
        ~LayoutTraitsIntKernel() final = default;

        void setupImpl(typename Base::DataStorage::ProblemSize const& probsize) final
        {
            // Need at least 1 element for the result
            auto& dataInstance = Base::DataStorage::instance();
            dataInstance->resizeStorage(probsize);

            dataInstance->hostOut().get()[0] = static_cast<DataT>(ERROR_VALUE);
            dataInstance->copyData(dataInstance->deviceOut(), dataInstance->hostOut(), 1);

            // Pass in warpSize from host to validate
            Base::mParam1 = static_cast<DataT>(Base::DeviceInfo::instance()->warpSize());
        }

        void validateResultsImpl() final
        {
            auto& dataInstance = Base::DataStorage::instance();

            // Cache current kernel result from device
            dataInstance->copyData(dataInstance->hostOut(), dataInstance->deviceOut(), 1);

            // Check the single output result
            Base::mValidationResult = (dataInstance->hostOut().get()[0] == DataT(SUCCESS_VALUE));
        }

        bool checkQuirks() const final
        {
            auto waveSize   = Base::DeviceInfo::instance()->warpSize();
            auto deviceArch = Base::DeviceInfo::instance()->getGcnArch();

            // The test guard for this class requires 2 values at runtime.
            auto dispatchGuard = [waveSize, deviceArch]() {
                bool dispatchResult = false;

#define CASE_IMPL_ASSIGN2(WAVE_SIZE, ARCH_ID) \
    dispatchResult = TestGuard<WAVE_SIZE, ARCH_ID>::enable();

#define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
    ROCWMMA_SWITCH_BODY2_ARG2(         \
        waveSize, CASE_IMPL_ASSIGN2, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

#define DISPATCH_GUARD_BODY                          \
    ROCWMMA_SWITCH_BODY9_ARG1(deviceArch,            \
                              SWITCH_BODY_WAVE_SIZE, \
                              HipDevice::GFX908,     \
                              HipDevice::GFX90A,     \
                              HipDevice::GFX942,     \
                              HipDevice::GFX950,     \
                              HipDevice::GFX1100,    \
                              HipDevice::GFX1101,    \
                              HipDevice::GFX1102,    \
                              HipDevice::GFX1200,    \
                              HipDevice::GFX1201)

                DISPATCH_GUARD_BODY

#undef CASE_IMPL_ASSIGN2
#undef SWITCH_BODY_WAVE_SIZE
#undef DISPATCH_GUARD_BODY

                return dispatchResult;
            };

            return Base::checkQuirks() && dispatchGuard();
        }

        typename Base::KernelFunc kernelImpl() const final
        {
            return typename Base::KernelFunc(
                layoutTraitsIntTest<BlockM, BlockN, DataT, DataLayoutT, MmaDim, SplitK>);
        }
    };

    // This is the GeneratorImpl class
    struct LayoutTraitsIntGenerator
    {
        // Indices to test parameters
        enum : uint32_t
        {
            BlockM      = 0,
            BlockN      = 1,
            DataT       = 2,
            DataLayoutT = 3,
            MmaDim      = 4,
            SplitK      = 5,
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            // Map GTest params to Kernel params
            using TestParamsT = std::tuple<Ts...>;
            using KernelT     = LayoutTraitsIntKernel<
                std::tuple_element_t<BlockM, TestParamsT>::value, // BlockM
                std::tuple_element_t<BlockN, TestParamsT>::value, // BlockN
                std::tuple_element_t<DataT, TestParamsT>, // DataT
                std::tuple_element_t<DataLayoutT, TestParamsT>, // DataLayout
                std::tuple_element_t<MmaDim, TestParamsT>::value, // MmaDim
                std::tuple_element_t<SplitK, TestParamsT>::value // SplitK
                >;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRAITS_TEST_DETAIL_HPP
